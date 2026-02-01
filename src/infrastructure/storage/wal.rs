use crate::domain::model::{Record, StoredRecord};
use crate::infrastructure::index::hnsw::HnswIndex;
use crate::infrastructure::storage::file as storage_file;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC_V2: &[u8; 8] = b"MUSUBIW2";
const MAGIC_V1: &[u8; 8] = b"MUSUBIW1";
const VERSION: u32 = 2;

const OP_INSERT: u8 = 1;
const OP_UPDATE: u8 = 2;
const OP_DELETE: u8 = 3;

/// WAL operation for replay
#[derive(Debug, Clone)]
pub enum WalOp {
    Insert {
        id: String,
        record: Record,
        embedding: Vec<f32>,
    },
    Update {
        id: String,
        record: Record,
        embedding: Vec<f32>,
    },
    Delete {
        id: String,
    },
}

/// Configuration for WAL
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub path: PathBuf,
    pub max_bytes: Option<u64>,
    pub max_records: Option<usize>,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("hnsw.wal"),
            max_bytes: None,
            max_records: None,
        }
    }
}

pub struct WalWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    record_count: usize,
}

impl WalWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        let mut record_count = 0;

        if path.exists() && path.metadata()?.len() > 0 {
            let mut reader = BufReader::new(File::open(path)?);
            let mut magic = [0u8; 8];
            reader.read_exact(&mut magic)?;

            if &magic == MAGIC_V1 {
                // V1 WAL detected - cannot auto-migrate because v1 only stored vectors
                // without record metadata. User must verify records.jsonl is complete
                // and manually delete the v1 WAL file.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "WAL v1 detected at {:?}. V1 format cannot be replayed (no record metadata). \
                         Please verify records.jsonl is complete, then delete the WAL file to proceed.",
                        path
                    ),
                ));
            } else if &magic != MAGIC_V2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid wal magic: expected {:?}, got {:?}", MAGIC_V2, magic),
                ));
            }

            let version = read_u32(&mut reader)?;
            if version != VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported wal version: {}", version),
                ));
            }
            // Count existing records
            record_count = count_records(&mut reader)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        if path.metadata()?.len() == 0 {
            writer.write_all(MAGIC_V2)?;
            write_u32(&mut writer, VERSION)?;
            writer.flush()?;
        }

        Ok(Self {
            writer,
            path: path.to_path_buf(),
            record_count,
        })
    }

    /// Append an INSERT operation to WAL
    pub fn append_insert(&mut self, stored: &StoredRecord) -> io::Result<()> {
        self.writer.write_all(&[OP_INSERT])?;
        write_string(&mut self.writer, &stored.record.id)?;
        write_record(&mut self.writer, &stored.record)?;
        write_embedding(&mut self.writer, &stored.embedding)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.record_count += 1;
        Ok(())
    }

    /// Append an UPDATE operation to WAL
    pub fn append_update(&mut self, stored: &StoredRecord) -> io::Result<()> {
        self.writer.write_all(&[OP_UPDATE])?;
        write_string(&mut self.writer, &stored.record.id)?;
        write_record(&mut self.writer, &stored.record)?;
        write_embedding(&mut self.writer, &stored.embedding)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.record_count += 1;
        Ok(())
    }

    /// Append a DELETE operation to WAL
    pub fn append_delete(&mut self, id: &str) -> io::Result<()> {
        self.writer.write_all(&[OP_DELETE])?;
        write_string(&mut self.writer, id)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.record_count += 1;
        Ok(())
    }

    /// Get the current file size in bytes
    pub fn file_size(&self) -> io::Result<u64> {
        self.path.metadata().map(|m| m.len())
    }

    /// Get the number of records written to WAL
    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Get the WAL path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if WAL should be rotated based on config
    pub fn should_rotate(&self, config: &WalConfig) -> io::Result<bool> {
        if let Some(max_bytes) = config.max_bytes {
            if self.file_size()? >= max_bytes {
                return Ok(true);
            }
        }
        if let Some(max_records) = config.max_records {
            if self.record_count >= max_records {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Truncate the WAL file (after snapshot is saved)
    pub fn truncate(&mut self) -> io::Result<()> {
        drop(std::mem::replace(
            &mut self.writer,
            BufWriter::new(File::create(&self.path)?),
        ));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC_V2)?;
        write_u32(&mut writer, VERSION)?;
        writer.flush()?;
        writer.get_ref().sync_data()?;
        self.writer = writer;
        self.record_count = 0;
        Ok(())
    }
}

/// Replay WAL operations into records and collect operations for index rebuild
pub fn replay<P: AsRef<Path>>(path: P) -> io::Result<Vec<WalOp>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        return Ok(Vec::new());
    }

    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    if &magic == MAGIC_V1 {
        // V1 WAL cannot be replayed (no record metadata).
        // User must verify data integrity and manually delete the WAL file.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "WAL v1 detected at {:?}. V1 format cannot be replayed (no record metadata). \
                 Please verify records.jsonl is complete, then delete the WAL file to proceed.",
                path
            ),
        ));
    } else if &magic != MAGIC_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid wal magic: expected {:?}, got {:?}", MAGIC_V2, magic),
        ));
    }

    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported wal version: {}", version),
        ));
    }

    let mut ops = Vec::new();
    loop {
        let mut op = [0u8; 1];
        match reader.read_exact(&mut op) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }

        match op[0] {
            OP_INSERT => {
                let id = read_string(&mut reader)?;
                let record = read_record(&mut reader, id.clone())?;
                let embedding = read_embedding(&mut reader)?;
                ops.push(WalOp::Insert {
                    id,
                    record,
                    embedding,
                });
            }
            OP_UPDATE => {
                let id = read_string(&mut reader)?;
                let record = read_record(&mut reader, id.clone())?;
                let embedding = read_embedding(&mut reader)?;
                ops.push(WalOp::Update {
                    id,
                    record,
                    embedding,
                });
            }
            OP_DELETE => {
                let id = read_string(&mut reader)?;
                ops.push(WalOp::Delete { id });
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown wal op: {}", op[0]),
                ))
            }
        }
    }

    Ok(ops)
}

/// Apply WAL operations to records list using upsert semantics
/// - INSERT: replaces existing record if ID exists, otherwise appends
/// - UPDATE: replaces existing record if ID exists, otherwise appends (upsert)
/// - DELETE: removes record if ID exists, no-op otherwise
/// Returns true if any changes were made
pub fn apply_ops_to_records(ops: Vec<WalOp>, records: &mut Vec<StoredRecord>) -> bool {
    if ops.is_empty() {
        return false;
    }

    for op in ops {
        match op {
            WalOp::Insert {
                id,
                record,
                embedding,
            } => {
                // Upsert: replace if exists, insert if not
                if let Some(pos) = records.iter().position(|r| r.record.id == id) {
                    records[pos] = StoredRecord { record, embedding };
                } else {
                    records.push(StoredRecord { record, embedding });
                }
            }
            WalOp::Update {
                id,
                record,
                embedding,
            } => {
                // Upsert: replace if exists, insert if not
                if let Some(pos) = records.iter().position(|r| r.record.id == id) {
                    records[pos] = StoredRecord { record, embedding };
                } else {
                    records.push(StoredRecord { record, embedding });
                }
            }
            WalOp::Delete { id } => {
                // Remove if exists, no-op otherwise
                if let Some(pos) = records.iter().position(|r| r.record.id == id) {
                    records.remove(pos);
                }
            }
        }
    }

    true
}

/// Load index from snapshot, then apply WAL operations
pub fn load_with_wal<P: AsRef<Path>>(
    snapshot_path: P,
    _wal_path: P,
    m: usize,
    ef_construction: usize,
) -> io::Result<HnswIndex> {
    let index = if snapshot_path.as_ref().exists() {
        storage_file::load(snapshot_path)?
    } else {
        HnswIndex::new(m, ef_construction)
    };
    // Note: For full record-based WAL replay, use replay() + apply_ops_to_records()
    // This function is kept for backward compatibility with vector-only operations
    Ok(index)
}

// Count records in the WAL file (for resuming append)
fn count_records<R: Read>(reader: &mut R) -> io::Result<usize> {
    let mut count = 0;
    loop {
        let mut op = [0u8; 1];
        match reader.read_exact(&mut op) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }

        match op[0] {
            OP_INSERT | OP_UPDATE => {
                let id = read_string(reader)?;
                let _ = read_record(reader, id)?;
                let _ = read_embedding(reader)?;
                count += 1;
            }
            OP_DELETE => {
                let _ = read_string(reader)?;
                count += 1;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown wal op: {}", op[0]),
                ))
            }
        }
    }
    Ok(count)
}

// Helper functions for reading/writing primitive types

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_f32<W: Write>(writer: &mut W, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn write_string<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u32(writer, bytes.len() as u32)?;
    writer.write_all(bytes)
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = read_u32(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_optional_string<W: Write>(writer: &mut W, s: &Option<String>) -> io::Result<()> {
    match s {
        Some(s) => {
            writer.write_all(&[1])?;
            write_string(writer, s)
        }
        None => writer.write_all(&[0]),
    }
}

fn read_optional_string<R: Read>(reader: &mut R) -> io::Result<Option<String>> {
    let mut flag = [0u8; 1];
    reader.read_exact(&mut flag)?;
    if flag[0] == 0 {
        Ok(None)
    } else {
        read_string(reader).map(Some)
    }
}

fn write_record<W: Write>(writer: &mut W, record: &Record) -> io::Result<()> {
    // Write record fields as optional strings (excluding id which is written separately)
    write_optional_string(writer, &record.title)?;
    write_optional_string(writer, &record.body)?;
    write_optional_string(writer, &record.source)?;
    write_optional_string(writer, &record.updated_at)?;
    write_optional_string(writer, &record.tags)
}

fn read_record<R: Read>(reader: &mut R, id: String) -> io::Result<Record> {
    let title = read_optional_string(reader)?;
    let body = read_optional_string(reader)?;
    let source = read_optional_string(reader)?;
    let updated_at = read_optional_string(reader)?;
    let tags = read_optional_string(reader)?;
    Ok(Record {
        id,
        title,
        body,
        source,
        updated_at,
        tags,
    })
}

fn write_embedding<W: Write>(writer: &mut W, embedding: &[f32]) -> io::Result<()> {
    write_u32(writer, embedding.len() as u32)?;
    for &v in embedding {
        write_f32(writer, v)?;
    }
    Ok(())
}

fn read_embedding<R: Read>(reader: &mut R) -> io::Result<Vec<f32>> {
    let dim = read_u32(reader)? as usize;
    let mut embedding = Vec::with_capacity(dim);
    for _ in 0..dim {
        embedding.push(read_f32(reader)?);
    }
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        path
    }

    #[test]
    fn test_wal_insert_replay() {
        let wal_path = temp_path("wal_insert");

        let record = Record {
            id: "doc-1".to_string(),
            title: Some("Title".to_string()),
            body: Some("Body text".to_string()),
            source: None,
            updated_at: None,
            tags: Some("tag1,tag2".to_string()),
        };
        let stored = StoredRecord {
            record,
            embedding: vec![1.0, 0.0, 0.5],
        };

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&stored).unwrap();
            assert_eq!(wal.record_count(), 1);
        }

        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            WalOp::Insert {
                id,
                record,
                embedding,
            } => {
                assert_eq!(id, "doc-1");
                assert_eq!(record.title, Some("Title".to_string()));
                assert_eq!(record.body, Some("Body text".to_string()));
                assert_eq!(record.tags, Some("tag1,tag2".to_string()));
                assert_eq!(embedding, &vec![1.0, 0.0, 0.5]);
            }
            _ => panic!("Expected Insert operation"),
        }

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_update_delete_replay() {
        let wal_path = temp_path("wal_update_delete");

        let record1 = StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Original".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        };

        let record2 = StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Updated".to_string()),
                body: Some("New body".to_string()),
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![0.5, 0.5],
        };

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record1).unwrap();
            wal.append_update(&record2).unwrap();
            wal.append_delete("doc-1").unwrap();
            assert_eq!(wal.record_count(), 3);
        }

        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 3);

        assert!(matches!(&ops[0], WalOp::Insert { .. }));
        assert!(matches!(&ops[1], WalOp::Update { .. }));
        assert!(matches!(&ops[2], WalOp::Delete { id } if id == "doc-1"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_apply_ops_to_records() {
        let mut records = Vec::new();

        let ops = vec![
            WalOp::Insert {
                id: "doc-1".to_string(),
                record: Record {
                    id: "doc-1".to_string(),
                    title: Some("Title1".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![1.0, 0.0],
            },
            WalOp::Insert {
                id: "doc-2".to_string(),
                record: Record {
                    id: "doc-2".to_string(),
                    title: Some("Title2".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.0, 1.0],
            },
            WalOp::Update {
                id: "doc-1".to_string(),
                record: Record {
                    id: "doc-1".to_string(),
                    title: Some("Updated Title1".to_string()),
                    body: Some("New body".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.5, 0.5],
            },
            WalOp::Delete {
                id: "doc-2".to_string(),
            },
        ];

        let changed = apply_ops_to_records(ops, &mut records);
        assert!(changed);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record.id, "doc-1");
        assert_eq!(records[0].record.title, Some("Updated Title1".to_string()));
        assert_eq!(records[0].record.body, Some("New body".to_string()));
        assert_eq!(records[0].embedding, vec![0.5, 0.5]);
    }

    #[test]
    fn test_wal_truncate() {
        let wal_path = temp_path("wal_truncate");

        let record = StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Title".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        };

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record).unwrap();
            assert_eq!(wal.record_count(), 1);

            wal.truncate().unwrap();
            assert_eq!(wal.record_count(), 0);
        }

        let ops = replay(&wal_path).unwrap();
        assert!(ops.is_empty());

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_resume_append() {
        let wal_path = temp_path("wal_resume");

        let record1 = StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Title1".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        };

        let record2 = StoredRecord {
            record: Record {
                id: "doc-2".to_string(),
                title: Some("Title2".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![0.0, 1.0],
        };

        // Write first record
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record1).unwrap();
        }

        // Resume and write second record
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            assert_eq!(wal.record_count(), 1);
            wal.append_insert(&record2).unwrap();
            assert_eq!(wal.record_count(), 2);
        }

        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 2);

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_crash_recovery_simulation() {
        // This test simulates a crash recovery scenario:
        // 1. Initial records loaded from store
        // 2. Some operations written to WAL
        // 3. "Crash" happens (WAL not truncated)
        // 4. On restart, WAL is replayed to recover state

        let wal_path = temp_path("wal_crash_recovery");

        // Initial state: 2 records already in the store
        let mut records = vec![
            StoredRecord {
                record: Record {
                    id: "existing-1".to_string(),
                    title: Some("Existing 1".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![1.0, 0.0],
            },
            StoredRecord {
                record: Record {
                    id: "existing-2".to_string(),
                    title: Some("Existing 2".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.0, 1.0],
            },
        ];

        // Simulate operations that happen before "crash"
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();

            // Insert a new record
            let new_record = StoredRecord {
                record: Record {
                    id: "new-1".to_string(),
                    title: Some("New Record".to_string()),
                    body: Some("New body".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.5, 0.5],
            };
            wal.append_insert(&new_record).unwrap();

            // Update an existing record
            let updated_record = StoredRecord {
                record: Record {
                    id: "existing-1".to_string(),
                    title: Some("Updated Existing 1".to_string()),
                    body: Some("Updated body".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.8, 0.2],
            };
            wal.append_update(&updated_record).unwrap();

            // Delete a record
            wal.append_delete("existing-2").unwrap();

            assert_eq!(wal.record_count(), 3);
            // Simulate crash: WAL writer is dropped without truncate
        }

        // Simulate restart: replay WAL onto existing records
        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 3);

        let changed = apply_ops_to_records(ops, &mut records);
        assert!(changed);

        // Verify final state
        assert_eq!(records.len(), 2); // existing-1 + new-1 (existing-2 deleted)

        // Find and verify the updated existing-1
        let existing1 = records.iter().find(|r| r.record.id == "existing-1").unwrap();
        assert_eq!(existing1.record.title, Some("Updated Existing 1".to_string()));
        assert_eq!(existing1.record.body, Some("Updated body".to_string()));
        assert_eq!(existing1.embedding, vec![0.8, 0.2]);

        // Find and verify the new record
        let new1 = records.iter().find(|r| r.record.id == "new-1").unwrap();
        assert_eq!(new1.record.title, Some("New Record".to_string()));
        assert_eq!(new1.record.body, Some("New body".to_string()));
        assert_eq!(new1.embedding, vec![0.5, 0.5]);

        // Verify existing-2 was deleted
        assert!(records.iter().find(|r| r.record.id == "existing-2").is_none());

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_should_rotate() {
        let wal_path = temp_path("wal_rotation");

        let record = StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Title".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        };

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record).unwrap();
            wal.append_insert(&record).unwrap();

            // Test max_records threshold
            let config_records = WalConfig {
                path: wal_path.clone(),
                max_bytes: None,
                max_records: Some(2),
            };
            assert!(wal.should_rotate(&config_records).unwrap());

            let config_records_high = WalConfig {
                path: wal_path.clone(),
                max_bytes: None,
                max_records: Some(10),
            };
            assert!(!wal.should_rotate(&config_records_high).unwrap());

            // Test max_bytes threshold
            let file_size = wal.file_size().unwrap();
            let config_bytes = WalConfig {
                path: wal_path.clone(),
                max_bytes: Some(file_size),
                max_records: None,
            };
            assert!(wal.should_rotate(&config_bytes).unwrap());

            let config_bytes_high = WalConfig {
                path: wal_path.clone(),
                max_bytes: Some(file_size + 1000),
                max_records: None,
            };
            assert!(!wal.should_rotate(&config_bytes_high).unwrap());
        }

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_corrupted_magic() {
        let wal_path = temp_path("wal_corrupted");

        // Write invalid magic
        std::fs::write(&wal_path, b"INVALID!").unwrap();

        let result = WalWriter::new(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("invalid wal magic"));

        let result = replay(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("invalid wal magic"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_empty_ops() {
        let ops: Vec<WalOp> = Vec::new();
        let mut records = vec![StoredRecord {
            record: Record {
                id: "existing".to_string(),
                title: Some("Existing".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        }];

        let changed = apply_ops_to_records(ops, &mut records);
        assert!(!changed);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn test_wal_v1_detection() {
        let wal_path = temp_path("wal_v1_detection");

        // Write a v1 WAL file (vector-only format)
        {
            let mut file = File::create(&wal_path).unwrap();
            file.write_all(b"MUSUBIW1").unwrap(); // v1 magic
            file.write_all(&1u32.to_le_bytes()).unwrap(); // version 1
            // Write a simple insert: OP_INSERT(1) + dim(2) + [1.0, 0.0]
            file.write_all(&[1u8]).unwrap(); // OP_INSERT
            file.write_all(&2u32.to_le_bytes()).unwrap(); // dim
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.flush().unwrap();
        }

        // replay should return an error for v1 WAL
        let result = replay(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v1 detected"));

        // WalWriter::new should also return an error for v1 WAL
        let result = WalWriter::new(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v1 detected"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_apply_ops_upsert_semantics() {
        // Test that INSERT uses upsert (replaces if exists)
        let mut records = vec![StoredRecord {
            record: Record {
                id: "doc-1".to_string(),
                title: Some("Original".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            embedding: vec![1.0, 0.0],
        }];

        let ops = vec![
            WalOp::Insert {
                id: "doc-1".to_string(),
                record: Record {
                    id: "doc-1".to_string(),
                    title: Some("Replaced via INSERT".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.5, 0.5],
            },
        ];

        apply_ops_to_records(ops, &mut records);
        assert_eq!(records.len(), 1); // Still 1 record, not duplicated
        assert_eq!(records[0].record.title, Some("Replaced via INSERT".to_string()));
        assert_eq!(records[0].embedding, vec![0.5, 0.5]);

        // Test that UPDATE uses upsert (inserts if not exists)
        let ops = vec![
            WalOp::Update {
                id: "doc-new".to_string(),
                record: Record {
                    id: "doc-new".to_string(),
                    title: Some("Inserted via UPDATE".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                embedding: vec![0.0, 1.0],
            },
        ];

        apply_ops_to_records(ops, &mut records);
        assert_eq!(records.len(), 2); // Now 2 records
        let new_record = records.iter().find(|r| r.record.id == "doc-new").unwrap();
        assert_eq!(new_record.record.title, Some("Inserted via UPDATE".to_string()));
    }
}

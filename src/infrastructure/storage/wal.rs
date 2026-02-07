use chrono::NaiveDate;

use crate::domain::model::{Record, StoredRecord};
use crate::infrastructure::index::hnsw::HnswIndex;
use crate::infrastructure::storage::file as storage_file;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC_V5: &[u8; 8] = b"MUSUBIW5";
const MAGIC_V4: &[u8; 8] = b"MUSUBIW4";
const MAGIC_V3: &[u8; 8] = b"MUSUBIW3";
const MAGIC_V2: &[u8; 8] = b"MUSUBIW2";
const MAGIC_V1: &[u8; 8] = b"MUSUBIW1";
const VERSION: u32 = 5;

/// Operation kind for Write operations
const OP_WRITE: u8 = 1;
const OP_DELETE: u8 = 2;

/// Write operation kind (stored as single byte in WAL)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteKind {
    /// Insert a new record (upsert semantics)
    Insert,
    /// Update an existing record (upsert semantics)
    Update,
    /// Append a new record (idempotent: only if no active record exists)
    Append,
}

impl WriteKind {
    fn to_byte(self) -> u8 {
        match self {
            Self::Insert => 1,
            Self::Update => 2,
            Self::Append => 3,
        }
    }

    fn from_byte(b: u8) -> io::Result<Self> {
        match b {
            1 => Ok(Self::Insert),
            2 => Ok(Self::Update),
            3 => Ok(Self::Append),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown write kind: {}", b),
            )),
        }
    }
}

/// WAL operation for replay (consolidated)
#[derive(Debug, Clone)]
pub enum WalOp {
    /// Write a record (Insert, Update, or Append semantics)
    Write {
        kind: WriteKind,
        record: StoredRecord,
    },
    /// Mark a record as deleted (tombstone)
    Delete { id: String },
}

/// WAL rotation policy - determines when to rotate the WAL file
#[derive(Debug, Clone, Default)]
pub enum WalRotationPolicy {
    /// Never rotate automatically
    #[default]
    Disabled,
    /// Rotate when file size exceeds max_bytes
    MaxBytes(u64),
    /// Rotate when record count exceeds max_records
    MaxRecords(usize),
    /// Rotate when either condition is met
    MaxBytesOrRecords { max_bytes: u64, max_records: usize },
}

impl WalRotationPolicy {
    /// Check if rotation should occur based on current state
    fn should_rotate(&self, file_size: u64, record_count: usize) -> bool {
        match self {
            Self::Disabled => false,
            Self::MaxBytes(max) => file_size >= *max,
            Self::MaxRecords(max) => record_count >= *max,
            Self::MaxBytesOrRecords {
                max_bytes,
                max_records,
            } => file_size >= *max_bytes || record_count >= *max_records,
        }
    }
}

/// Configuration for WAL
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub path: PathBuf,
    pub rotation: WalRotationPolicy,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("hnsw.wal"),
            rotation: WalRotationPolicy::Disabled,
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

            if &magic == MAGIC_V1 || &magic == MAGIC_V2 || &magic == MAGIC_V3 || &magic == MAGIC_V4
            {
                // V1/V2/V3/V4 WAL detected - cannot auto-migrate.
                // V1: only stored vectors without record metadata.
                // V2: no deleted flag.
                // V3: tags stored as Option<String>.
                // V4: separate Insert/Update/Append ops, updated_at as String.
                // V5: consolidated Write op with WriteKind, updated_at as NaiveDate.
                // User must verify records.jsonl is complete and manually delete the WAL file.
                let version_str = if &magic == MAGIC_V1 {
                    "v1"
                } else if &magic == MAGIC_V2 {
                    "v2"
                } else if &magic == MAGIC_V3 {
                    "v3"
                } else {
                    "v4"
                };
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "WAL {} detected at {:?}. This format cannot be replayed. \
                         Please verify records.jsonl is complete, then delete the WAL file to proceed.",
                        version_str, path
                    ),
                ));
            } else if &magic != MAGIC_V5 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid wal magic: expected {:?}, got {:?}",
                        MAGIC_V5, magic
                    ),
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
            writer.write_all(MAGIC_V5)?;
            write_u32(&mut writer, VERSION)?;
            writer.flush()?;
        }

        Ok(Self {
            writer,
            path: path.to_path_buf(),
            record_count,
        })
    }

    /// Append a write operation to WAL (unified for Insert/Update/Append)
    fn append_write(&mut self, kind: WriteKind, stored: &StoredRecord) -> io::Result<()> {
        self.writer.write_all(&[OP_WRITE])?;
        self.writer.write_all(&[kind.to_byte()])?;
        write_stored_record(&mut self.writer, stored)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.record_count += 1;
        Ok(())
    }

    /// Append an INSERT operation to WAL
    pub fn append_insert(&mut self, stored: &StoredRecord) -> io::Result<()> {
        self.append_write(WriteKind::Insert, stored)
    }

    /// Append an UPDATE operation to WAL
    pub fn append_update(&mut self, stored: &StoredRecord) -> io::Result<()> {
        self.append_write(WriteKind::Update, stored)
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

    /// Append an APPEND operation to WAL (for tombstone-append pattern)
    pub fn append_append(&mut self, stored: &StoredRecord) -> io::Result<()> {
        self.append_write(WriteKind::Append, stored)
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
        let file_size = self.file_size()?;
        Ok(config.rotation.should_rotate(file_size, self.record_count))
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
        writer.write_all(MAGIC_V5)?;
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

    if &magic == MAGIC_V1 || &magic == MAGIC_V2 || &magic == MAGIC_V3 || &magic == MAGIC_V4 {
        // V1/V2/V3/V4 WAL cannot be replayed.
        // V1: only stored vectors without record metadata.
        // V2: no deleted flag.
        // V3: tags stored as Option<String>.
        // V4: separate ops, updated_at as String.
        // V5: consolidated Write op, updated_at as NaiveDate.
        // User must verify data integrity and manually delete the WAL file.
        let version_str = if &magic == MAGIC_V1 {
            "v1"
        } else if &magic == MAGIC_V2 {
            "v2"
        } else if &magic == MAGIC_V3 {
            "v3"
        } else {
            "v4"
        };
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "WAL {} detected at {:?}. This format cannot be replayed. \
                 Please verify records.jsonl is complete, then delete the WAL file to proceed.",
                version_str, path
            ),
        ));
    } else if &magic != MAGIC_V5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid wal magic: expected {:?}, got {:?}",
                MAGIC_V5, magic
            ),
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
            OP_WRITE => {
                let mut kind_byte = [0u8; 1];
                reader.read_exact(&mut kind_byte)?;
                let kind = WriteKind::from_byte(kind_byte[0])?;
                let record = read_stored_record(&mut reader)?;
                ops.push(WalOp::Write { kind, record });
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

/// Apply WAL operations to records list (idempotent replay)
/// - Write(Insert): upsert (replaces if ID exists, otherwise appends)
/// - Write(Update): upsert (replaces if ID exists, otherwise appends)
/// - Write(Append): appends only if no active record with same ID exists (idempotent)
/// - Delete: tombstones first occurrence if not already tombstoned (idempotent)
///
/// Returns true if any changes were made
pub fn apply_ops_to_records(ops: Vec<WalOp>, records: &mut Vec<StoredRecord>) -> bool {
    if ops.is_empty() {
        return false;
    }

    for op in ops {
        match op {
            WalOp::Write { kind, record } => {
                let id = record.record.id.clone();
                match kind {
                    WriteKind::Insert | WriteKind::Update => {
                        // Upsert: replace if exists, insert if not
                        if let Some(pos) = records.iter().position(|r| r.record.id == id) {
                            records[pos] = record;
                        } else {
                            records.push(record);
                        }
                    }
                    WriteKind::Append => {
                        // Idempotent: only append if no active record with same ID exists
                        let has_active = records.iter().any(|r| r.record.id == id && !r.deleted);
                        if !has_active {
                            records.push(record);
                        }
                        // If active exists, skip (already applied)
                    }
                }
            }
            WalOp::Delete { id } => {
                // Idempotent: only tombstone if first occurrence is not already deleted
                if let Some(pos) = records.iter().position(|r| r.record.id == id) {
                    if !records[pos].deleted {
                        records[pos].deleted = true;
                    }
                    // If already deleted, skip (idempotent)
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
            OP_WRITE => {
                let mut kind_byte = [0u8; 1];
                reader.read_exact(&mut kind_byte)?;
                let _ = WriteKind::from_byte(kind_byte[0])?;
                let _ = read_stored_record(reader)?;
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
    // Write record fields (excluding id which is written separately)
    write_string(writer, &record.id)?;
    write_optional_string(writer, &record.title)?;
    write_optional_string(writer, &record.body)?;
    write_optional_string(writer, &record.source)?;
    write_optional_date(writer, &record.updated_at)?;
    write_tags(writer, &record.tags)
}

fn read_record<R: Read>(reader: &mut R) -> io::Result<Record> {
    let id = read_string(reader)?;
    let title = read_optional_string(reader)?;
    let body = read_optional_string(reader)?;
    let source = read_optional_string(reader)?;
    let updated_at = read_optional_date(reader)?;
    let tags = read_tags(reader)?;
    Ok(Record {
        id,
        title,
        body,
        source,
        updated_at,
        tags,
    })
}

fn write_stored_record<W: Write>(writer: &mut W, stored: &StoredRecord) -> io::Result<()> {
    write_record(writer, &stored.record)?;
    write_embedding(writer, &stored.embedding)?;
    write_bool(writer, stored.deleted)
}

fn read_stored_record<R: Read>(reader: &mut R) -> io::Result<StoredRecord> {
    let record = read_record(reader)?;
    let embedding = read_embedding(reader)?;
    let deleted = read_bool(reader)?;
    Ok(StoredRecord::with_deleted(record, embedding, deleted))
}

fn write_optional_date<W: Write>(writer: &mut W, date: &Option<NaiveDate>) -> io::Result<()> {
    match date {
        Some(d) => {
            writer.write_all(&[1])?;
            // Store as days since epoch (i32)
            let days = d
                .signed_duration_since(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
                .num_days() as i32;
            writer.write_all(&days.to_le_bytes())
        }
        None => writer.write_all(&[0]),
    }
}

fn read_optional_date<R: Read>(reader: &mut R) -> io::Result<Option<NaiveDate>> {
    let mut flag = [0u8; 1];
    reader.read_exact(&mut flag)?;
    if flag[0] == 0 {
        Ok(None)
    } else {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let days = i32::from_le_bytes(buf);
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        epoch
            .checked_add_signed(chrono::Duration::days(days as i64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid date"))
            .map(Some)
    }
}

fn write_tags<W: Write>(writer: &mut W, tags: &[crate::domain::model::Tag]) -> io::Result<()> {
    write_u32(writer, tags.len() as u32)?;
    for tag in tags {
        write_string(writer, tag.as_str())?;
    }
    Ok(())
}

fn read_tags<R: Read>(reader: &mut R) -> io::Result<Vec<crate::domain::model::Tag>> {
    let len = read_u32(reader)? as usize;
    let mut tags = Vec::with_capacity(len);
    for _ in 0..len {
        let s = read_string(reader)?;
        tags.push(crate::domain::model::Tag::new(s));
    }
    Ok(tags)
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

fn write_bool<W: Write>(writer: &mut W, value: bool) -> io::Result<()> {
    writer.write_all(&[if value { 1 } else { 0 }])
}

fn read_bool<R: Read>(reader: &mut R) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
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
        use crate::domain::model::Tag;

        let wal_path = temp_path("wal_insert");

        let record = Record {
            id: "doc-1".to_string(),
            title: Some("Title".to_string()),
            body: Some("Body text".to_string()),
            source: None,
            updated_at: None,
            tags: vec![Tag::new("tag1"), Tag::new("tag2")],
        };
        let stored = StoredRecord::new(record, vec![1.0, 0.0, 0.5]);

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&stored).unwrap();
            assert_eq!(wal.record_count(), 1);
        }

        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 1);

        match &ops[0] {
            WalOp::Write { kind, record } => {
                assert_eq!(*kind, WriteKind::Insert);
                assert_eq!(record.record.id, "doc-1");
                assert_eq!(record.record.title, Some("Title".to_string()));
                assert_eq!(record.record.body, Some("Body text".to_string()));
                assert_eq!(record.record.tags.len(), 2);
                assert_eq!(record.record.tags[0].as_str(), "tag1");
                assert_eq!(record.record.tags[1].as_str(), "tag2");
                assert_eq!(record.embedding, vec![1.0, 0.0, 0.5]);
                assert!(!record.deleted);
            }
            _ => panic!("Expected Write operation"),
        }

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_update_delete_replay() {
        let wal_path = temp_path("wal_update_delete");

        let record1 = StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Original".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        );

        let record2 = StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Updated".to_string()),
                body: Some("New body".to_string()),
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![0.5, 0.5],
        );

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record1).unwrap();
            wal.append_update(&record2).unwrap();
            wal.append_delete("doc-1").unwrap();
            assert_eq!(wal.record_count(), 3);
        }

        let ops = replay(&wal_path).unwrap();
        assert_eq!(ops.len(), 3);

        assert!(matches!(
            &ops[0],
            WalOp::Write {
                kind: WriteKind::Insert,
                ..
            }
        ));
        assert!(matches!(
            &ops[1],
            WalOp::Write {
                kind: WriteKind::Update,
                ..
            }
        ));
        assert!(matches!(&ops[2], WalOp::Delete { id } if id == "doc-1"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_apply_ops_to_records() {
        let mut records = Vec::new();

        let ops = vec![
            WalOp::Write {
                kind: WriteKind::Insert,
                record: StoredRecord::new(
                    Record {
                        id: "doc-1".to_string(),
                        title: Some("Title1".to_string()),
                        body: None,
                        source: None,
                        updated_at: None,
                        tags: vec![],
                    },
                    vec![1.0, 0.0],
                ),
            },
            WalOp::Write {
                kind: WriteKind::Insert,
                record: StoredRecord::new(
                    Record {
                        id: "doc-2".to_string(),
                        title: Some("Title2".to_string()),
                        body: None,
                        source: None,
                        updated_at: None,
                        tags: vec![],
                    },
                    vec![0.0, 1.0],
                ),
            },
            WalOp::Write {
                kind: WriteKind::Update,
                record: StoredRecord::new(
                    Record {
                        id: "doc-1".to_string(),
                        title: Some("Updated Title1".to_string()),
                        body: Some("New body".to_string()),
                        source: None,
                        updated_at: None,
                        tags: vec![],
                    },
                    vec![0.5, 0.5],
                ),
            },
            WalOp::Delete {
                id: "doc-2".to_string(),
            },
        ];

        let changed = apply_ops_to_records(ops, &mut records);
        assert!(changed);
        // Tombstone semantics: Delete marks as deleted, doesn't remove
        assert_eq!(records.len(), 2);

        let doc1 = records.iter().find(|r| r.record.id == "doc-1").unwrap();
        assert_eq!(doc1.record.title, Some("Updated Title1".to_string()));
        assert_eq!(doc1.record.body, Some("New body".to_string()));
        assert_eq!(doc1.embedding, vec![0.5, 0.5]);
        assert!(!doc1.deleted);

        let doc2 = records.iter().find(|r| r.record.id == "doc-2").unwrap();
        assert!(doc2.deleted); // Tombstoned
    }

    #[test]
    fn test_wal_truncate() {
        let wal_path = temp_path("wal_truncate");

        let record = StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Title".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        );

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

        let record1 = StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Title1".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        );

        let record2 = StoredRecord::new(
            Record {
                id: "doc-2".to_string(),
                title: Some("Title2".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![0.0, 1.0],
        );

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
            StoredRecord::new(
                Record {
                    id: "existing-1".to_string(),
                    title: Some("Existing 1".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![1.0, 0.0],
            ),
            StoredRecord::new(
                Record {
                    id: "existing-2".to_string(),
                    title: Some("Existing 2".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.0, 1.0],
            ),
        ];

        // Simulate operations that happen before "crash"
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();

            // Insert a new record
            let new_record = StoredRecord::new(
                Record {
                    id: "new-1".to_string(),
                    title: Some("New Record".to_string()),
                    body: Some("New body".to_string()),
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.5, 0.5],
            );
            wal.append_insert(&new_record).unwrap();

            // Update an existing record
            let updated_record = StoredRecord::new(
                Record {
                    id: "existing-1".to_string(),
                    title: Some("Updated Existing 1".to_string()),
                    body: Some("Updated body".to_string()),
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.8, 0.2],
            );
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

        // Verify final state (tombstone semantics: deleted records remain but marked)
        assert_eq!(records.len(), 3); // existing-1 + existing-2 (tombstoned) + new-1

        // Find and verify the updated existing-1
        let existing1 = records
            .iter()
            .find(|r| r.record.id == "existing-1")
            .unwrap();
        assert_eq!(
            existing1.record.title,
            Some("Updated Existing 1".to_string())
        );
        assert_eq!(existing1.record.body, Some("Updated body".to_string()));
        assert_eq!(existing1.embedding, vec![0.8, 0.2]);
        assert!(!existing1.deleted);

        // Find and verify the new record
        let new1 = records.iter().find(|r| r.record.id == "new-1").unwrap();
        assert_eq!(new1.record.title, Some("New Record".to_string()));
        assert_eq!(new1.record.body, Some("New body".to_string()));
        assert_eq!(new1.embedding, vec![0.5, 0.5]);
        assert!(!new1.deleted);

        // Verify existing-2 was marked as deleted (tombstoned)
        let existing2 = records
            .iter()
            .find(|r| r.record.id == "existing-2")
            .unwrap();
        assert!(existing2.deleted);

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_should_rotate() {
        let wal_path = temp_path("wal_rotation");

        let record = StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Title".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        );

        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.append_insert(&record).unwrap();
            wal.append_insert(&record).unwrap();

            // Test max_records threshold
            let config_records = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::MaxRecords(2),
            };
            assert!(wal.should_rotate(&config_records).unwrap());

            let config_records_high = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::MaxRecords(10),
            };
            assert!(!wal.should_rotate(&config_records_high).unwrap());

            // Test max_bytes threshold
            let file_size = wal.file_size().unwrap();
            let config_bytes = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::MaxBytes(file_size),
            };
            assert!(wal.should_rotate(&config_bytes).unwrap());

            let config_bytes_high = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::MaxBytes(file_size + 1000),
            };
            assert!(!wal.should_rotate(&config_bytes_high).unwrap());

            // Test combined policy
            let config_combined = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::MaxBytesOrRecords {
                    max_bytes: file_size + 1000,
                    max_records: 2,
                },
            };
            assert!(wal.should_rotate(&config_combined).unwrap()); // records hit

            let config_disabled = WalConfig {
                path: wal_path.clone(),
                rotation: WalRotationPolicy::Disabled,
            };
            assert!(!wal.should_rotate(&config_disabled).unwrap());
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
        let mut records = vec![StoredRecord::new(
            Record {
                id: "existing".to_string(),
                title: Some("Existing".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        )];

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
    fn test_wal_v2_detection() {
        let wal_path = temp_path("wal_v2_detection");

        // Write a v2 WAL file (no deleted flag)
        {
            let mut file = File::create(&wal_path).unwrap();
            file.write_all(b"MUSUBIW2").unwrap(); // v2 magic
            file.write_all(&2u32.to_le_bytes()).unwrap(); // version 2
            file.flush().unwrap();
        }

        // replay should return an error for v2 WAL
        let result = replay(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v2 detected"));

        // WalWriter::new should also return an error for v2 WAL
        let result = WalWriter::new(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v2 detected"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_v3_detection() {
        let wal_path = temp_path("wal_v3_detection");

        // Write a v3 WAL file (Option<String> tags format)
        {
            let mut file = File::create(&wal_path).unwrap();
            file.write_all(b"MUSUBIW3").unwrap(); // v3 magic
            file.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
            file.flush().unwrap();
        }

        // replay should return an error for v3 WAL
        let result = replay(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v3 detected"));

        // WalWriter::new should also return an error for v3 WAL
        let result = WalWriter::new(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v3 detected"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_wal_v4_detection() {
        let wal_path = temp_path("wal_v4_detection");

        // Write a v4 WAL file (separate ops, updated_at as String)
        {
            let mut file = File::create(&wal_path).unwrap();
            file.write_all(b"MUSUBIW4").unwrap(); // v4 magic
            file.write_all(&4u32.to_le_bytes()).unwrap(); // version 4
            file.flush().unwrap();
        }

        // replay should return an error for v4 WAL
        let result = replay(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v4 detected"));

        // WalWriter::new should also return an error for v4 WAL
        let result = WalWriter::new(&wal_path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("WAL v4 detected"));

        std::fs::remove_file(&wal_path).ok();
    }

    #[test]
    fn test_apply_ops_upsert_semantics() {
        // Test that INSERT uses upsert (replaces if exists)
        let mut records = vec![StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Original".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        )];

        let ops = vec![WalOp::Write {
            kind: WriteKind::Insert,
            record: StoredRecord::new(
                Record {
                    id: "doc-1".to_string(),
                    title: Some("Replaced via INSERT".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.5, 0.5],
            ),
        }];

        apply_ops_to_records(ops, &mut records);
        assert_eq!(records.len(), 1); // INSERT upserts (replaces)
        assert_eq!(
            records[0].record.title,
            Some("Replaced via INSERT".to_string())
        );

        // Test that UPDATE uses upsert (inserts if not exists)
        let ops = vec![WalOp::Write {
            kind: WriteKind::Update,
            record: StoredRecord::new(
                Record {
                    id: "doc-new".to_string(),
                    title: Some("Inserted via UPDATE".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.8, 0.2],
            ),
        }];

        apply_ops_to_records(ops, &mut records);
        assert_eq!(records.len(), 2); // UPDATE inserts new record
        let new_record = records.iter().find(|r| r.record.id == "doc-new").unwrap();
        assert_eq!(
            new_record.record.title,
            Some("Inserted via UPDATE".to_string())
        );
    }

    #[test]
    fn test_tombstone_append_via_delete_append() {
        // Test the tombstone-append pattern: DELETE(old) + APPEND(new)
        let mut records = vec![StoredRecord::new(
            Record {
                id: "doc-1".to_string(),
                title: Some("Original".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            vec![1.0, 0.0],
        )];

        // Simulate embedding-change update: DELETE then APPEND
        let ops = vec![
            WalOp::Delete {
                id: "doc-1".to_string(),
            },
            WalOp::Write {
                kind: WriteKind::Append,
                record: StoredRecord::new(
                    Record {
                        id: "doc-1".to_string(),
                        title: Some("New version".to_string()),
                        body: None,
                        source: None,
                        updated_at: None,
                        tags: vec![],
                    },
                    vec![0.5, 0.5],
                ),
            },
        ];

        apply_ops_to_records(ops, &mut records);
        assert_eq!(records.len(), 2); // Old (tombstoned) + new
        assert!(records[0].deleted); // Old is tombstoned
        assert_eq!(records[0].record.title, Some("Original".to_string()));
        assert!(!records[1].deleted); // New is active
        assert_eq!(records[1].record.title, Some("New version".to_string()));
    }

    #[test]
    fn test_idempotent_replay_after_save() {
        // Test that replay is idempotent when crash happens after save
        // Scenario: DELETE + APPEND already applied to records.jsonl
        let mut records = vec![
            StoredRecord::with_deleted(
                Record {
                    id: "doc-1".to_string(),
                    title: Some("Original".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![1.0, 0.0],
                true, // Already tombstoned
            ),
            StoredRecord::new(
                Record {
                    id: "doc-1".to_string(),
                    title: Some("New version".to_string()),
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: vec![],
                },
                vec![0.5, 0.5],
            ),
        ];

        // Replay the same ops (simulating crash after save but before WAL truncate)
        let ops = vec![
            WalOp::Delete {
                id: "doc-1".to_string(),
            },
            WalOp::Write {
                kind: WriteKind::Append,
                record: StoredRecord::new(
                    Record {
                        id: "doc-1".to_string(),
                        title: Some("New version".to_string()),
                        body: None,
                        source: None,
                        updated_at: None,
                        tags: vec![],
                    },
                    vec![0.5, 0.5],
                ),
            },
        ];

        apply_ops_to_records(ops, &mut records);
        // Should remain unchanged (idempotent)
        assert_eq!(records.len(), 2);
        assert!(records[0].deleted); // Still tombstoned
        assert!(!records[1].deleted); // Still active
    }
}

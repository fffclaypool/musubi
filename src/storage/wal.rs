use crate::index::hnsw::HnswIndex;
use crate::storage::file as storage_file;
use crate::types::Vector;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"MUSUBIW1";
const VERSION: u32 = 1;

const OP_INSERT: u8 = 1;
const OP_DELETE: u8 = 2;

pub struct WalWriter {
    writer: BufWriter<File>,
}

impl WalWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        if path.exists() && path.metadata()?.len() > 0 {
            let mut reader = BufReader::new(File::open(path)?);
            let mut magic = [0u8; 8];
            reader.read_exact(&mut magic)?;
            if &magic != MAGIC {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid wal magic"));
            }
            let version = read_u32(&mut reader)?;
            if version != VERSION {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported wal version"));
            }
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);
        if path.metadata()?.len() == 0 {
            writer.write_all(MAGIC)?;
            write_u32(&mut writer, VERSION)?;
            writer.flush()?;
        }
        Ok(Self { writer })
    }

    pub fn append_insert(&mut self, vector: &[f32]) -> io::Result<()> {
        self.writer.write_all(&[OP_INSERT])?;
        write_u32(&mut self.writer, vector.len() as u32)?;
        for &v in vector {
            write_f32(&mut self.writer, v)?;
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_data()
    }
}

pub fn append_insert_to<P: AsRef<Path>>(path: P, vector: &[f32]) -> io::Result<()> {
    let mut wal = WalWriter::new(path)?;
    wal.append_insert(vector)
}

pub fn load_with_wal<P: AsRef<Path>>(
    snapshot_path: P,
    wal_path: P,
    m: usize,
    ef_construction: usize,
) -> io::Result<HnswIndex> {
    let mut index = if snapshot_path.as_ref().exists() {
        storage_file::load(snapshot_path)?
    } else {
        HnswIndex::new(m, ef_construction)
    };
    replay(wal_path, &mut index)?;
    Ok(index)
}

pub fn replay<P: AsRef<Path>>(path: P, index: &mut HnswIndex) -> io::Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }

    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid wal magic"));
    }
    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported wal version"));
    }

    loop {
        let mut op = [0u8; 1];
        match reader.read_exact(&mut op) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }

        match op[0] {
            OP_INSERT => {
                let dim = read_u32(&mut reader)? as usize;
                let mut v = Vec::with_capacity(dim);
                for _ in 0..dim {
                    v.push(read_f32(&mut reader)?);
                }
                index.insert(v);
            }
            OP_DELETE => {
                let _id = read_u32(&mut reader)?;
                // delete is reserved for future use
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown wal op")),
        }
    }

    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .elapsed()
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        path
    }

    #[test]
    fn test_wal_replay() {
        let wal_path = temp_path("hnsw_wal");
        let snapshot_path = temp_path("hnsw_snapshot");

        let mut wal = WalWriter::new(&wal_path).unwrap();
        wal.append_insert(&[1.0, 0.0]).unwrap();
        wal.append_insert(&[0.0, 1.0]).unwrap();

        let index = load_with_wal(&snapshot_path, &wal_path, 4, 20).unwrap();
        assert_eq!(index.len(), 2);

        std::fs::remove_file(&wal_path).ok();
    }
}

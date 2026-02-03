use crate::domain::model::{Record, StoredRecord};
use crate::domain::ports::RecordStore;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub struct JsonlRecordStore {
    path: PathBuf,
}

impl JsonlRecordStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl RecordStore for JsonlRecordStore {
    fn load(&self) -> io::Result<Vec<StoredRecord>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<StoredRecord>(trimmed) {
                Ok(record) => records.push(record),
                Err(_) => {
                    let record: Record = serde_json::from_str(trimmed)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
                    records.push(StoredRecord::new(record, Vec::new()));
                }
            }
        }
        Ok(records)
    }

    fn append(&self, record: &StoredRecord) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(record)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    fn save_all(&self, records: &[StoredRecord]) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&self.path)?;
        for record in records {
            let line = serde_json::to_string(record)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            writeln!(file, "{}", line)?;
        }
        Ok(())
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

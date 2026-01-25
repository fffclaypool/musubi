use crate::infrastructure::index::hnsw::HnswIndex;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"MUSUBI01";
const VERSION: u32 = 1;
const NONE_U32: u32 = u32::MAX;

pub fn save<P: AsRef<Path>>(index: &HnswIndex, path: P) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    writer.write_all(MAGIC)?;
    write_u32(&mut writer, VERSION)?;

    write_u32(&mut writer, index.dim as u32)?;
    write_u32(&mut writer, index.m as u32)?;
    write_u32(&mut writer, index.m_max0 as u32)?;
    write_u32(&mut writer, index.ef_construction as u32)?;

    write_u32(&mut writer, index.vectors.len() as u32)?;
    write_u32(&mut writer, index.max_level as u32)?;

    let entry = index.entry_point.map(|v| v as u32).unwrap_or(NONE_U32);
    write_u32(&mut writer, entry)?;

    for vector in &index.vectors {
        for &v in vector {
            write_f32(&mut writer, v)?;
        }
    }

    for &level in &index.node_levels {
        write_u32(&mut writer, level as u32)?;
    }

    for level in 0..=index.max_level {
        let layer = &index.layers[level];
        write_u32(&mut writer, layer.len() as u32)?;
        for neighbors in layer {
            write_u32(&mut writer, neighbors.len() as u32)?;
            for &id in neighbors {
                write_u32(&mut writer, id as u32)?;
            }
        }
    }

    writer.flush()
}

pub fn load<P: AsRef<Path>>(path: P) -> io::Result<HnswIndex> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
    }

    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
    }

    let dim = read_u32(&mut reader)? as usize;
    let m = read_u32(&mut reader)? as usize;
    let m_max0 = read_u32(&mut reader)? as usize;
    let ef_construction = read_u32(&mut reader)? as usize;

    let num_vectors = read_u32(&mut reader)? as usize;
    let max_level = read_u32(&mut reader)? as usize;
    let entry_point_raw = read_u32(&mut reader)?;
    let entry_point = if entry_point_raw == NONE_U32 {
        None
    } else {
        Some(entry_point_raw as usize)
    };

    let mut vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(read_f32(&mut reader)?);
        }
        vectors.push(v);
    }

    let mut node_levels = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        node_levels.push(read_u32(&mut reader)? as usize);
    }

    let mut layers: Vec<Vec<Vec<usize>>> = Vec::with_capacity(max_level + 1);
    for _ in 0..=max_level {
        let num_nodes = read_u32(&mut reader)? as usize;
        let mut layer = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let num_neighbors = read_u32(&mut reader)? as usize;
            let mut neighbors = Vec::with_capacity(num_neighbors);
            for _ in 0..num_neighbors {
                neighbors.push(read_u32(&mut reader)? as usize);
            }
            layer.push(neighbors);
        }
        layers.push(layer);
    }

    Ok(HnswIndex {
        vectors,
        node_levels,
        dim,
        layers,
        entry_point,
        m,
        m_max0,
        ef_construction,
        level_mult: 1.0 / (m as f64).ln(),
        max_level,
    })
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
        let unique = format!("{}_{}_{}", name, std::process::id(), std::time::SystemTime::now().elapsed().unwrap().as_nanos());
        path.push(unique);
        path
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut index = HnswIndex::new(4, 20);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![1.0, 1.0]);

        let path = temp_path("hnsw_roundtrip");
        save(&index, &path).unwrap();

        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(index.len(), loaded.len());
        let results = loaded.search(&[1.0, 0.0], 1, 10);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.001);
    }
}

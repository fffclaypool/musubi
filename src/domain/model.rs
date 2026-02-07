use chrono::NaiveDate;
use serde::{Deserialize, Deserializer, Serialize};

/// A normalized tag (lowercase, trimmed)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Tag(String);

impl<'de> Deserialize<'de> for Tag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Tag::new(s))
    }
}

impl Tag {
    /// Create a new tag, normalizing to lowercase and trimmed
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into().trim().to_lowercase())
    }

    /// Get the tag value as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parse comma-separated tags into a Vec<Tag>
    pub fn parse_many(tags: &str) -> Vec<Tag> {
        tags.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(Tag::new)
            .collect()
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A chunk of text extracted from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Index of this chunk within the parent document (0-based)
    pub chunk_index: usize,
    /// The text content of this chunk
    pub text: String,
    /// Start offset in the original document (character position)
    pub start_offset: usize,
    /// End offset in the original document (character position)
    pub end_offset: usize,
}

/// A stored chunk with its embedding and parent reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    /// The ID of the parent document
    pub parent_id: String,
    /// The chunk data
    #[serde(flatten)]
    pub chunk: Chunk,
    /// The embedding vector for this chunk
    pub embedding: Vec<f32>,
    /// Tombstone flag for soft deletion
    #[serde(default)]
    pub deleted: bool,
}

impl StoredChunk {
    pub fn new(parent_id: String, chunk: Chunk, embedding: Vec<f32>) -> Self {
        Self {
            parent_id,
            chunk,
            embedding,
            deleted: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub title: Option<String>,
    pub body: Option<String>,
    pub source: Option<String>,
    /// Date when the record was last updated (YYYY-MM-DD format in JSON)
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_naive_date"
    )]
    pub updated_at: Option<NaiveDate>,
    /// Tags associated with this record (normalized to lowercase)
    /// Supports both old format ("tag1, tag2") and new format (["tag1", "tag2"])
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_tags_compat"
    )]
    pub tags: Vec<Tag>,
}

/// Deserialize tags with backward compatibility for old comma-separated string format
fn deserialize_tags_compat<'de, D>(deserializer: D) -> Result<Vec<Tag>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct TagsVisitor;

    impl<'de> Visitor<'de> for TagsVisitor {
        type Value = Vec<Tag>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string of comma-separated tags or an array of tags")
        }

        // Old format: "tag1, tag2, tag3"
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Tag::parse_many(v))
        }

        // New format: ["tag1", "tag2", "tag3"]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut tags = Vec::new();
            while let Some(tag) = seq.next_element::<Tag>()? {
                tags.push(tag);
            }
            Ok(tags)
        }

        // Handle null
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }
    }

    deserializer.deserialize_any(TagsVisitor)
}

/// Custom serde module for Option<NaiveDate> with YYYY-MM-DD format
mod optional_naive_date {
    use chrono::NaiveDate;
    use serde::{self, Deserialize, Deserializer, Serializer};

    const FORMAT: &str = "%Y-%m-%d";

    pub fn serialize<S>(date: &Option<NaiveDate>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match date {
            Some(d) => serializer.serialize_str(&d.format(FORMAT).to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<NaiveDate>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => NaiveDate::parse_from_str(&s, FORMAT)
                .map(Some)
                .map_err(serde::de::Error::custom),
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRecord {
    #[serde(flatten)]
    pub record: Record,
    pub embedding: Vec<f32>,
    /// Tombstone flag for soft deletion. Defaults to false for backwards compatibility.
    #[serde(default)]
    pub deleted: bool,
}

impl StoredRecord {
    pub fn new(record: Record, embedding: Vec<f32>) -> Self {
        Self {
            record,
            embedding,
            deleted: false,
        }
    }

    pub fn with_deleted(record: Record, embedding: Vec<f32>, deleted: bool) -> Self {
        Self {
            record,
            embedding,
            deleted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tags_deserialize_new_format_array() {
        let json = r#"{"id":"1","tags":["ai","rust"]}"#;
        let record: Record = serde_json::from_str(json).unwrap();
        assert_eq!(record.tags.len(), 2);
        assert_eq!(record.tags[0].as_str(), "ai");
        assert_eq!(record.tags[1].as_str(), "rust");
    }

    #[test]
    fn test_tags_deserialize_old_format_string() {
        let json = r#"{"id":"1","tags":"ai, rust, ML"}"#;
        let record: Record = serde_json::from_str(json).unwrap();
        assert_eq!(record.tags.len(), 3);
        assert_eq!(record.tags[0].as_str(), "ai");
        assert_eq!(record.tags[1].as_str(), "rust");
        assert_eq!(record.tags[2].as_str(), "ml"); // normalized to lowercase
    }

    #[test]
    fn test_tags_deserialize_null() {
        let json = r#"{"id":"1","tags":null}"#;
        let record: Record = serde_json::from_str(json).unwrap();
        assert!(record.tags.is_empty());
    }

    #[test]
    fn test_tags_deserialize_missing() {
        let json = r#"{"id":"1"}"#;
        let record: Record = serde_json::from_str(json).unwrap();
        assert!(record.tags.is_empty());
    }

    #[test]
    fn test_tags_serialize_outputs_array() {
        let record = Record {
            id: "1".to_string(),
            title: None,
            body: None,
            source: None,
            updated_at: None,
            tags: vec![Tag::new("ai"), Tag::new("rust")],
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains(r#""tags":["ai","rust"]"#));
    }
}

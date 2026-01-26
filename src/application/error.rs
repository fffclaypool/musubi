use std::fmt;
use std::io;

#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Io(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::BadRequest(message) => write!(f, "bad request: {}", message),
            AppError::NotFound(message) => write!(f, "not found: {}", message),
            AppError::Conflict(message) => write!(f, "conflict: {}", message),
            AppError::Io(message) => write!(f, "io error: {}", message),
        }
    }
}

impl From<io::Error> for AppError {
    fn from(err: io::Error) -> Self {
        AppError::Io(err.to_string())
    }
}

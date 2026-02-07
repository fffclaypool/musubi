use crate::application::error::AppError;
use crate::application::service::{
    DocumentResponse, DocumentService, DocumentSummary, InsertCommand, InsertResult, SearchHit,
    SearchRequest, SearchValidationError, UpdateCommand, ValidatedSearchQuery,
};
use crate::domain::model::Record;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::io;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<DocumentService>>,
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    id: String,
    title: Option<String>,
    body: Option<String>,
    source: Option<String>,
    updated_at: Option<String>,
    tags: Option<String>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateRequest {
    title: Option<String>,
    body: Option<String>,
    source: Option<String>,
    updated_at: Option<String>,
    tags: Option<String>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Debug, Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct ListResponse {
    total: usize,
    items: Vec<DocumentSummary>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl From<AppError> for ApiError {
    fn from(err: AppError) -> Self {
        match err {
            AppError::BadRequest(message) => Self {
                status: StatusCode::BAD_REQUEST,
                message,
            },
            AppError::NotFound(message) => Self {
                status: StatusCode::NOT_FOUND,
                message,
            },
            AppError::Conflict(message) => Self {
                status: StatusCode::CONFLICT,
                message,
            },
            AppError::Io(message) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message,
            },
        }
    }
}

impl From<SearchValidationError> for ApiError {
    fn from(err: SearchValidationError) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

async fn insert_handler(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<InsertResult>, ApiError> {
    let record = Record {
        id: req.id,
        title: req.title,
        body: req.body,
        source: req.source,
        updated_at: req.updated_at,
        tags: req.tags,
    };
    let cmd = InsertCommand {
        record,
        text: req.text,
    };

    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = state.service.blocking_write();
        guard.insert(cmd).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(Json(result))
}

async fn update_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateRequest>,
) -> Result<Json<DocumentResponse>, ApiError> {
    let cmd = UpdateCommand {
        title: req.title,
        body: req.body,
        source: req.source,
        updated_at: req.updated_at,
        tags: req.tags,
        text: req.text,
    };

    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = state.service.blocking_write();
        guard.update(&id, cmd).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(Json(result))
}

async fn delete_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = state.service.blocking_write();
        guard.delete(&id).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(StatusCode::NO_CONTENT)
}

async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let state = state.clone();
    let results = tokio::task::spawn_blocking(move || {
        let guard = state.service.blocking_read();
        // Validate the request using service's default values
        let query = ValidatedSearchQuery::from_request(req, guard.default_k(), guard.default_ef())?;
        guard.search(query).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(Json(SearchResponse { results }))
}

async fn embed_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, ApiError> {
    let state = state.clone();
    let embeddings = tokio::task::spawn_blocking(move || {
        let guard = state.service.blocking_read();
        guard.embed_texts(req.texts).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(Json(EmbedResponse { embeddings }))
}

async fn get_document_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<DocumentResponse>, ApiError> {
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let guard = state.service.blocking_read();
        guard.get(&id).map_err(ApiError::from)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })??;

    Ok(Json(result))
}

async fn list_documents_handler(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, ApiError> {
    let limit = query.limit.unwrap_or(20).min(200);
    let offset = query.offset.unwrap_or(0);

    let state = state.clone();
    let (total, items) = tokio::task::spawn_blocking(move || {
        let guard = state.service.blocking_read();
        guard.list(offset, limit)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })?;

    Ok(Json(ListResponse { total, items }))
}

async fn health_handler() -> &'static str {
    "ok"
}

pub async fn serve(addr: String, service: DocumentService) -> io::Result<()> {
    let state = AppState {
        service: Arc::new(RwLock::new(service)),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/embed", post(embed_handler))
        .route(
            "/documents",
            post(insert_handler).get(list_documents_handler),
        )
        .route(
            "/documents/:id",
            get(get_document_handler)
                .put(update_handler)
                .delete(delete_handler),
        )
        .route("/search", post(search_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

use crate::application::error::AppError;
use crate::application::service::{
    job_store::JobStore, run_sync, BatchDocument, BatchInsertResult, DocumentDefaults,
    DocumentIngestion, DocumentSearch, DocumentService, IngestionJob, JobProgress, JobStatus,
    LastSyncInfo, SearchHit, SearchRequest, SearchValidationError, ValidatedSearchQuery,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::io;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<DocumentService>>,
    job_store: Arc<Mutex<JobStore>>,
}

#[derive(Debug, Deserialize)]
struct BatchInsertRequest {
    documents: Vec<BatchDocument>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct StartJobResponse {
    job_id: String,
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

// POST /documents/batch - Batch insert documents as pending
async fn batch_insert_handler(
    State(state): State<AppState>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<Json<BatchInsertResult>, ApiError> {
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = state.service.blocking_write();
        guard.batch_insert(req.documents)
    })
    .await
    .map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("join error: {}", err),
    })?
    .map_err(ApiError::from)?;

    Ok(Json(result))
}

// POST /ingestion/jobs - Start a sync job
async fn start_ingestion_handler(
    State(state): State<AppState>,
) -> Result<Json<StartJobResponse>, ApiError> {
    // Check and create in a single lock to prevent race condition
    let (job_id, job_arc) = {
        let mut job_store = state.job_store.lock().await;
        if job_store.has_running_job().await {
            return Err(ApiError {
                status: StatusCode::CONFLICT,
                message: "a sync job is already running".to_string(),
            });
        }
        job_store.create_job()
    };

    let response_job_id = job_id.clone();
    let service = Arc::clone(&state.service);
    let job_store = Arc::clone(&state.job_store);

    // Spawn the sync job in the background
    tokio::task::spawn_blocking(move || {
        let mut guard = service.blocking_write();

        // Get pending count for total
        let pending_ids = guard.get_pending_ids();
        let total = pending_ids.len();

        // Update job with total
        {
            let job = job_arc.blocking_write();
            let mut job = job;
            job.progress.total = total;
        }

        // Run the sync with progress callback
        let job_arc_for_callback = Arc::clone(&job_arc);
        let result = run_sync(&mut guard, |progress: &JobProgress| {
            let job = job_arc_for_callback.blocking_write();
            let mut job = job;
            job.progress = progress.clone();
        });

        // Update job status based on result
        {
            let job = job_arc.blocking_write();
            let mut job = job;
            match result {
                Ok(final_progress) => {
                    job.progress = final_progress;
                    job.status = JobStatus::Ready;
                }
                Err(e) => {
                    job.status = JobStatus::Failed {
                        error: e.to_string(),
                    };
                }
            }
            job.completed_at = Some(chrono::Utc::now());
        }

        // Mark job as completed in store
        let job_store = job_store.blocking_lock();
        let mut job_store = job_store;
        job_store.mark_completed(&job_id);
    });

    Ok(Json(StartJobResponse {
        job_id: response_job_id,
    }))
}

// GET /ingestion/jobs/:id - Get job status
async fn get_job_status_handler(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<IngestionJob>, ApiError> {
    let job_store = state.job_store.lock().await;

    let job_arc = job_store.get_job(&job_id).ok_or_else(|| ApiError {
        status: StatusCode::NOT_FOUND,
        message: "job not found".to_string(),
    })?;

    let job = job_arc.read().await;
    Ok(Json(job.clone()))
}

// GET /ingestion/last - Get last sync info
async fn get_last_sync_handler(
    State(state): State<AppState>,
) -> Result<Json<LastSyncInfo>, ApiError> {
    let job_store = state.job_store.lock().await;

    let last_sync = job_store.get_last_sync().await.ok_or_else(|| ApiError {
        status: StatusCode::NOT_FOUND,
        message: "no completed sync found".to_string(),
    })?;

    Ok(Json(last_sync))
}

// POST /search - Search indexed documents
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

// GET /health - Health check
async fn health_handler() -> &'static str {
    "ok"
}

pub async fn serve(addr: String, service: DocumentService) -> io::Result<()> {
    let state = AppState {
        service: Arc::new(RwLock::new(service)),
        job_store: Arc::new(Mutex::new(JobStore::new())),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/documents/batch", post(batch_insert_handler))
        .route("/ingestion/jobs", post(start_ingestion_handler))
        .route("/ingestion/jobs/:id", get(get_job_status_handler))
        .route("/ingestion/last", get(get_last_sync_handler))
        .route("/search", post(search_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

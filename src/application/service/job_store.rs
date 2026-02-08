//! In-memory job store for ingestion jobs.
//!
//! Jobs are not persisted across restarts. If a job was running when the server
//! restarted, the user must manually re-start the sync.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::types::{IngestionJob, JobProgress, JobStatus, LastSyncInfo};

/// In-memory store for ingestion jobs
pub struct JobStore {
    /// Active and completed jobs, keyed by job ID
    jobs: HashMap<String, Arc<RwLock<IngestionJob>>>,
    /// ID of the last completed job (for /ingestion/last endpoint)
    last_completed_id: Option<String>,
}

impl Default for JobStore {
    fn default() -> Self {
        Self::new()
    }
}

impl JobStore {
    /// Create a new empty job store
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            last_completed_id: None,
        }
    }

    /// Create a new job and return its ID and a handle to update it
    pub fn create_job(&mut self) -> (String, Arc<RwLock<IngestionJob>>) {
        let job_id = uuid::Uuid::new_v4().to_string();
        let job = IngestionJob {
            id: job_id.clone(),
            status: JobStatus::Processing,
            progress: JobProgress::default(),
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        let job_arc = Arc::new(RwLock::new(job));
        self.jobs.insert(job_id.clone(), Arc::clone(&job_arc));
        (job_id, job_arc)
    }

    /// Get a job by ID
    pub fn get_job(&self, job_id: &str) -> Option<Arc<RwLock<IngestionJob>>> {
        self.jobs.get(job_id).cloned()
    }

    /// Mark a job as completed and update the last_completed reference
    pub fn mark_completed(&mut self, job_id: &str) {
        self.last_completed_id = Some(job_id.to_string());
    }

    /// Get the last completed sync info
    pub async fn get_last_sync(&self) -> Option<LastSyncInfo> {
        let job_id = self.last_completed_id.as_ref()?;
        let job_arc = self.jobs.get(job_id)?;
        let job = job_arc.read().await;

        // Only return if the job is actually completed
        job.completed_at?;

        Some(LastSyncInfo {
            job_id: job.id.clone(),
            status: job.status.clone(),
            progress: job.progress.clone(),
            completed_at: job.completed_at.unwrap(),
        })
    }

    /// Check if there's currently a job running
    pub async fn has_running_job(&self) -> bool {
        for job_arc in self.jobs.values() {
            let job = job_arc.read().await;
            if matches!(job.status, JobStatus::Processing) {
                return true;
            }
        }
        false
    }
}

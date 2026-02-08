//! Read operations (defaults) for DocumentService.

use super::core::DocumentService;
use super::traits::DocumentDefaults;

impl DocumentDefaults for DocumentService {
    fn default_k(&self) -> usize {
        self.default_k
    }

    fn default_ef(&self) -> usize {
        self.default_ef
    }
}

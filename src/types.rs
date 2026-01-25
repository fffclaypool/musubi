/// f32値のスライスとして表現されるベクトル型
pub type Vector = Vec<f32>;

/// 最近傍探索の結果: (ノードID, 距離)
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

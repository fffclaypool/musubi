use crate::domain::types::{SearchResult, Vector};
use crate::infrastructure::index::metrics::{cosine_distance, normalize};
use crate::infrastructure::index::neighbor::select_neighbors_heuristic;
use crate::infrastructure::storage::file as storage_file;
use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::{BinaryHeap, HashSet};
use std::io;

/// 近似最近傍探索のためのHNSWインデックス
pub struct HnswIndex {
    /// インデックスに格納された全ベクトル
    pub(crate) vectors: Vec<Vector>,
    /// 各ノードのレベル
    pub(crate) node_levels: Vec<usize>,
    /// ベクトル次元数 (0は未設定)
    pub(crate) dim: usize,
    /// グラフレイヤー: layers[level][node_id] = 近傍ノードIDのリスト
    pub(crate) layers: Vec<Vec<Vec<usize>>>,
    /// エントリーポイントのノードID (最高レベルを持つノード)
    pub(crate) entry_point: Option<usize>,
    /// 各ノードの最大接続数 (各レイヤーごと)
    pub(crate) m: usize,
    /// レイヤー0での最大接続数 (通常は 2 * M)
    pub(crate) m_max0: usize,
    /// 構築時の動的候補リストのサイズ
    pub(crate) ef_construction: usize,
    /// レベル生成の乗数 (1 / ln(M))
    pub(crate) level_mult: f64,
    /// インデックス内の現在の最大レベル
    pub(crate) max_level: usize,
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new(16, 200)
    }
}

impl HnswIndex {
    /// 新しいHNSWインデックスを作成する
    ///
    /// # 引数
    /// * `m` - 各ノードの最大接続数 (デフォルト: 16)
    /// * `ef_construction` - 動的候補リストのサイズ (デフォルト: 200)
    pub fn new(m: usize, ef_construction: usize) -> Self {
        assert!(m >= 2, "m must be >= 2");
        Self {
            vectors: Vec::new(),
            node_levels: Vec::new(),
            dim: 0,
            layers: Vec::new(),
            entry_point: None,
            m,
            m_max0: m * 2,
            ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
            max_level: 0,
        }
    }

    /// インデックス内のベクトル数を取得する
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// インデックスに保存されたベクトルを取得する
    pub fn vector(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(id)
    }

    /// 次元数を取得する
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// インデックスが空かどうかを確認する
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// 新しいノードのランダムなレベルを生成する
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>().max(f64::EPSILON);
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// ベクトルをインデックスに挿入する
    ///
    /// 挿入されたベクトルのIDを返す
    pub fn insert(&mut self, vector: Vector) -> usize {
        let normalized = normalize(&vector).expect("zero vector is not supported for cosine");
        if self.dim == 0 {
            self.dim = normalized.len();
        } else {
            assert_eq!(self.dim, normalized.len(), "dimension mismatch");
        }

        let node_id = self.vectors.len();
        let node_level = self.random_level();

        // ベクトルを保存
        self.vectors.push(normalized);
        self.node_levels.push(node_level);

        // 必要なレイヤー数を確保
        while self.layers.len() <= node_level {
            self.layers.push(Vec::new());
        }

        // 各レベルで新しいノードの空の近傍リストを追加
        for level in 0..=self.max_level.max(node_level) {
            while self.layers[level].len() <= node_id {
                self.layers[level].push(Vec::new());
            }
        }

        // 最初のノードの場合、エントリーポイントとして設定
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_level = node_level;
            return node_id;
        }

        let entry_point = self.entry_point.unwrap();
        let query = &self.vectors[node_id];

        // 挿入のためのエントリーポイントを探す
        let mut current_node = entry_point;

        // 最上位レイヤーから node_level + 1 まで下降
        for level in (node_level + 1..=self.max_level).rev() {
            current_node = self.search_layer_single(query, current_node, level);
        }

        // node_level から 0 まで挿入処理
        for level in (0..=node_level.min(self.max_level)).rev() {
            let m_max = if level == 0 { self.m_max0 } else { self.m };

            // このレベルで最近傍を探す
            let neighbors = self.search_layer(query, current_node, self.ef_construction, level);

            // M個の最良の近傍を選択
            let selected: Vec<usize> = select_neighbors_heuristic(&self.vectors, &neighbors, m_max)
                .into_iter()
                .filter(|&id| self.node_levels[id] >= level)
                .collect();

            // 双方向の接続を追加
            self.layers[level][node_id] = selected.clone();

            for &neighbor_id in &selected {
                // 近傍ノードがこのレベルにエントリを持つことを確認
                while self.layers[level].len() <= neighbor_id {
                    self.layers[level].push(Vec::new());
                }

                let neighbor_ids = {
                    let neighbor_connections = &mut self.layers[level][neighbor_id];
                    neighbor_connections.push(node_id);
                    if neighbor_connections.len() <= m_max {
                        continue;
                    }
                    neighbor_connections.clone()
                };

                // 必要に応じて枝刈り
                let neighbor_vec = &self.vectors[neighbor_id];
                let mut scored: Vec<(usize, f32)> = neighbor_ids
                    .iter()
                    .map(|&id| (id, cosine_distance(neighbor_vec, &self.vectors[id])))
                    .collect();
                scored.sort_by_key(|&(_, d)| OrderedFloat(d));
                let selected = select_neighbors_heuristic(&self.vectors, &scored, m_max);
                self.layers[level][neighbor_id] = selected;
            }

            if !neighbors.is_empty() {
                current_node = neighbors[0].0;
            }
        }

        // 新しいノードがより高いレベルを持つ場合、エントリーポイントを更新
        if node_level > self.max_level {
            self.entry_point = Some(node_id);
            self.max_level = node_level;
        }

        node_id
    }

    /// 指定されたレベルで単一の最近傍ノードを探す (貪欲探索)
    fn search_layer_single(&self, query: &[f32], entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = cosine_distance(query, &self.vectors[current]);

        loop {
            let mut changed = false;

            if level < self.layers.len() && current < self.layers[level].len() {
                for &neighbor in &self.layers[level][current] {
                    let dist = cosine_distance(query, &self.vectors[neighbor]);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// 指定されたレベルでef個の最近傍を探す
    /// (node_id, distance) のソート済みベクトルを返す
    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_dist = cosine_distance(query, &self.vectors[entry]);

        // 候補: 最小ヒープ (最も近いものが先) - 最大ヒープの動作のために負の距離を使用
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        candidates.push((OrderedFloat(-entry_dist), entry));

        // 結果: 最大ヒープ (枝刈りのために最も遠いものが先)
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry));

        while let Some((neg_dist, current)) = candidates.pop() {
            let current_dist = -neg_dist.0;

            // 結果の中で最も遠い距離を取得
            let furthest_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

            // 現在の候補が最も遠い結果より遠い場合は停止
            if current_dist > furthest_dist && results.len() >= ef {
                break;
            }

            // 近傍を探索
            if level < self.layers.len() && current < self.layers[level].len() {
                for &neighbor in &self.layers[level][current] {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let neighbor_dist = cosine_distance(query, &self.vectors[neighbor]);
                    let furthest_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

                    if neighbor_dist < furthest_dist || results.len() < ef {
                        candidates.push((OrderedFloat(-neighbor_dist), neighbor));
                        results.push((OrderedFloat(neighbor_dist), neighbor));

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // 結果をソート済みベクトルに変換
        let mut result_vec: Vec<(usize, f32)> =
            results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_by_key(|&(_, d)| OrderedFloat(d));
        result_vec
    }

    /// クエリベクトルのk個の最近傍を探す
    ///
    /// # 引数
    /// * `query` - クエリベクトル
    /// * `k` - 返す最近傍の数
    /// * `ef` - 動的候補リストのサイズ (大きいほど精度が高いが遅くなる)
    ///
    /// # 戻り値
    /// k個の最近傍を含むSearchResultのベクトル
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() {
            return Vec::new();
        }
        if self.dim == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let normalized = match normalize(query) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let entry_point = self.entry_point.unwrap();
        let mut current_node = entry_point;

        // 最上位レイヤーからレイヤー1まで下降
        for level in (1..=self.max_level).rev() {
            current_node = self.search_layer_single(&normalized, current_node, level);
        }

        // レイヤー0でef個の候補を探索
        let results = self.search_layer(&normalized, current_node, ef.max(k), 0);

        // 上位k件の結果を返す
        results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }

    /// デフォルトのefパラメータで検索する
    pub fn search_default(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search(query, k, self.ef_construction)
    }

    /// インデックスをファイルに保存する
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<()> {
        storage_file::save(self, path)
    }

    /// ファイルからインデックスを読み込む
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        storage_file::load(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(4, 20);

        // ベクトルを挿入
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let v3 = vec![1.0, 1.0];
        let v4 = vec![-1.0, 0.0];

        index.insert(v1);
        index.insert(v2);
        index.insert(v3);
        index.insert(v4);

        assert_eq!(index.len(), 4);

        // [1,0]に最も近いベクトルを検索
        let results = index.search(&[1.0, 0.0], 2, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // v1のはず
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let a = normalize(&a).unwrap();
        let b = normalize(&b).unwrap();
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.001);
    }
}

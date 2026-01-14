use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::{BinaryHeap, HashSet};

/// f32値のスライスとして表現されるベクトル型
pub type Vector = Vec<f32>;

/// 最近傍探索の結果: (ノードID, 距離)
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// 近似最近傍探索のためのHNSWインデックス
pub struct HnswIndex {
    /// インデックスに格納された全ベクトル
    vectors: Vec<Vector>,
    /// グラフレイヤー: layers[level][node_id] = 近傍ノードIDのリスト
    layers: Vec<Vec<Vec<usize>>>,
    /// エントリーポイントのノードID (最高レベルを持つノード)
    entry_point: Option<usize>,
    /// 各ノードの最大接続数 (各レイヤーごと)
    m: usize,
    /// レイヤー0での最大接続数 (通常は 2 * M)
    m_max0: usize,
    /// 構築時の動的候補リストのサイズ
    ef_construction: usize,
    /// レベル生成の乗数 (1 / ln(M))
    level_mult: f64,
    /// インデックス内の現在の最大レベル
    max_level: usize,
}

impl HnswIndex {
    /// 新しいHNSWインデックスを作成する
    ///
    /// # 引数
    /// * `m` - 各ノードの最大接続数 (デフォルト: 16)
    /// * `ef_construction` - 動的候補リストのサイズ (デフォルト: 200)
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(),
            layers: Vec::new(),
            entry_point: None,
            m,
            m_max0: m * 2,
            ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
            max_level: 0,
        }
    }

    /// デフォルトパラメータで新しいHNSWインデックスを作成する
    pub fn default() -> Self {
        Self::new(16, 200)
    }

    /// インデックス内のベクトル数を取得する
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// インデックスが空かどうかを確認する
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// 2つのベクトル間のユークリッド距離を計算する
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// 新しいノードのランダムなレベルを生成する
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// ベクトルをインデックスに挿入する
    ///
    /// 挿入されたベクトルのIDを返す
    pub fn insert(&mut self, vector: Vector) -> usize {
        let node_id = self.vectors.len();
        let node_level = self.random_level();

        // ベクトルを保存
        self.vectors.push(vector);

        // 必要なレイヤー数を確保
        while self.layers.len() <= node_level {
            self.layers.push(Vec::new());
        }

        // 各レベルで新しいノードの空の近傍リストを追加
        for level in 0..=node_level {
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
        let query = &self.vectors[node_id].clone();

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
            let selected: Vec<usize> = neighbors
                .iter()
                .take(m_max)
                .map(|&(id, _)| id)
                .collect();

            // 双方向の接続を追加
            self.layers[level][node_id] = selected.clone();

            for &neighbor_id in &selected {
                // 近傍ノードがこのレベルにエントリを持つことを確認
                while self.layers[level].len() <= neighbor_id {
                    self.layers[level].push(Vec::new());
                }

                let neighbor_connections = &mut self.layers[level][neighbor_id];
                neighbor_connections.push(node_id);

                // 必要に応じて枝刈り
                if neighbor_connections.len() > m_max {
                    let neighbor_vec = &self.vectors[neighbor_id];
                    let mut scored: Vec<(usize, f32)> = neighbor_connections
                        .iter()
                        .map(|&id| (id, Self::euclidean_distance(neighbor_vec, &self.vectors[id])))
                        .collect();
                    scored.sort_by_key(|&(_, d)| OrderedFloat(d));
                    *neighbor_connections = scored.into_iter().take(m_max).map(|(id, _)| id).collect();
                }
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
        let mut current_dist = Self::euclidean_distance(query, &self.vectors[current]);

        loop {
            let mut changed = false;

            if level < self.layers.len() && current < self.layers[level].len() {
                for &neighbor in &self.layers[level][current] {
                    let dist = Self::euclidean_distance(query, &self.vectors[neighbor]);
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
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, level: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_dist = Self::euclidean_distance(query, &self.vectors[entry]);

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

                    let neighbor_dist = Self::euclidean_distance(query, &self.vectors[neighbor]);
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
        let mut result_vec: Vec<(usize, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
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

        let entry_point = self.entry_point.unwrap();
        let mut current_node = entry_point;

        // 最上位レイヤーからレイヤー1まで下降
        for level in (1..=self.max_level).rev() {
            current_node = self.search_layer_single(query, current_node, level);
        }

        // レイヤー0でef個の候補を探索
        let results = self.search_layer(query, current_node, ef.max(k), 0);

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(4, 20);

        // ベクトルを挿入
        let v1 = vec![0.0, 0.0];
        let v2 = vec![1.0, 0.0];
        let v3 = vec![0.0, 1.0];
        let v4 = vec![1.0, 1.0];

        index.insert(v1);
        index.insert(v2);
        index.insert(v3);
        index.insert(v4);

        assert_eq!(index.len(), 4);

        // 原点に最も近いベクトルを検索
        let results = index.search(&[0.0, 0.0], 2, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // v1のはず
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = HnswIndex::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }
}

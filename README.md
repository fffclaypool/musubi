# musubi

コサイン類似度を前提としたシンプルなHNSW実装と、最小の永続化レイヤーを持つRustプロジェクトです。

## 特徴

- コサイン距離 (正規化済み) のHNSW
- 近傍選択の多様性ヒューリスティック
- 単一ファイルへのスナップショット保存/復元

## 使い方

ビルドと実行:

```bash
cargo run
```

リリースビルド:

```bash
cargo run --release
```

## API例

```rust
use musubi::HnswIndex;

let mut index = HnswIndex::new(16, 200);
index.insert(vec![1.0, 0.0, 0.0]);

let results = index.search(&[1.0, 0.0, 0.0], 10, 100);
println!("{:?}", results);

index.save("hnsw.bin")?;
let loaded = HnswIndex::load("hnsw.bin")?;
let results2 = loaded.search(&[1.0, 0.0, 0.0], 10, 100);
println!("{:?}", results2);
```

## 構成

- `src/index/hnsw.rs` HNSW本体
- `src/index/metrics.rs` 距離/正規化
- `src/index/neighbor.rs` 近傍選択
- `src/storage/file.rs` 永続化 (save/load)
- `src/types.rs` 共通型

## パラメータ概要

- `M`: 各ノードの近傍数上限。大きいほど精度は上がりやすいが、メモリと計算量が増える (レイヤー0は `2*M`)。
- `ef_construction`: 構築時の探索幅。大きいほどインデックス品質が上がるが構築が遅くなる。
- `ef`: 検索時の探索幅。大きいほど recall が上がるが検索が遅くなる。

## ベンチ手順

`src/main.rs` の `num_vectors_list` / `ef_construction_values` / `ef_values` を調整して、
`cargo run --release` で実行してください。

```bash
cargo run --release
```

計測時間を短くしたい場合は `num_vectors_list` を減らすか、
`ef_values` を小さめにすると良いです。

## パラメータの目安 (10万ベクトル / 1536次元)

- `M`: 16〜32
- `ef_construction`: 200〜400
- `ef`: 50〜200

目標の recall@10 が 0.9 以上なら、まず `ef` を上げて調整し、
必要なら `M` / `ef_construction` を引き上げるのが近道です。

## 注意点

- すべてのベクトルは挿入時に正規化されます (ゼロベクトル不可)
- 次元は最初の挿入ベクトルで固定され、以降は一致が必要です
- 永続化は小さなスナップショット形式です (WALやCRCは未実装)

## テスト

```bash
cargo test
```

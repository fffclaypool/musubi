# musubi

学習用の HNSW (Hierarchical Navigable Small World) 実装です。
論文のアルゴリズムを素直に Rust で書き、その挙動を測って確かめることを目的にしています。

名前は「結び」から。ベクトル同士を近傍として結んでいくグラフ、という意味です。

## 現在できること

- HNSW インデックスの構築と検索 (コサイン距離)
- 多様性を保つヒューリスティックによる近傍選択
- インデックスのファイル保存・読み込み
- ANN-Benchmarks データセットでの recall / レイテンシ計測

## 使い方

```rust
use musubi::HnswIndex;

// m = 16, ef_construction = 200
let mut index = HnswIndex::new(16, 200);
index.insert(vec![1.0, 0.0]);
index.insert(vec![0.0, 1.0]);

// k = 1, ef = 50
let results = index.search(&[0.9, 0.1], 1, 50);
assert_eq!(results[0].id, 0);
```

ID は挿入順の連番です。ベクトルは挿入時に L2 正規化されます。

## サンプル

```bash
# ランダムなベクトルで構築し、総当たり検索と recall を比べる
make example

# data/sample.csv を埋め込み、似たドキュメントを探す (Python が必要)
make install
make example-docs
```

## ベンチマーク

ベンチマークは別クレート (`bench/`) に分かれており、`libhdf5-dev` が必要です。
ライブラリ本体のビルドとテストには不要です。

```bash
make bench-download   # glove-100-angular をダウンロード
make bench            # 1万件のサブセットで計測
make bench-full       # 全件で計測
```

結果は `data/bench/` に JSON で保存されます。

## 構成

| パス | 内容 |
|---|---|
| `src/hnsw.rs` | インデックス本体 (挿入・検索) |
| `src/neighbor.rs` | 近傍選択ヒューリスティック |
| `src/metrics.rs` | 距離関数と正規化 |
| `src/snapshot.rs` | 保存・読み込み |
| `examples/` | サンプル |
| `bench/` | ANN-Benchmarks 計測ツール |

## 開発

```bash
make test   # cargo test
make lint   # cargo fmt --check + cargo clippy
```

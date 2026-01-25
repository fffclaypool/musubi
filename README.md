# musubi

コサイン類似度を前提としたシンプルなHNSW実装と、最小の永続化レイヤーを持つRustプロジェクトです。
現在はREST APIでのドキュメント挿入/検索/取得/更新/削除に対応しています。

## 特徴

- コサイン距離 (正規化済み) のHNSW
- 近傍選択の多様性ヒューリスティック
- 単一ファイルへのスナップショット保存/復元
- REST APIでの挿入/検索/取得/更新/削除

## 使い方

サーバー起動:

```bash
cargo run
```

環境変数:
- `MUSUBI_ADDR` (既定: `127.0.0.1:8080`)
- `MUSUBI_PYTHON` (既定: `python3`)
- `MUSUBI_MODEL` (既定: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

主なエンドポイント:
- `POST /documents` 追加 (サーバー側で埋め込み)
- `PUT /documents/:id` 更新 (再埋め込み)
- `DELETE /documents/:id` 削除
- `POST /search` 検索 (text or embedding)
- `GET /documents/:id` 取得
- `GET /documents` 一覧

リクエスト/レスポンス例:

`POST /documents`
```json
{
  "id": "doc-1",
  "title": "Hello",
  "body": "World",
  "source": "example",
  "updated_at": "2024-01-01",
  "tags": "news",
  "text": "Hello\n\nWorld"
}
```
```json
{
  "index_id": 0,
  "id": "doc-1",
  "dim": 384
}
```

`PUT /documents/:id`
```json
{
  "title": "New title",
  "body": "New body",
  "text": "override text"
}
```

`POST /search`
```json
{
  "text": "Hello world",
  "k": 5,
  "ef": 100
}
```
```json
{
  "results": [
    {
      "index_id": 0,
      "id": "doc-1",
      "distance": 0.1234,
      "title": "Hello",
      "source": "example",
      "tags": "news"
    }
  ]
}
```

`GET /documents/:id`
```json
{
  "index_id": 0,
  "record": {
    "id": "doc-1",
    "title": "Hello",
    "body": "World",
    "source": "example",
    "updated_at": "2024-01-01",
    "tags": "news"
  },
  "embedding": [0.01, 0.02, 0.03]
}
```

例:
```bash
curl -s -X POST http://127.0.0.1:8080/documents \
  -H 'content-type: application/json' \
  -d '{"id":"doc-1","title":"Hello","body":"World"}'

curl -s -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"text":"Hello world","k":5,"ef":100}'

curl -s http://127.0.0.1:8080/documents/doc-1
```

CSV 埋め込み生成:

```bash
python scripts/embed_csv.py --input data/sample.csv --output data/embeddings.jsonl
```

リリースビルド:

```bash
cargo run --release
```

## ライブラリ利用例

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

## 構成 (Clean Architecture)

- `src/domain/` ドメインモデル/型/ポート
- `src/application/` ユースケース (DocumentService)
- `src/infrastructure/` HNSW/永続化/埋め込み実装
- `src/interface/` HTTP API (axum)
- `src/main.rs` 起動/DI

## パラメータ概要

- `M`: 各ノードの近傍数上限。大きいほど精度は上がりやすいが、メモリと計算量が増える (レイヤー0は `2*M`)。
- `ef_construction`: 構築時の探索幅。大きいほどインデックス品質が上がるが構築が遅くなる。
- `ef`: 検索時の探索幅。大きいほど recall が上がるが検索が遅くなる。

## 永続化

- HNSWスナップショット: `hnsw.bin`
- レコード/埋め込み: `data/records.jsonl`

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

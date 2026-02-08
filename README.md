# musubi

Rust製の小さなベクトル検索サーバーです。HNSW + BM25 のハイブリッド検索に対応し、
バッチ投入 → 同期ジョブ → 検索の流れで使います。

## できること

- HNSW (cosine) + BM25 のハイブリッド検索
- バッチ投入と差分同期 (content_hash)
- WAL による再起動耐性

## クイックスタート

```bash
# 依存関係
make install  # uv sync

# embeddingサーバー (ターミナル1)
make run-embed

# APIサーバー (ターミナル2)
make run-api
```

## 使い方

### 1) バッチ投入

```bash
curl -s -X POST http://127.0.0.1:8080/documents/batch \
  -H 'content-type: application/json' \
  -d '{
    "documents": [
      {"id":"doc-1","title":"Hello","body":"World","tags":["ai","rust"]},
      {"id":"doc-2","text":"Explicit text only"}
    ]
  }'
```

```json
{"accepted":2,"failed":0,"errors":[]}
```

### 2) 同期ジョブ

```bash
curl -s -X POST http://127.0.0.1:8080/ingestion/jobs
```

```json
{"job_id":"550e8400-e29b-41d4-a716-446655440000"}
```

```bash
curl -s http://127.0.0.1:8080/ingestion/jobs/550e8400-e29b-41d4-a716-446655440000
```

```json
{
  "id":"550e8400-e29b-41d4-a716-446655440000",
  "status":"ready",
  "progress": {"total":10,"processed":10,"indexed":8,"skipped":2,"failed":0}
}
```

### 3) 検索

```bash
curl -s -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"text":"hello world","k":5,"alpha":0.7}'
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | /documents/batch | バッチ投入 (pending として保存) |
| POST | /ingestion/jobs | 同期ジョブ開始 |
| GET | /ingestion/jobs/:id | 同期ジョブ状態 |
| GET | /ingestion/last | 直近の同期ジョブ |
| POST | /search | 検索 |
| GET | /health | ヘルスチェック |

## 検索リクエスト

```json
{
  "text": "query",
  "embedding": [0.1, 0.2],
  "k": 10,
  "ef": 100,
  "alpha": 0.7,
  "filter": {
    "source": "news",
    "mode": "any",
    "tags": ["ai", "rust"],
    "updated_at_gte": "2024-01-01",
    "updated_at_lte": "2024-12-31"
  }
}
```

## 検索レスポンス

```json
{
  "results": [
    {
      "index_id": 0,
      "id": "doc-1",
      "type": "hybrid",
      "distance": 0.1234,
      "bm25_score": 2.45,
      "hybrid_score": 0.83,
      "title": "Hello",
      "source": "example",
      "tags": ["news"],
      "best_chunk": {
        "chunk_index": 2,
        "text_preview": "This is the most relevant part..."
      }
    }
  ]
}
```

## 環境変数

| 変数 | 説明 | デフォルト |
|---|---|---|
| MUSUBI_ADDR | APIサーバーのアドレス | 127.0.0.1:8080 |
| MUSUBI_EMBED_URL | embeddingサーバーURL | (未設定) |
| MUSUBI_PYTHON | Pythonバイナリ | python3 |
| MUSUBI_MODEL | embeddingモデル | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| MUSUBI_WAL_ENABLED | WAL有効化 | true |
| MUSUBI_WAL_PATH | WALパス | hnsw.wal |
| MUSUBI_WAL_MAX_BYTES | WALローテーション (bytes) | (無制限) |
| MUSUBI_WAL_MAX_RECORDS | WALローテーション (records) | (無制限) |
| MUSUBI_CHUNK_TYPE | none/fixed/semantic | none |
| MUSUBI_CHUNK_SIZE | fixed サイズ | 512 |
| MUSUBI_CHUNK_OVERLAP | fixed overlap | 50 |
| MUSUBI_CHUNK_MIN_SIZE | semantic 最小 | 100 |
| MUSUBI_CHUNK_MAX_SIZE | semantic 最大 | 1000 |
| MUSUBI_CHUNK_THRESHOLD | semantic 閾値 | 0.5 |

## ストレージ

- レコード: data/records.jsonl
- pending: data/pending.jsonl (チャンク無効時のみ)
- チャンク: data/chunks.jsonl (チャンク有効時)
- WAL: hnsw.wal
- スナップショット: hnsw.bin

## 注意

- チャンク有効時はバッチ投入/同期は使えません (HTTP 400)。

## テスト

```bash
make test
# or
cargo test
```

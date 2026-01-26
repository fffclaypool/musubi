# musubi

コサイン類似度を前提としたシンプルなHNSW実装と、最小の永続化レイヤーを持つRustプロジェクトです。
現在はREST APIでのドキュメント挿入/検索/取得/更新/削除に対応しています。

## 特徴

- コサイン距離 (正規化済み) のHNSW
- 近傍選択の多様性ヒューリスティック
- 単一ファイルへのスナップショット保存/復元
- REST APIでの挿入/検索/取得/更新/削除
- 埋め込みモデルの事前ロード対応

## クイックスタート

```bash
# 依存関係インストール
make install  # uv sync

# embeddingサーバー起動 (ターミナル1)
make run-embed

# APIサーバー起動 (ターミナル2)
make run-api
```

## 使い方

### 方法1: HTTP Embedder (推奨)

embeddingモデルを事前にロードし、高速なレスポンスを実現します。

```bash
# 1. embeddingサーバー起動 (モデルを事前ロード)
uv run python/embedding_server.py

# 2. APIサーバー起動 (別ターミナル)
MUSUBI_EMBED_URL=http://127.0.0.1:8081/embed cargo run --release
```

### 方法2: Python Embedder (シンプル)

リクエストごとにPythonプロセスを起動します。セットアップは簡単ですが、レスポンスは遅くなります。

```bash
cargo run --release
```

## Makefile ターゲット

```
make build         - Rustバイナリをビルド (release)
make install       - Python依存関係をインストール
make run-embed     - embeddingサーバーを起動 (port 8081)
make run-api       - HTTP embedderでAPIサーバーを起動 (port 8080)
make run-api-python- Python embedderでAPIサーバーを起動
make test          - テスト実行
make clean         - ビルド成果物を削除
make help          - ヘルプを表示
```

## 環境変数

### APIサーバー

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `MUSUBI_ADDR` | APIサーバーのアドレス | `127.0.0.1:8080` |
| `MUSUBI_EMBED_URL` | embeddingサーバーのURL (設定時HTTP embedder使用) | (未設定) |
| `MUSUBI_PYTHON` | Pythonバイナリパス (Python embedder用) | `python3` |
| `MUSUBI_MODEL` | embeddingモデル名 (Python embedder用) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

### Embeddingサーバー

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `MUSUBI_EMBED_HOST` | バインドするホスト | `127.0.0.1` |
| `MUSUBI_EMBED_PORT` | バインドするポート | `8081` |
| `MUSUBI_MODEL` | embeddingモデル名 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

## API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/documents` | ドキュメント追加 (サーバー側で埋め込み) |
| `GET` | `/documents` | ドキュメント一覧 |
| `GET` | `/documents/:id` | ドキュメント取得 |
| `PUT` | `/documents/:id` | ドキュメント更新 (再埋め込み) |
| `DELETE` | `/documents/:id` | ドキュメント削除 |
| `POST` | `/search` | 検索 (text or embedding) |
| `POST` | `/embed` | テキストを埋め込みに変換 |
| `GET` | `/health` | ヘルスチェック |

## リクエスト/レスポンス例

### POST /documents

```bash
curl -s -X POST http://127.0.0.1:8080/documents \
  -H 'content-type: application/json' \
  -d '{"id":"doc-1","title":"Hello","body":"World"}'
```

```json
{
  "index_id": 0,
  "id": "doc-1",
  "dim": 384
}
```

### POST /search

```bash
curl -s -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"text":"Hello world","k":5,"ef":100}'
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

### GET /documents/:id

```bash
curl -s http://127.0.0.1:8080/documents/doc-1
```

```json
{
  "index_id": 0,
  "id": "doc-1",
  "title": "Hello",
  "body": "World",
  "source": "example",
  "updated_at": "2024-01-01",
  "tags": "news",
  "embedding": [0.01, 0.02, 0.03]
}
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

```
src/
├── domain/          # ドメインモデル/型/ポート
├── application/     # ユースケース (DocumentService)
├── infrastructure/  # HNSW/永続化/埋め込み実装
│   └── embedding/
│       ├── http.rs  # HTTP Embedder (事前ロード対応)
│       └── python.rs# Python Embedder (プロセス起動)
├── interface/       # HTTP API (axum)
└── main.rs          # 起動/DI

python/
├── embedding_server.py  # FastAPI embeddingサーバー
└── embed_text.py        # CLIでの埋め込み生成
```

## パラメータ概要

- `M`: 各ノードの近傍数上限。大きいほど精度は上がりやすいが、メモリと計算量が増える (レイヤー0は `2*M`)。
- `ef_construction`: 構築時の探索幅。大きいほどインデックス品質が上がるが構築が遅くなる。
- `ef`: 検索時の探索幅。大きいほど recall が上がるが検索が遅くなる。

### パラメータの目安 (10万ベクトル / 1536次元)

- `M`: 16〜32
- `ef_construction`: 200〜400
- `ef`: 50〜200

目標の recall@10 が 0.9 以上なら、まず `ef` を上げて調整し、
必要なら `M` / `ef_construction` を引き上げるのが近道です。

## 永続化

- HNSWスナップショット: `hnsw.bin`
- レコード/埋め込み: `data/records.jsonl`

## 注意点

- すべてのベクトルは挿入時に正規化されます (ゼロベクトル不可)
- 次元は最初の挿入ベクトルで固定され、以降は一致が必要です

## テスト

```bash
make test
# または
cargo test
```

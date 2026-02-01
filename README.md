# musubi

コサイン類似度を前提としたシンプルなHNSW実装と、最小の永続化レイヤーを持つRustプロジェクトです。
現在はREST APIでのドキュメント挿入/検索/取得/更新/削除に対応しています。

## 特徴

- コサイン距離 (正規化済み) のHNSW
- 近傍選択の多様性ヒューリスティック
- 単一ファイルへのスナップショット保存/復元
- **WAL (Write-Ahead Log) による再起動耐性**
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

### WAL (Write-Ahead Log)

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `MUSUBI_WAL_ENABLED` | WALを有効化 (`0`または`false`で無効化) | `true` |
| `MUSUBI_WAL_PATH` | WALファイルのパス | `hnsw.wal` |
| `MUSUBI_WAL_MAX_BYTES` | WALローテーションの閾値 (バイト数) | (無制限) |
| `MUSUBI_WAL_MAX_RECORDS` | WALローテーションの閾値 (レコード数) | (無制限) |

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
- WAL (Write-Ahead Log): `hnsw.wal`

## WAL (Write-Ahead Log)

WALにより、クラッシュや予期しない再起動からのデータ復旧が可能になります。

### 仕組み

1. **書き込み時**: 各操作 (INSERT/UPDATE/DELETE) は先にWALに記録され、その後でインデックスとレコードストアが更新されます
2. **起動時**: スナップショットをロードし、WALをリプレイして最新状態に復元します
3. **ローテーション**: WALサイズまたはレコード数が閾値を超えると、スナップショットを保存してWALをクリアします

### 運用フロー

```
[起動]
  └── records.jsonl をロード
  └── hnsw.wal をリプレイ (存在する場合)
  └── records.jsonl を更新 (WALの内容をマージ)
  └── hnsw.bin を保存
  └── hnsw.wal をクリア

[操作 (INSERT/UPDATE/DELETE)]
  └── WAL に追記
  └── インデックス/レコード更新
  └── hnsw.bin を保存
  └── (閾値超過時) WAL ローテーション

[クラッシュ時]
  └── WAL に操作が残っている
  └── 次回起動時にリプレイで復元
```

### 設定例

```bash
# WALを有効化し、10MBまたは1000件でローテーション
MUSUBI_WAL_PATH=data/hnsw.wal \
MUSUBI_WAL_MAX_BYTES=10485760 \
MUSUBI_WAL_MAX_RECORDS=1000 \
cargo run --release
```

### WALを無効化する場合

```bash
MUSUBI_WAL_ENABLED=false cargo run --release
```

### クラッシュ時の動作

| シナリオ | 結果 |
|---------|------|
| INSERT 途中 | 操作なし扱い (データ消失なし) |
| UPDATE (メタデータのみ) 途中 | 操作なし扱い |
| UPDATE (埋め込み変更) 途中 | **旧レコードが削除扱いになる可能性あり** |
| DELETE 途中 | 操作なし扱い |

**設計上の注意**: 埋め込み変更を伴う UPDATE は内部的に `DELETE(旧) → APPEND(新)` の2オペレーションで実行されます。極めて稀なケースですが、DELETE 直後にクラッシュすると旧レコードのみが削除扱い (tombstone) となり、新レコードが作成されない状態になります。この場合、ユーザーが再度 INSERT することで復旧できます。

### WAL バージョン

現在の WAL フォーマットは **v3** です。v1/v2 の WAL ファイルが存在する場合、起動時にエラーとなります。`records.jsonl` の内容が正しいことを確認した上で、WAL ファイルを手動で削除してください。

## 注意点

- すべてのベクトルは挿入時に正規化されます (ゼロベクトル不可)
- 次元は最初の挿入ベクトルで固定され、以降は一致が必要です

## ベンチマーク (ANN-Benchmarks)

ANN-Benchmarks の標準データセットを使って Recall@k とレイテンシを測定できます。

### データ取得

```bash
# glove-100-angular (デフォルト、約463MB)
./scripts/download_ann_data.sh

# または手動で
mkdir -p data/ann
wget -O data/ann/glove-100-angular.hdf5 http://ann-benchmarks.com/glove-100-angular.hdf5
```

### ベンチマーク実行

```bash
# フルデータセット (1.2M train, 10k test)
cargo run --release --bin bench_ann -- \
    --dataset data/ann/glove-100-angular.hdf5

# 10k サブセット (高速、brute-force で正解近傍を再計算)
cargo run --release --bin bench_ann -- \
    --dataset data/ann/glove-100-angular.hdf5 \
    --train-limit 10000 \
    --test-limit 1000

# パラメータ調整
cargo run --release --bin bench_ann -- \
    --dataset data/ann/glove-100-angular.hdf5 \
    --train-limit 10000 \
    --test-limit 1000 \
    --m 32 \
    --ef-construction 400 \
    --ef 200 \
    --k 10 \
    --save-index
```

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--dataset` | HDF5ファイルパス | (必須) |
| `--k` | 取得する近傍数 | 10 |
| `--ef` | 検索時の探索幅 | 100 |
| `--m` | HNSW M パラメータ | 16 |
| `--ef-construction` | 構築時の探索幅 | 200 |
| `--train-limit` | 訓練ベクトル数の上限 | (全件) |
| `--test-limit` | テストクエリ数の上限 | (全件) |
| `--output` | 結果出力ディレクトリ | data/bench |
| `--save-index` | インデックスサイズを計測 | false |

### 出力例 (JSON)

```json
{
  "dataset": "glove-100-angular",
  "train": 10000,
  "test": 1000,
  "k": 10,
  "m": 16,
  "ef_construction": 200,
  "ef_search": 100,
  "build_time_ms": 1234,
  "avg_ms": 1.5,
  "p50_ms": 1.2,
  "p95_ms": 3.4,
  "p99_ms": 5.1,
  "recall_at_k": 0.92,
  "index_size_bytes": 12345678
}
```

### 注意事項

- **サブセット時の正解近傍**: `--train-limit` を指定した場合、HDF5 の neighbors はフルデータセット用なので、brute-force で正解近傍を再計算します
- **cosine/angular**: glove-100-angular はコサイン類似度前提。ベクトルは自動で正規化されます

## テスト

```bash
make test
# または
cargo test
```

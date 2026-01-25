#!/usr/bin/env python3
import argparse
import csv
import json
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample.csv")
    parser.add_argument("--output", default="data/embeddings.jsonl")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    texts = [f"{r['title']}\n\n{r['body']}" for r in rows]
    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=args.batch, normalize_embeddings=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for row, emb in zip(rows, embeddings):
            record = {
                "id": row["id"],
                "title": row.get("title"),
                "body": row.get("body"),
                "source": row.get("source"),
                "updated_at": row.get("updated_at"),
                "tags": row.get("tags"),
                "embedding": emb.tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

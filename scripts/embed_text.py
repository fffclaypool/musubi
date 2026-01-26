#!/usr/bin/env python3
import argparse
import json
import sys
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    texts = payload.get("texts", [])
    if not texts:
        raise SystemExit("texts is required")

    print(f"loading embedding model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)
    print("embedding model loaded", file=sys.stderr)
    embeddings = model.encode(texts, batch_size=args.batch, normalize_embeddings=True)

    response = {
        "model": args.model,
        "embeddings": [emb.tolist() for emb in embeddings],
    }
    json.dump(response, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()

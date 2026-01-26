#!/usr/bin/env python3
"""Embedding server with pre-loaded model."""
import argparse
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model: SentenceTransformer | None = None
model_name: str = ""


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name
    print(f"Loading embedding model: {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name)
    print("Embedding model loaded", file=sys.stderr)
    yield
    model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is required")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    embeddings = model.encode(
        req.texts,
        batch_size=64,
        normalize_embeddings=True,
    )
    return EmbedResponse(
        model=model_name,
        embeddings=[emb.tolist() for emb in embeddings],
    )


def main() -> None:
    global model_name

    parser = argparse.ArgumentParser(description="Embedding server")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MUSUBI_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        help="Model name or path",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MUSUBI_EMBED_HOST", "127.0.0.1"),
        help="Host to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MUSUBI_EMBED_PORT", "8081")),
        help="Port to bind",
    )
    args = parser.parse_args()

    model_name = args.model
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np

def embed(texts, model):
    if model == "text-embedding-3-small":
        from openai import OpenAI
        client = OpenAI()
        return [v.embedding for start in range(0, len(texts), 100)
                for v in sorted(client.embeddings.create(
                    model=model, input=texts[start:start + 100]).data,
                    key=lambda item: item.index)]
    if model == "nomic-embed-text":
        import ollama
        return [vector for start in range(0, len(texts), 50)
                for vector in ollama.embed(
                    model=model, input=texts[start:start + 50]).embeddings]
    if model == "all-MiniLM-L6-v2":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return encoder.encode(texts, batch_size=64, show_progress_bar=False)
    raise ValueError(f"Unknown embedding model: {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    texts = json.loads(args.input.read_text(encoding="utf-8"))
    np.save(args.output, np.asarray(embed(texts, args.model), dtype=np.float32))

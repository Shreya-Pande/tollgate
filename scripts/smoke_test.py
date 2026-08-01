from fastembed import TextEmbedding
import numpy as np
import time

model = TextEmbedding(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=r"E:\fastembed-cache",
)
texts = ["summarize this passage in three bullet points"] * 100

list(model.embed(texts[:5]))  # warm-up

t = time.perf_counter()
vectors = [list(model.embed([t]))[0] for t in texts]
elapsed_ms = (time.perf_counter() - t) / len(texts) * 1000

print("dim:", len(vectors[0]))
print("norm:", np.linalg.norm(vectors[0]))
print("mean per-encode ms:", elapsed_ms)

from datasets import load_dataset
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

load_dataset("databricks/databricks-dolly-15k", split="train") \
    .to_parquet("data/raw/dolly.parquet")
load_dataset("google-research-datasets/paws", "labeled_final", split="train") \
    .select(range(20000)).to_parquet("data/raw/paws.parquet")
load_dataset("nyu-mll/glue", "qqp", split="train") \
    .select(range(20000)).to_parquet("data/raw/qqp.parquet")

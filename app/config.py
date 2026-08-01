from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    upstream_api_key: str = ""
    upstream_base_url: str = ""
    embedding_model: str
    fastembed_cache_path: str = "E:/fastembed-cache"
    similarity_threshold: float = 0.95
    top_k: int = 3
    workload_label: str = "mixed"

    # Cents per 1000 tokens, input/output priced separately, per upstream
    # model. Gemini 2.5 Flash rates (what gemini-flash-latest currently
    # aliases): $0.30/1M input, $2.50/1M output — ai.google.dev/gemini-api/docs/pricing,
    # confirmed 2026-08-01. Note: "-latest" is an alias Google can repoint;
    # these rates could silently go stale if the alias starts pointing at a
    # different underlying model.
    model_cost_cents_per_1k: dict[str, dict[str, float]] = {
        "gemini-flash-latest": {"input": 0.03, "output": 0.25},
    }
    default_cost_cents_per_1k: dict[str, float] = {"input": 0.03, "output": 0.25}


settings = Settings()

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
    # model. Pinned to gemini-3.5-flash-lite (NOT the "-latest" alias) —
    # $0.30/1M input, $2.50/1M output, ai.google.dev/gemini-api/docs/pricing,
    # confirmed 2026-08-01. DECISION-V2: switched off "gemini-flash-latest"
    # after it silently drifted mid-project (was resolving to a model whose
    # own pricing lookup was already stale by the time this was caught —
    # see the git history for what that cost). A pinned version can still
    # get deprecated out from under you (gemini-2.5-flash and
    # gemini-2.5-flash-lite both returned 404 "no longer available to new
    # users" when checked, despite being listed in /models), but at least
    # it won't change WHICH model you're paying for without you noticing.
    model_cost_cents_per_1k: dict[str, dict[str, float]] = {
        "gemini-3.5-flash-lite": {"input": 0.03, "output": 0.25},
    }
    default_cost_cents_per_1k: dict[str, float] = {"input": 0.03, "output": 0.25}


settings = Settings()

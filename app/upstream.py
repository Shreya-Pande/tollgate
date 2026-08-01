import httpx

from app.config import settings

CHARGE_SQL = """
UPDATE tenants SET spent_cents = spent_cents + $1
WHERE id = $2 AND spent_cents + $1 <= budget_cents
RETURNING spent_cents
"""


async def charge_budget(executor, tenant_id: str, cost_cents: float) -> float | None:
    """Atomic budget charge. No SELECT-then-UPDATE, no explicit locking —
    the WHERE clause folds the check into the same statement as the write.
    Returns the new spent_cents, or None if the charge would exceed budget.

    `executor` is either the pool (a standalone charge, its own transaction)
    or an open transaction's connection, so a reconcile charge can land in
    the same transaction as the cache_entries/cache_decisions writes.
    cost_cents may be negative (a refund/reconcile-down).
    """
    row = await executor.fetchrow(CHARGE_SQL, cost_cents, tenant_id)
    return row["spent_cents"] if row else None


RECONCILE_SQL = """
UPDATE tenants SET spent_cents = spent_cents + $1
WHERE id = $2
RETURNING spent_cents
"""


async def reconcile_budget(executor, tenant_id: str, delta_cents: float) -> float:
    """Unconditional budget adjustment — no budget_cents check, cannot fail
    or no-op. Used only for the miss-path reconcile (estimate -> real cost):
    by that point the upstream call has already happened and real money has
    been spent, so this write must never be skippable — if charge_budget's
    conditional WHERE silently matched zero rows here, the reconcile delta
    would be lost entirely and the ledger would permanently undercount real
    spend. If the real cost pushes the tenant over budget, they go slightly
    over — a rounding issue, not a data-integrity issue. The pre-flight gate
    and the estimate charge (charge_budget, above) still enforce the budget;
    only this final adjustment cannot.
    """
    row = await executor.fetchrow(RECONCILE_SQL, delta_cents, tenant_id)
    return row["spent_cents"]


def compute_cost_cents(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = settings.model_cost_cents_per_1k.get(model, settings.default_cost_cents_per_1k)
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


async def call_upstream(
    model: str, messages: list[dict], temperature: float, max_tokens: int
) -> dict:
    """Calls the OpenAI-compatible upstream chat-completions endpoint."""
    async with httpx.AsyncClient(base_url=settings.upstream_base_url, timeout=60.0) as client:
        # No leading slash: httpx merges a leading-slash path against just
        # the base_url's host, dropping its path (e.g. Gemini's
        # /v1beta/openai/ prefix). A relative path merges under it instead.
        resp = await client.post(
            "chat/completions",
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()

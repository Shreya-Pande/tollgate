from app.normalize import normalize, params_hash, raw_hash


def test_raw_hash_stable_for_identical_input():
    messages = [{"role": "user", "content": "what is a b-tree"}]
    assert raw_hash(messages) == raw_hash(messages)
    assert raw_hash(messages) == raw_hash([{"role": "user", "content": "what is a b-tree"}])


def test_raw_hash_differs_on_whitespace_change():
    a = [{"role": "user", "content": "what is a b-tree"}]
    b = [{"role": "user", "content": "what is a b-tree  "}]
    assert raw_hash(a) != raw_hash(b)


def test_normalize_collapses_whitespace_and_lowercases():
    messages = [{"role": "user", "content": "  Summarize   THIS.  "}]
    assert normalize(messages) == "summarize this."


def test_different_temperature_produces_different_params_hash():
    messages = [{"role": "user", "content": "what is a b-tree"}]
    h1 = params_hash(messages, temperature=0.0, max_tokens=1024, model="gemini-3.5-flash-lite")
    h2 = params_hash(messages, temperature=1.0, max_tokens=1024, model="gemini-3.5-flash-lite")
    assert h1 != h2


def test_different_tenants_same_prompt_have_different_cache_keys():
    """raw_hash alone doesn't scope by tenant — tenant_id is a separate
    column in the compound lookup key (tenant_id, prompt_hash,
    upstream_model, params_hash). Two tenants asking the identical prompt
    get the identical prompt_hash but a different overall cache key,
    because tenant_id differs. This is what stops tenant A's cache from
    ever serving tenant B (live-DB version: test_tenant_isolation.py,
    Phase 9).
    """
    messages = [{"role": "user", "content": "what is a b-tree"}]
    p_hash = raw_hash(messages)
    params = params_hash(messages, temperature=1.0, max_tokens=1024, model="gemini-3.5-flash-lite")

    key_tenant_a = ("tenant-a", p_hash, "gemini-3.5-flash-lite", params)
    key_tenant_b = ("tenant-b", p_hash, "gemini-3.5-flash-lite", params)

    assert key_tenant_a != key_tenant_b

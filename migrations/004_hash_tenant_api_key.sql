-- api_key was storing the raw bearer token in plaintext. Renamed to make
-- clear it holds a sha256 hash, not the secret itself — the secret is
-- shown to the caller once at seed time and never persisted anywhere.
-- Note: this rename does not transform existing data. Any tenant seeded
-- before this migration has its old plaintext value sitting under the
-- new column name and must be re-seeded, not just renamed.
ALTER TABLE tenants RENAME COLUMN api_key TO api_key_hash;

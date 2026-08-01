-- §3's tenants table has no way to map a bearer token to a tenant, but
-- the request path (§4 step 2) requires exactly that. Using the UUID id
-- itself as the credential would avoid a schema change, but ids aren't
-- meant to be secret (they leak into logs, URLs, FKs) and can't be
-- rotated independently of the row's identity. A dedicated column is
-- the standard fix. Table is empty at migration time, so NOT NULL is
-- safe without a default/backfill.
ALTER TABLE tenants ADD COLUMN api_key TEXT NOT NULL UNIQUE;

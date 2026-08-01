-- Exact-match hits (HIT_EXACT) skip embedding computation entirely by
-- design (§4 step 3 returns before step 6's embed step, to keep that
-- path ~2ms instead of paying the ~62ms embed cost for nothing). embedding
-- must be nullable so those decisions can still be logged — §5f requires
-- every request to write a decision row, hits included, and a fabricated
-- zero-vector would silently corrupt Phase 7's ROC data instead.
ALTER TABLE cache_decisions ALTER COLUMN embedding DROP NOT NULL;

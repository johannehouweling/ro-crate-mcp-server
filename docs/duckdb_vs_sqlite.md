DuckDB vs SQLite for on-disk index (pros, cons, and recommendation)

Summary
-------
Both SQLite (with FTS) and DuckDB are attractive embedded options for storing an on-disk index of RO-crate metadata, but they target different workloads. SQLite+FTS is a mature, lightweight choice optimized for transactional workloads and has excellent, battle-tested full-text search (FTS3/FTS4/FTS5). DuckDB is a columnar, analytical engine offering high-performance, parallel query execution and excellent support for large-scale analytical queries and Parquet integration — but its native full-text search story is less mature than SQLite's.

SQLite + FTS — Pros
- Mature and ubiquitous: available in virtually every Python runtime and OS distribution; FTS5 offers robust, well-tested full-text search features (tokenization, ranking, NEAR/AND/OR semantics).
- Simple integration: sqlite3 is in Python standard library; easy to initialize and ship a single .sqlite file.
- Good for typical RO-crate workloads: thousands to hundreds of thousands of small documents with frequent reads and occasional bulk writes.
- Very small operational footprint: zero dependencies (unless you want JSON/FTS features beyond the standard build).
- FTS is optimized specifically for text search use-cases (relevance ranking, phrase queries).

SQLite + FTS — Cons
- Single-writer constraint: SQLite allows many concurrent readers but only a single writer at a time. Under heavy concurrent writes this requires a writer queue or a background writer thread; WAL mode mitigates reader-writer blocking but doesn’t fully remove the single-writer model.
- Not columnar: analytical queries that scan many rows are less efficient than a columnar engine for large-scale analytics.
- FTS depends on the SQLite build: FTS5 availability is common but not guaranteed in some older or trimmed Python builds.

DuckDB — Pros
- Excellent analytical performance: vectorized, columnar execution, multi-threaded query engine that excels at scanning and aggregations across many rows — great if you plan analytics or batch processing on the index.
- Parquet integration and interoperability: native Parquet read/write and efficient columnar storage are useful for pipelines that produce Parquet artefacts.
- Better parallelism: DuckDB can execute queries in parallel and typically handles concurrent analytical queries better than SQLite.
- Flexible for ad-hoc analytics: if future requirements include heavy analytics or exporting the index into analytical workflows, DuckDB is convenient.

DuckDB — Cons
- Full-text search is not as mature as SQLite's FTS. You can still implement text search (e.g., using tokenization and inverted-index tables, or extensions), but you lose the simplicity and features of SQLite FTS out of the box.
- Additional dependency: duckdb python package must be added and available in the runtime.
- Not a drop-in replacement for FTS workflows: implementing ranking, proximity queries, or complex phrase queries is more work.

Recommendation for this project
--------------------------------
- If your primary requirement is reliable, feature-rich full-text search over crate metadata (title, description, authors, keywords) and the total scale is modest (10^2 to 10^5 crates), SQLite+FTS is the pragmatic and low-risk choice. It’s simple to implement, easy to test, and widely supported.

- If you expect to run heavy analytical queries over millions of extracted metadata fields, want Parquet integration, or plan to use the on-disk index as part of an analytics pipeline (aggregations, joins, columnar scans), DuckDB becomes attractive — but you’ll need to build or adopt a full-text strategy to match SQLite's FTS features.

- Hybrid approach: Make persistence pluggable. Implement a small persistence interface (the IndexStore already separates concerns) and provide two implementations: SqliteFTSIndexStore (simple, default) and DuckDBIndexStore (optional, for analytics). This adds some upfront work but gives flexibility.

Next steps I can take for you
- Implement SqliteFTSIndexStore now (fast, reliable). Or:
- Implement a DuckDB-backed store with a simple tokenized inverted-index table and basic ranking (more work). Or:
- Implement a pluggable persistence layer with both implementations and tests.

Tell me which option you prefer and I will implement it.

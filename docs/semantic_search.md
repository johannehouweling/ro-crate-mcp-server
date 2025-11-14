Semantic search and keyword search for RO-Crate metadata

Overview
--------
For LLM-driven MCP workflows, semantic (embedding) search is very valuable: it lets you match natural-language queries to crate metadata even when the exact keywords differ. However, keyword search remains important for precise filtering (IDs, authors, dates, controlled vocabulary fields).

How they complement each other
- Keyword/field search: exact or partial matches on structured fields (e.g., crate_id, author, DOI, date, specific metadata fields). Fast to run with SQL or in-memory filtering; deterministic and good for filtering down result sets.
- Semantic (embedding) search: converts text (title, description, extracted metadata) to dense vectors using an embedding model, then finds entries with closest vectors to the query. Great for synonyms, paraphrases, and concept-level matches.

Can we search keywords without FTS?
- Yes. If you only need exact or substring matching on small fields (title, author, keywords), a simple SQL WHERE LIKE or an in-memory string search is sufficient and performs well at modest scales (10^2–10^5 entries).
- FTS gives richer text search features (tokenization, ranking, phrase/NEAR queries). If you want relevance scoring and complex text queries, FTS is recommended.

Recommended architecture (practical and pluggable)
1. Keep the IndexStore interface pluggable.
2. Always store extracted_fields (title, description, authors, keywords) as structured data for exact/field filters.
3. At indexing time compute an embedding vector for each crate's combined textual signature (e.g., title + description + keywords). Store that vector in the persistence layer alongside the other fields.
   - For small-to-medium repos (<= 10^5 crates) you can store embeddings in SQLite as a BLOB and perform brute-force similarity (cosine) with numpy on a filtered set.
   - For production/scale or faster similarity search, use an ANN index (FAISS, Annoy, hnswlib) or a vector-enabled DB (Qdrant, Milvus, or DuckDB with vector extension if available).
4. Query flow:
   a. If field_filters are provided, apply those first to reduce candidate set (SQL or in-memory).
   b. If q (free text) is provided and semantic search is enabled, compute embedding for q, then run ANN search (or brute-force) against candidate vectors to get top-K nearest matches.
   c. Optionally combine with keyword scoring: run FTS/text-match score and blending with embedding similarity (weighted sum) for final ranking.

Implementation choices and trade-offs
- Simplicity (good for PoC / modest scale):
  - Compute embeddings using a chosen model (OpenAI, SentenceTransformers) at index time.
  - Store embeddings as numpy float32 arrays in sqlite BLOB or as separate files indexed by crate_id.
  - Search by loading candidate vectors into memory and computing cosine distances (fast enough for hundreds–thousands of vectors).
- Performance / production:
  - Use FAISS (or hnswlib) to build an on-disk ANN index that can be persisted and updated incrementally.
  - Or use a vector DB (Qdrant/Milvus) which gives persistence, replication, and REST APIs.
  - If you choose DuckDB for analytics, you can store vectors in columns or Parquet and use external ANN libs for search.

How this fits with existing design
- IndexEntry already includes extracted_fields — use those for keyword filters.
- Add new storage field: embedding: Optional[List[float]] in IndexEntry.
- Extend IndexStore to optionally persist embeddings and expose a search_semantic(query: str, top_k: int, filter: SearchFilter) method.
- Query API: add a parameter to /api/search to choose mode: "keyword" | "semantic" | "hybrid" and allow tuning weights for ranking.

Minimal implementation plan (next steps)
1. Add support for embeddings in models.IndexEntry (embedding: Optional[list[float]]).
2. Implement embedding generation hook in Indexer when parsing JSON-LD (configurable model/provider).
3. Extend IndexStore to persist embedding vectors (initially as BLOB in sqlite or in-memory list).
4. Implement a simple brute-force embedding search (numpy) over candidate set; add unit tests.
5. Later: add FAISS/hnswlib integration or a pluggable backend for ANN if performance requires it.

Recommendation for your MCP server
- Start with a hybrid approach: keep keyword/field filters (no FTS required yet), and add semantic embeddings stored alongside entries. For your expected scale (10^2 currently, possibly more), brute-force or an in-memory ANN will be sufficient and easy to implement.
- If you later need very large scale or persistent ANN features, add a pluggable ANN backend (FAISS or an external vector DB).

If you want, I can now implement the minimal embedding support:
- Update models.IndexEntry with embedding field
- Modify Indexer to compute embeddings (using a pluggable provider; default: a mock embedding function for tests)
- Extend IndexStore.search to accept a "semantic" mode that does brute-force vector similarity after applying field_filters
- Add unit tests

Reply "implement embeddings" to proceed.

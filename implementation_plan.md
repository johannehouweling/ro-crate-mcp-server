# Implementation Plan

[Overview]
Create an MCP server that indexes RO-Crate zip files stored in Azure Blob Storage by reading the embedded RO-Crate JSON-LD (ro-crate-metadata.json / ro-crate.json) from each archive without unpacking entire archives, and exposes a FastAPI-based query API for listing, searching, and fetching crate JSON-LD.

The server will be written in Python (FastAPI) and will default to an eager indexing strategy that scans an Azure Blob container for RO-crates and reads the canonical JSON-LD file from each zip archive to build a complete index on startup (with a hybrid/incremental mode available). The implementation provides a pluggable storage-backend abstraction so other backends (S3, local FS, HTTP) can be added.

Key architectural decision (updated): a single-engine DuckDB-based index will be used for structured storage, lexical BM25-style full-text, and vector similarity search (VSS). This keeps a single on-disk DB file and allows both columnar analytics and integrated lexical+semantic search using DuckDB extensions (duck-fts and VSS). Embeddings are generated via a pluggable EmbeddingProvider (default: local sentence-transformers) with optional cloud providers supported.

[Types]
Define strict Pydantic models to represent index entries, storage identifiers, and search filters.

Detailed type definitions:
- IndexEntry (pydantic.BaseModel)
  - crate_id: str  # unique id derived primarily from crate metadata (preferred) or generated from the resource locator
  - resource_locator: str  # opaque locator for the crate resource within the backend (e.g., blob name, s3 key, filesystem path, or URI)
  - resource_size: Optional[int]  # size in bytes when available
  - resource_last_modified: Optional[datetime]
  - metadata_path: str  # path inside the crate where the JSON-LD was found (e.g., "ro-crate-metadata.json" or "ro-crate.json")
  - top_level_metadata: dict[str, Any]  # top-level json-ld (subset or full)
  - extracted_fields: dict[str, Any]  # searchable fields extracted (title, authors, keywords, description)
  - checksum?: str  # optional content checksum
  - version?: str  # optional
  - storage_backend_id?: str  # optional identifier of the storage backend hosting the resource
  - indexed_at: datetime  # index insertion time
  - validation_status: Enum("unknown","valid","invalid")

Validation rules:
- crate_id must be non-empty and unique within an index
- resource_locator must be present and treated as an opaque identifier; do not encode backend-specific semantics into crate_id
- top_level_metadata must at minimum contain an "@context" or "@type" when present
- extracted_fields must include title if available; normalize author lists to arrays of names

- StorageBackendConfig (TypedDict)
  - backend_id: str  # unique identifier for this backend configuration
  - type: Literal["azure_blob", "s3", "filesystem", "generic_http"]  # backend type
  - config: dict[str, Any]  # backend-specific configuration (connection strings, buckets, prefixes); treated as opaque by the indexer
  - root_prefix?: str  # optional root prefix/path inside the backend

- SearchFilter (pydantic.BaseModel)
  - q: Optional[str]  # full-text across indexed extracted_fields
  - field_filters: Optional[dict[str, str]]  # exact or partial matches
  - limit: int = 50
  - offset: int = 0
  - semantic: bool = False  # whether to use semantic vector search
  - combine_with_bm25: bool = True  # if semantic, whether to combine BM25 scores with vector scores

[Files]
All file paths are relative to the repository root c:/Users/ArrasM/ro-crate-mcp-server.

Single sentence: Create a new Python package rocrate_mcp with modular subpackages for storage backends, indexing, API, models, and a DuckDB-based index store; add tests and configuration.

Detailed breakdown (updated to reflect DuckDB-centric design):
- New and existing files (create or update as needed)
  - rocrate_mcp/__init__.py  # package init
  - rocrate_mcp/main.py  # FastAPI app and application startup/shutdown
  - rocrate_mcp/models.py  # Pydantic models (IndexEntry, SearchFilter, StorageBackendConfig)
  - rocrate_mcp/storage/__init__.py
  - rocrate_mcp/storage/base.py  # StorageBackend abstract base class and exceptions
  - rocrate_mcp/storage/azure_blob.py  # AzureBlobStorageBackend implementation (uses azure-storage-blob)
  - rocrate_mcp/index/__init__.py
  - rocrate_mcp/index/indexer.py  # Indexer class: eager and hybrid modes, incremental updates
  - rocrate_mcp/index/store.py  # DuckDB-based IndexStore (duck-fts for BM25, VSS for vectors) and in-memory cache layer
  - rocrate_mcp/api/__init__.py
  - rocrate_mcp/api/routes.py  # FastAPI routers: /health, /index, /crates, /search, /crate/{id}
  - rocrate_mcp/utils/zip_reader.py  # read a single file from zip in stream (without unpacking whole archive)
  - rocrate_mcp/utils/json.py  # helpers to extract canonical ro-crate JSON-LD path and normalize metadata
  - rocrate_mcp/config.py  # settings (pydantic BaseSettings) including connection strings, indexing mode and embedding provider config
  - rocrate_mcp/embeddings.py  # EmbeddingProvider interface and implementations
  - tests/test_indexer.py
  - tests/test_storage_mock.py
  - tests/test_api.py
  - scripts/cli_index.py  # optional CLI utility to run indexing separately

- Existing files to be modified
  - pyproject.toml  # add dependencies: azure-storage-blob, duckdb, sentence-transformers, python-dotenv (optional), pytest extras if needed
  - README.md  # add usage, configuration and API examples

- Files to be deleted or moved
  - None

[Functions]
Single sentence: Add functions for listing blobs, reading a single file from a zip blob, indexing a container, searching the DuckDB index (BM25 and VSS), and API handlers.

Detailed breakdown:
- New functions
  - StorageBackend.list_blobs(prefix: Optional[str] = None) -> Iterator[BlobInfo]
    - File: rocrate_mcp/storage/base.py
    - Purpose: yield blob metadata (name, size, last_modified)
  - AzureBlobStorageBackend.get_blob_stream(blob_name: str) -> BinaryIO
    - File: rocrate_mcp/storage/azure_blob.py
    - Purpose: return a streaming response or io.BytesIO-like stream for reading a blob
  - utils.zip_reader.find_file_in_zip_stream(stream: BinaryIO, target_names: List[str]) -> bytes | None
    - File: rocrate_mcp/utils/zip_reader.py
    - Purpose: scan zip central directory in stream and extract specific file bytes without full extraction (use zipfile module with BytesIO but avoid saving to disk)
  - index.indexer.Indexer.build_index(eager: bool = True) -> None
    - File: rocrate_mcp/index/indexer.py
    - Purpose: orchestrate listing blobs, reading ro-crate JSON-LD from each crate zip, generating embeddings (when enabled), and populating DuckDB IndexStore
  - index.indexer.Indexer.refresh_incremental() -> None
    - File: rocrate_mcp/index/indexer.py
    - Purpose: check container changes by last_modified and update index incrementally
  - index.store.IndexStore.search(filter: SearchFilter) -> List[IndexEntry]
    - File: rocrate_mcp/index/store.py
    - Purpose: run BM25 (duck-fts) and/or VSS searches inside DuckDB and merge/rerank results for hybrid search
  - api.routes.search_crates(filter: SearchFilter) -> JSONResponse
    - File: rocrate_mcp/api/routes.py
    - Purpose: HTTP endpoint that returns paginated search results
  - cli_index.main() -> None
    - File: scripts/cli_index.py
    - Purpose: run indexing as an ad-hoc command

- Modified functions
  - main.py app startup to call Indexer.build_index

[Classes]
Single sentence: Introduce StorageBackend ABC, AzureBlobStorageBackend, Indexer, and DuckDB IndexStore classes to encapsulate core responsibilities.

Detailed breakdown:
- New classes
  - StorageBackend (abc.ABC)
    - File: rocrate_mcp/storage/base.py
    - Key methods: list_blobs(prefix) -> Iterator[BlobInfo], get_blob_stream(blob_name) -> BinaryIO, get_blob_metadata(blob_name) -> BlobInfo
    - Purpose: abstract storage operations
  - AzureBlobStorageBackend(StorageBackend)
    - File: rocrate_mcp/storage/azure_blob.py
    - Key methods: implements list_blobs (uses ContainerClient.list_blobs), get_blob_stream (BlobClient.download_blob().readall or .chunks())
    - Initialization: accepts connection_string, container, root_prefix
  - Indexer
    - File: rocrate_mcp/index/indexer.py
    - Key methods: __init__(backend: StorageBackend, store: IndexStore, mode: Literal["eager","hybrid"]), build_index(), refresh_incremental(), get_crate(crate_id)
    - Behavior: orchestrates bounded concurrency via asyncio.Semaphore, handles transient blob read errors, optionally validates JSON-LD with ro-crate-py, and generates embeddings via EmbeddingProvider when semantic search is enabled
  - IndexStore (DuckDBIndexStore)
    - File: rocrate_mcp/index/store.py
    - Key methods: insert(entry: IndexEntry), bulk_insert(entries: Iterable[IndexEntry]), search(filter: SearchFilter) -> List[IndexEntry], get(crate_id) -> IndexEntry | None, persist() -> None, load() -> None
    - Implementation: DuckDB table with duck-fts for full-text indexing and VSS vector support for embeddings. Embeddings stored as DuckDB vectors or arrays.
  - EmbeddingProvider (protocol / ABC)
    - File: rocrate_mcp/embeddings.py
    - Key methods: embed_texts(texts: List[str]) -> np.ndarray
    - Implementations: SentenceTransformersProvider (local) and OpenAIProvider (optional)

- Modified classes
  - None existing

[Dependencies]
Single sentence: Use DuckDB (with FTS and VSS), sentence-transformers (default embeddings), azure-storage-blob and optionally python-dotenv; keep ro-crate-py and FastAPI.

Details:
- New packages
  - duckdb>=1.**  # ensure VSS and FTS extensions available
  - sentence-transformers>=2.0.0  # default local embeddings
  - azure-storage-blob>=12.14.0
  - python-dotenv>=1.0 (optional, for local env loading)
- Optional packages
  - faiss-cpu (optional)  # only if FAISS sidecar is requested for very large vector collections
  - openai (optional)  # to support OpenAI embeddings as a configurable provider
- Existing packages to keep (from pyproject.toml)
  - ro-crate-py>=0.9.0
  - fastapi>=0.110.0
  - uvicorn>=0.29.0

[Testing]
Single sentence: Use pytest to cover storage backend (mocked), indexer logic, DuckDB IndexStore behavior (fts + vss), and API endpoints with an in-memory index.

Test plan details (updated):
- tests/test_storage_mock.py
  - Mock AzureContainerClient and BlobClient behavior to simulate blob listings and streaming a small zip file containing ro-crate.json
- tests/test_indexer.py
  - Unit tests for Indexer.build_index with a mocked StorageBackend, asserting DuckDB IndexStore contents, embedding generation and error handling
- tests/test_store_duckdb.py
  - Tests that verify duck-fts queries, VSS nearest-neighbour results, hybrid ranking and limit/offset behavior
- tests/test_api.py
  - Use FastAPI TestClient to exercise /search and /crate/{id} endpoints against a populated DuckDB IndexStore
- Validation strategies
  - Fuzz invalid JSON-LD to ensure validation_status is set to invalid; ensure exceptions are captured and do not abort indexing
  - Performance tests (manual): simple script that generates N small zip blobs and times indexing (documented in README)

[Implementation Order] (updated for DuckDB single-engine)
Single sentence: Implement the storage abstraction, the Azure backend, the zip metadata reader, the DuckDB IndexStore (duck-fts + VSS), the EmbeddingProvider and Indexer, then the FastAPI API and tests in that order.

Numbered steps:
1. Create package scaffolding and configuration (rocrate_mcp/, config.py, pyproject updates)
2. Implement StorageBackend ABC and AzureBlobStorageBackend (list_blobs, get_blob_stream)
3. Implement utils/zip_reader.py to extract ro-crate JSON-LD from a zip stream
4. Implement models.py (IndexEntry, SearchFilter, StorageBackendConfig)
5. Implement DuckDB IndexStore with duck-fts (BM25-like) and VSS (vector similarity search). Provide fallback brute-force vector similarity if VSS unavailable.
6. Implement EmbeddingProvider interface and local SentenceTransformers provider (with optional OpenAI provider)
7. Implement Indexer (eager build_index and refresh_incremental) and wire embedding generation into indexing flow (configurable)
8. Wire FastAPI app in main.py and api/routes.py (health, /crates, /search, /crate/{id}) and ensure startup calls Indexer.build_index
9. Add CLI script scripts/cli_index.py for manual reindexing
10. Add tests for storage, indexer, DuckDB store, and API; update CI/test config in pyproject.toml
11. Update README.md with configuration, usage, and scaling notes (DuckDB VSS vs brute-force, embedding provider options)

[Notes and operational guidance]
- Single-file deployment: default to keeping duckdb file (duckdb_index.db) in the configured index directory for persistence and portability.
- Fallbacks: detect DuckDB VSS/FTS availability at runtime and fall back to brute-force vector computation or in-process token matching if extensions are unavailable. Log prominent warnings during startup.
- Embeddings: default to a small sentence-transformers model to avoid requiring cloud credentials; document how to switch to OpenAI in config.py and README.
- Rebuild strategy: provide a CLI command to rebuild the DuckDB index from the storage backend; Indexer should be idempotent and able to reconcile on startup.

[Next actions]
- Update repository files (pyproject.toml, README.md) and implement DuckDB IndexStore, EmbeddingProvider, and updated Indexer. Add tests for DuckDB FTS/VSS behavior.

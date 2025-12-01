# Implementation Plan

[Overview]
Single sentence describing the overall goal.

Add MCP tools and small utilities to make the existing rocrate MCP server practical to use: provide robust indexed search (keyword + mock semantic), crate file listing, crate download, and entity-based search helpers so LLMs or other clients can discover, inspect, and fetch RO-Crate archives from configured storage backends.

This change extends the current SqliteFTSIndexStore, Indexer, and StorageBackend implementations by exposing new FastMCP tools and zip-inspection utilities. The approach reuses existing code for indexing and storage, keeps semantic search as a deterministic/mock embedding provider (per user preference), and focuses on safety when reading zip contents and bounding downloads to avoid resource exhaustion. The implementation fits into the existing mcp tool model (no additional HTTP/FastAPI endpoints) and is backend-agnostic via the StorageBackend protocol.

[Types]
Single sentence describing the type system changes.

Detailed type definitions, interfaces, enums, or data structures with complete specifications. Include field names, types, validation rules, and relationships.

- FileInfo (Pydantic model; add to src/rocrate_mcp/models.py)
  - path: str — path of the file inside the crate (posix style). Validation: non-empty, normalized (no leading ../)
  - size: int | None — uncompressed size in bytes when available
  - last_modified: datetime | None — last modification time when available
  - compressed_size: int | None — compressed size in archive if available

- CrateFilesResponse (ad-hoc response shape; can be returned as dict or Pydantic model)
  - crate_id: str
  - resource_locator: str
  - files: list[FileInfo]
  - count: int

- CrateDownloadResponse (Pydantic model; add to src/rocrate_mcp/models.py)
  - crate_id: str
  - resource_locator: str
  - content_base64: str — base64-encoded binary content of the crate (warning: memory use)
  - size: int | None — number of bytes

- SearchByEntityRequest (function params; reuse SearchFilter where sensible)
  - type_name: str
  - prop_path: str
  - prop_value: str
  - entry_field: str | None
  - entry_value: str | None
  - exact: bool = True

Validation rules
- FileInfo.path must be non-empty and should not contain path traversal components when returned to clients.
- Downloads larger than configured limit (ROC_MCP_DOWNLOAD_SIZE_LIMIT_MB) should be rejected with a clean error message.

[Files]
Single sentence describing file modifications.

Detailed breakdown:
- New files to be created
  - src/rocrate_mcp/utils/zip_utils.py — utilities to inspect and optionally read members from a zip stream without loading entire archive into memory; functions:
    - list_files_from_zip_stream(stream: BinaryIO) -> list[dict]
      - returns list of dicts like {"path": "...", "size": int|None, "compressed_size": int|None, "last_modified": str|None}
    - read_member_from_zip_stream(stream: BinaryIO, member_path: str) -> bytes
      - reads a single member's raw bytes into memory (used optionally for partial extraction)
    - Implementation notes: reuse pattern from utils/zip_reader.py (write stream to temp file, open with zipfile.ZipFile), ensure safe member names and preserve posix-style paths. Leave temp dir for caller to clean or ensure deletion after reading.

- Existing files to be modified
  - src/rocrate_mcp/models.py
    - Add FileInfo and CrateDownloadResponse Pydantic models (with docstrings and type hints).
  - src/rocrate_mcp/config.py
    - Add integer setting: download_size_limit_mb: int = 50 (env var ROC_MCP_DOWNLOAD_SIZE_LIMIT_MB)
  - src/rocrate_mcp/main.py
    - Add MCP tools (mcp.tool() decorated functions):
      - list_crate_files(crate_id: str, ctx: Context | None = None) -> dict
      - download_crate(crate_id: str, ctx: Context | None = None) -> dict
      - search_by_entity(type_name: str, prop_path: str, prop_value: str, entry_field: str | None = None, entry_value: str | None = None, exact: bool = True, limit: int = 50, offset: int = 0, ctx: Context | None = None) -> dict
    - Improve validation and docstrings for existing tools search_index and get_crate.

- Files to be deleted or moved
  - None

- Configuration file updates
  - src/rocrate_mcp/config.py: add download_size_limit_mb: int = 50, document env ROC_MCP_DOWNLOAD_SIZE_LIMIT_MB in Config or README.

[Functions]
Single sentence describing function modifications.

Detailed breakdown:
- New functions
  - list_files_from_zip_stream(stream: BinaryIO) -> list[dict]
    - file: src/rocrate_mcp/utils/zip_utils.py
    - purpose: write the stream to a temp file, open with zipfile.ZipFile, gather file entries (path, size, compressed_size, mtime) while ensuring member names are safe.
  - read_member_from_zip_stream(stream: BinaryIO, member_path: str) -> bytes
    - file: src/rocrate_mcp/utils/zip_utils.py
    - purpose: read and return the raw bytes for a specific member in the zip archive; raise FileNotFoundError if not present.

  - mcp tool: list_crate_files(crate_id: str, ctx: Context|None = None) -> dict
    - file: src/rocrate_mcp/main.py
    - signature: async def list_crate_files(crate_id: str, ctx: Context | None = None) -> dict[str, Any]
    - purpose: fetch index entry via mcp.state.store.get(crate_id); if not present return {"error": "not_found"} or empty dict; fetch binary stream via mcp.state.backend.get_resource_stream(entry.resource_locator); call list_files_from_zip_stream to get file listing; map to FileInfo model and return {"crate_id": ..., "resource_locator": ..., "count": N, "files": [...]}
    - error handling: return user-friendly error dict on missing backend, missing resource, or extraction issue.

  - mcp tool: download_crate(crate_id: str, ctx: Context|None = None) -> dict
    - file: src/rocrate_mcp/main.py
    - signature: async def download_crate(crate_id: str, ctx: Context | None = None) -> dict[str, Any]
    - purpose: fetch index entry, call backend.get_resource_stream(locator) -> stream; determine size if possible (entry.resource_size or stream size); enforce settings.download_size_limit_mb; read bytes and return base64 encoded string and size.
    - safety: if size unknown, read up to limit+1 bytes to decide whether to allow; if limit exceeded return informative error.

  - mcp tool: search_by_entity(...)
    - file: src/rocrate_mcp/main.py
    - purpose: call store.find_crates_by_entity_property or find_crates_by_entity_and_entry depending on whether entry filters are provided; apply pagination and return list of crate metadata (crate_id, title, description, resource_locator).

- Modified functions
  - search_index (src/rocrate_mcp/main.py)
    - tighten validation of field_filters (ensure dict[str,str] or return 400-like error), document behavior when FTS unavailable, keep mock semantic mode unchanged.
  - get_crate (src/rocrate_mcp/main.py)
    - ensure return shape stable: use entry.dict() but also ensure datetime values are isoformat strings and non-serializable values are converted.

- Removed functions
  - None

[Classes]
Single sentence describing class modifications.

Detailed breakdown:
- New classes
  - FileInfo (Pydantic model) in src/rocrate_mcp/models.py
  - CrateDownloadResponse (Pydantic model) in src/rocrate_mcp/models.py

- Modified classes
  - Settings (src/rocrate_mcp/config.py): add property download_size_limit_mb: int

- Removed classes
  - None

[Dependencies]
Single sentence describing dependency modifications.

Details of new packages, version changes, and integration requirements.
- No new third-party packages required. Use Python stdlib modules (zipfile, tempfile, base64) and existing packages (pydantic, azure.storage.blob if Azure backend used).
- If later upgrading to real embeddings, add optional dependency (e.g., openai) and configuration; not part of this plan.

[Testing]
Single sentence describing testing approach.

Test file requirements, existing test modifications, and validation strategies.

- Add tests/test_zip_utils.py
  - test_list_files_from_zip_stream: create temporary zip with known file paths and metadata, feed through a BytesIO stream or filesystem-backed stream (via FilesystemStorageBackend), assert returned file list contains expected paths and sizes.
  - test_read_member_from_zip_stream: assert that reading a specific member returns expected bytes and that not-found raises FileNotFoundError.

- Extend tests/test_mcp_tools.py
  - test_list_crate_files_tool: insert or index a sample crate into the SqliteFTSIndexStore (or stub store.get to return IndexEntry with resource_locator pointing to a temp zip), call list_crate_files tool function directly, assert correct response structure.
  - test_download_crate_tool: similar to above, call download_crate and assert base64-decoded data matches original zip bytes; test limit enforcement by setting a low ROC_MCP_DOWNLOAD_SIZE_LIMIT_MB.
  - test_search_by_entity_tool: create entries with materialized entities and assert search_by_entity returns expected crate_ids.

- Use pytest fixtures from tests/conftest.py where available (temporary DB, FilesystemStorageBackend). Tests should be deterministic and avoid network access.

[Implementation Order]
Single sentence describing the implementation sequence.

Numbered steps showing the logical order of changes to minimize conflicts and ensure successful integration.

1. Add configuration option: add download_size_limit_mb: int = 50 to src/rocrate_mcp/config.py and document ROC_MCP_DOWNLOAD_SIZE_LIMIT_MB in README.  
2. Create utilities module: add src/rocrate_mcp/utils/zip_utils.py implementing list_files_from_zip_stream and read_member_from_zip_stream (reusing safe-extract patterns from utils/zip_reader.py).  
3. Add Pydantic models: update src/rocrate_mcp/models.py to include FileInfo and CrateDownloadResponse with docstrings and type hints.  
4. Implement MCP tools: in src/rocrate_mcp/main.py add mcp tools list_crate_files, download_crate, search_by_entity, and improve validation for search_index/get_crate.  
5. Add tests: create tests/test_zip_utils.py and extend tests/test_mcp_tools.py with new cases for listing and downloading crates; run pytest and fix issues.  
6. Update README.md and docs to document new MCP tools and the new env option for download limit.  
7. (Optional/follow-up) Implement streaming downloads or signed URLs for large crates; add server-side progress reporting.

Notes / Risk mitigations
- Download memory usage: base64-encoding full crate content is simple but not suitable for large archives. The download_size_limit_mb setting mitigates this risk; document and enforce limit. Future improvement: return signed URLs or stream multipart responses.
- Zip member safety: zip_utils must prevent path traversal and only return safe posix-style member names.
- Backend compatibility: backends that return BytesIO must be supported; ensure list_files_from_zip_stream accepts any file-like object.

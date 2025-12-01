# ro-crate-mcp-server

[![version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://img.shields.io/badge/version-0.1.0-blue.svg) [![python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/) [![build](https://img.shields.io/badge/build-unknown-lightgrey.svg)](https://github.com/)

MCP server for RO-crates

Overview
--------
This project implements a small MCP (Model Context Protocol) server that indexes RO-Crate zip archives by reading the embedded JSON-LD metadata file (typically named `ro-crate-metadata.json` or `ro-crate.json`) without unpacking the entire archive. The server exposes a FastAPI-based HTTP API to list, search, and fetch crate metadata.

Development / Quickstart
----------

Development (recommended)

1. Install build & development dependencies using uv (reads pyproject.toml):

   uv install

2. Install in editable mode for development:

   pip install -e .[dev]

Install as a package (normal)

If you want to install the package normally (when published to PyPI) run:

    pip install rocrate-mcp

Note: this package is not published to PyPI yet. To install from the local source tree use:

    pip install .

Next steps (common)

1. Copy `.env.example` to `.env` and set the ROC_MCP_AZURE_CONNECTION_STRING and ROC_MCP_AZURE_CONTAINER environment variables if using Azure blob storage (or set ROC_MCP_BACKEND=filesystem and ROC_MCP_FILESYSTEM_ROOT for filesystem).

2. Run the server locally:

   uv run uvicorn rocrate_mcp.main:app --reload --port 8000

3. Endpoints (all prefixed with /api/v1):
   - GET /api/v1/health — health check
   - POST /api/v1/search — search indexed crates (body: SearchFilter)
   - GET /api/v1/crate/{crate_id} — fetch a crate's indexed metadata
   - GET /api/v1/search/by-entity — find crate ids by entity property

Storage backends
----------------
This project currently supports two storage backends. Selection is controlled by ROC_MCP_BACKEND (filesystem | azure | none).

- Filesystem (local): a simple local filesystem backend implemented in `src/rocrate_mcp/rocrate_storage/filesystem.py`. Configure with:
  - ROC_MCP_BACKEND=filesystem
  - ROC_MCP_FILESYSTEM_ROOT — path to the root directory containing RO-Crate zip files
  - ROC_MCP_FILESYSTEM_ROOT_PREFIX — optional logical prefix inside the root
  - ROC_MCP_FILESYSTEM_DEFAULT_SUFFIXES — comma-separated suffixes (e.g. .zip)

- Azure Blob Storage: an Azure-backed backend implemented in `src/rocrate_mcp/rocrate_storage/azure_blob.py`. Configure with:
  - ROC_MCP_BACKEND=azure
  - ROC_MCP_AZURE_CONNECTION_STRING — Azure connection string
  - ROC_MCP_AZURE_CONTAINER — Azure container name

Configuration
-------------
Configuration is available via environment variables with prefix ROC_MCP_. See `src/rocrate_mcp/config.py` for keys. Common options include:

- ROC_MCP_INDEX_MODE: eager (default) or hybrid
- ROC_MCP_INDEXED_DB_PATH: optional sqlite file for index persistence (recommended for production)
- ROC_MCP_BACKEND: filesystem | azure | none
- ROC_MCP_FILESYSTEM_ROOT / ROC_MCP_FILESYSTEM_ROOT_PREFIX / ROC_MCP_FILESYSTEM_DEFAULT_SUFFIXES (when using filesystem backend)
- ROC_MCP_AZURE_CONNECTION_STRING / ROC_MCP_AZURE_CONTAINER (when using azure backend)
- ROC_MCP_FIELDS_TO_INDEX (or legacy ROC_MCP_ROC_FIELDS_TO_INDEX) — comma-separated fields to index
- ROC_MCP_EMBEDDINGS_PROVIDER / ROC_MCP_EMBEDDINGS_API_KEY — placeholders for embedding provider configuration

Design notes
------------
- Default indexing strategy is eager (full scan at startup). A hybrid mode is supported conceptually (lightweight listing + lazy metadata fetch) but not yet fully implemented.
- The code uses a pluggable StorageBackend protocol so additional backends (S3, filesystem) can be added without changing index semantics.
- The canonical index store is the sqlite-backed `SqliteFTSIndexStore`; the package exposes it via `rocrate_mcp.index.storage.store.IndexStore`.
- The API router is mounted under the `/api/v1` prefix (see `src/rocrate_mcp/main.py`).

Testing
-------
Run tests with:

    pytest

Licensing
---------
MIT

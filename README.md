# ro-crate-mcp-server
MCP server for RO-crates

Overview
--------
This project implements a small MCP (Model Context Protocol) server that indexes RO-Crate zip archives by reading the embedded JSON-LD metadata file (typically named `ro-crate-metadata.json` or `ro-crate.json`) without unpacking the entire archive. The server exposes a FastAPI-based HTTP API to list, search, and fetch crate metadata.

Quickstart
----------
1. Install dependencies (recommend using a virtualenv or conda env):

   pip install -e .[dev]

2. Copy `.env.example` to `.env` and set the RMP_AZURE_CONNECTION_STRING and RMP_AZURE_CONTAINER environment variables if using Azure blob storage.

3. Run the server locally:

   uvicorn rocrate_mcp.main:app --reload --port 8000

4. Endpoints (prefixed with /api):
   - GET /api/health — health check
   - POST /api/search — search indexed crates (body: SearchFilter)
   - GET /api/crate/{crate_id} — fetch a crate's indexed metadata

Configuration
-------------
Configuration is available via environment variables with prefix RMP_. See `rocrate_mcp/config.py` for keys. Example:

- RMP_INDEX_MODE: eager (default) or hybrid
- RMP_AZURE_CONNECTION_STRING: Azure connection string (if using Azure backend)
- RMP_AZURE_CONTAINER: Azure container name
- RMP_INDEXED_DB_PATH: optional sqlite file for index persistence

Design notes
------------
- Default indexing strategy is eager (full scan at startup). A hybrid mode is supported conceptually (lightweight listing + lazy metadata fetch) but not yet fully implemented.
- The code uses a pluggable StorageBackend protocol so additional backends (S3, filesystem) can be added without changing index semantics.
- IndexStore currently provides an in-memory store and simple search. Adding sqlite persistence with FTS is a recommended next step.

Testing
-------
Run tests with:

    pytest

Licensing
---------
MIT

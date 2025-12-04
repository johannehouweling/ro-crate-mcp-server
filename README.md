ro-crate-mcp-server
=============

[![version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://img.shields.io/badge/version-0.1.1-blue.svg) [![python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/) [![build](https://img.shields.io/badge/build-unknown-lightgrey.svg)](https://github.com/)

MCP server for indexing and querying RO-crates.

--------

This repository provides a FastMCP-based server exposing tools to list and inspect indexed RO-crates, perform semantic search (optional), and list storage resources from pluggable backends (filesystem or Azure Blob Storage). The server is configured via a pydantic Settings class that reads environment variables prefixed with ROC_MCP_. See src/rocrate_mcp/config.py for the authoritative mapping.

Highlights
- FastMCP-based server (use `mcp` CLI for development and invocation)
- Pluggable storage backends: filesystem, azure, or none (in-memory/skippable)
- SQLite FTS-backed index (SqliteFTSIndexStore) for fast text search
- Tools exposed via MCP: list_all_indexed_crates, semantic_search, get_crate_metadata, storage_list_resources

Requirements
- Python >= 3.12
- Recommended dependencies are declared in pyproject.toml. Important packages include:
  - mcp[cli] (FastMCP CLI)
  - rocrate
  - rdflib, sqlalchemy, aiosqlite
  - azure-storage-blob (optional, for Azure backend)

Installation

Clone and install editable (development) environment:

    git clone <repo-url>
    cd ro-crate-mcp-server
    python -m pip install -e .[dev]

Or install the package for normal usage:

    python -m pip install .

Development quickstart

1) Configure environment

Copy the example env file and edit values as needed:

    cp src/rocrate_mcp/.env.example .env

The server reads environment variables with the ROC_MCP_ prefix (see Configuration below).

2) Run development server with MCP

The recommended development flow is to use the mcp CLI which understands the FastMCP lifespan and tooling:

    # Run the app in development mode (auto-reload, lifecycle invoked)
    mcp dev

You can also run the server directly (production/one-off):

    python -m rocrate_mcp.main

Configuration (environment variables)

All runtime configuration is provided via pydantic-settings Settings and environment variables with prefix ROC_MCP_. The authoritative names are defined in src/rocrate_mcp/config.py; the most commonly used variables are:

- ROC_MCP_INDEX_MODE: 'eager' | other (default: eager) — controls whether the index is built during server lifespan startup
- ROC_MCP_STORAGE_BACKEND: 'sqlite+fts' | 'rdflib' — internal index storage selection
- ROC_MCP_BACKEND: 'filesystem' | 'azure' | 'http' | 'none' — select which crate storage backend to use
- ROC_MCP_FILESYSTEM_ROOT: Path for filesystem backend root (required when BACKEND=filesystem)
- ROC_MCP_FILESYSTEM_ROOT_PREFIX: Optional prefix for filesystem locations
- ROC_MCP_FILESYSTEM_DEFAULT_SUFFIXES: Comma-separated suffixes (default: .zip)
- ROC_MCP_AZURE_CONNECTION_STRING: Azure connection string (required when BACKEND=azure)
- ROC_MCP_AZURE_CONTAINER: Azure container name (required when BACKEND=azure)
- ROC_MCP_INDEXED_DB_PATH: Path to the sqlite index DB file (defaults to rocrate_index.sqlite)
- ROC_MCP_FIELDS_TO_INDEX: Comma-separated list of fields from ro-crate to index (see Settings.get_fields_to_index())
- ROC_MCP_EMBEDDINGS_PROVIDER: 'local' | 'openai' | 'none' (controls semantic embeddings provider)
- ROC_MCP_EMBEDDINGS_API_KEY: API key for external embedding providers (kept as SecretStr)

Note: Some historical/alternate env names are tolerated by the Settings class (e.g. ROC_MCP_ROC_FIELDS_TO_INDEX or ROC_MCP_ROC_MCP_FIELDS_TO_INDEX) — see src/rocrate_mcp/config.py for details.

Storage backends

Filesystem backend (local directory)

Example .env settings for filesystem backend:

    ROC_MCP_BACKEND=filesystem
    ROC_MCP_FILESYSTEM_ROOT=C:/data/ro-crates
    ROC_MCP_FILESYSTEM_DEFAULT_SUFFIXES=.zip,.tar.gz

Azure Blob Storage backend

Example .env settings for Azure:

    ROC_MCP_BACKEND=azure
    ROC_MCP_AZURE_CONNECTION_STRING=<your-azure-connection-string>
    ROC_MCP_AZURE_CONTAINER=<container-name>

HTTP directory-index based backend

The HTTPStorageBackend can be used to index crates exposed via a web server that provides directory-style HTML listings (e.g. Apache or Nginx autoindex). It treats the configured base URL as a root and discovers files and subdirectories by parsing anchor links in listing pages. Downloaded files are streamed and treated similarly to filesystem/azure backends (for example, ZIP archives are extracted to locate ro-crate-metadata.json).

Example .env settings to configure an HTTP backend (the project reads ROC_MCP_ prefixed vars; adapt names to your deployment):

    ROC_MCP_BACKEND=http
    ROC_MCP_HTTP_BASE_URL=https://example.org/
    ROC_MCP_HTTP_ROOT_PREFIX=optional/prefix/
    ROC_MCP_HTTP_DEFAULT_SUFFIXES=.zip,.tar.gz
    ROC_MCP_HTTP_TIMEOUT=10

If ROC_MCP_BACKEND is not set (or set to 'none'), the server will not wire a storage backend — useful for testing non-storage features.

MCP Tools / API

This project exposes a small set of tooling via FastMCP. The tools are registered in src/rocrate_mcp/tools.py and can be invoked using the `mcp` CLI once the server is running (or with the `mcp call` command against the package when appropriate).

Available tools (high level)
- list_all_indexed_crates(limit: int = 100, offset: int = 0)
  Returns paged list of indexed crate IDs and some metadata.

- semantic_search(query: str)
  Performs semantic search over the indexed content (requires embeddings provider configured).

- get_crate_metadata(crate_id: str)
  Returns the top-level ro-crate-metadata.json for a given crate (convenience endpoint).

- storage_list_resources(prefix: str | None = None, suffixes: list[str] | None = (".zip",), limit: int = 100, offset: int = 0)
  Lists raw resources from the configured storage backend.

Example mcp CLI usage (after starting the server with `mcp dev`):

    mcp call rocrate-mcp.list_all_indexed_crates --args '{"limit":10}'
    mcp call rocrate-mcp.get_crate_metadata --args '{"crate_id":"some-crate-id"}'
    mcp call rocrate-mcp.storage_list_resources --args '{"prefix":"2025/","limit":50}'

Testing

Run tests with pytest (dev extras required):

    python -m pip install -e .[dev]
    pytest -q

You can execute a single test module:

    pytest tests/test_zip_utils.py -q

Note: Integration tests requiring Azure or other external services may need additional env configuration and are not covered by the default test suite.

Contributing

- Use conventional commit messages. The preferred commit message for this change was:

    docs(readme): rewrite README for FastMCP usage

- Keep README and pyproject metadata in sync. If you add env-driven features, update src/rocrate_mcp/.env.example and this README.

License

This project is MIT licensed. See the LICENSE file for details.

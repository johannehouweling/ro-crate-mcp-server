from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    # Indexing mode: 'eager' will build the index on startup, 'hybrid' or other modes may defer
    index_mode: str = "eager"

    # Explicit backend selector: 'filesystem' | 'azure' | 'none'
    backend: str = "filesystem"
    default_backend_id: str = "azure_default"

    # Azure connection via connection string
    azure_connection_string: str | None = None
    azure_container: str | None = None

    # Filesystem backend settings
    filesystem_root: str | None = None
    filesystem_root_prefix: str | None = None
    # Comma-separated list of suffixes (e.g. ".zip,.tar.gz"). If omitted defaults to ".zip".
    filesystem_default_suffixes: str | None = ".zip"

    # General options
    indexed_db_path: str | None = None  # path to sqlite file for persistence
    # historical field name `roc_fields_to_index` kept for backward compatibility
    roc_fields_to_index: str = ""  # list of fields from rocrate to index (comma-separated)
    # new canonical env name: FIELDS_TO_INDEX (maps to ROC_MCP_FIELDS_TO_INDEX)
    fields_to_index: str | None = None
    # accept possible raw env var key that may appear in some environments
    roc_mcp_fields_to_index: str | None = None

    # Embeddings / semantic search placeholders
    embeddings_provider: str | None = None
    embeddings_api_key: SecretStr | None = None

    # Limits
    max_list_limit: int = 1000  # configurable server-side max for paged listing

    class Config:
        env_prefix = "ROC_MCP_"
        env_file = ".env"

    def get_fields_to_index(self) -> list[str]:
        """Return the configured fields to index as a list of strings.

        Precedence (highest->lowest):
        - ROC_MCP_FIELDS_TO_INDEX -> fields_to_index
        - ROC_MCP_ROC_MCP_FIELDS_TO_INDEX -> roc_mcp_fields_to_index (some env loaders may not strip prefix)
        - ROC_MCP_ROC_FIELDS_TO_INDEX -> roc_fields_to_index
        """
        raw = ""
        if self.fields_to_index:
            raw = self.fields_to_index
        elif self.roc_mcp_fields_to_index:
            raw = self.roc_mcp_fields_to_index
        elif self.roc_fields_to_index:
            raw = self.roc_fields_to_index
        raw = raw or ""
        return [p.strip() for p in raw.split(",") if p.strip()]

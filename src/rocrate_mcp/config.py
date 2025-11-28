from pydantic import BaseSettings


class Settings(BaseSettings):
    index_mode: str = "eager"  # or 'hybrid'
    default_backend_id: str = "azure_default"
    # Azure connection via connection string
    azure_connection_string: str|None = None
    azure_container: str|None = None
    # General options
    indexed_db_path: str|None = None  # path to sqlite file for persistence
    roc_fields_to_index: str|None = None  # list of fields from rocrate to index

    class Config:
        env_prefix = "RMP_"
        env_file = ".env"

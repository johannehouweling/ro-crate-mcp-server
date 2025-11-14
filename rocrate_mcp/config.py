from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    index_mode: str = "eager"  # or 'hybrid'
    default_backend_id: str = "azure_default"
    # Azure connection via connection string
    azure_connection_string: Optional[str] = None
    azure_container: Optional[str] = None
    # General options
    indexed_db_path: Optional[str] = None  # path to sqlite file for persistence

    class Config:
        env_prefix = "RMP_"
        env_file = ".env"

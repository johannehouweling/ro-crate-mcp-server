# tools are defined in tools.py
from rocrate_mcp import tools  # noqa: F401
from rocrate_mcp.roc_mcp import mcp

if __name__ == "__main__":
    mcp.run()

import asyncio

from rocrate_mcp import main


def test_search_index_basic():
    # call the MCP tool directly
    res = asyncio.run(main.search_index(q=None, mode="keyword", limit=1, offset=0))
    assert isinstance(res, dict)
    assert "count" in res and "results" in res


def test_get_crate_not_found():
    res = asyncio.run(main.get_crate("this-crate-id-does-not-exist"))
    # when not found we return an empty dict
    assert res == {}

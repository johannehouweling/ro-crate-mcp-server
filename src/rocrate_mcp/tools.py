import asyncio
import json
from typing import Any

from mcp.server.fastmcp import Context

from rocrate_mcp.roc_mcp import mcp
from rocrate_mcp.models import SearchFilter


@mcp.tool()
async def list_all_indexed_crates(
    limit: int = 100,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    settings_local = mcp.state.settings
    try:
        requested_limit = max(1, int(limit or 0))
    except Exception:
        requested_limit = 100
    max_limit = max(1, int(getattr(settings_local, "max_list_limit", 1000)))
    hard_cap = 10000
    effective_limit = min(requested_limit, max_limit, hard_cap)

    # call async listentries and await result
    entries = await mcp.state.store.listentries(offset=offset, limit=effective_limit)
    result = {}
    crate_ids = [ent.crate_id for ent in entries if ent is not None]
    crate_names = [e.name for e in entries if e is not None]
    total = len(crate_ids)
    page = crate_ids[offset : offset + effective_limit]
    truncated = offset + effective_limit < total

    meta = {
        "requested_limit": requested_limit,
        "effective_limit": effective_limit,
        "offset": offset,
        "total": total,
        "truncated": truncated,
    }
    return {"count": len(page), "crate_ids": page, "meta": meta}

@mcp.tool()
async def semantic_search(query:str)-> dict[str,Any]:
    """Perform a semantic search over indexed crates using the given query string."""
    results = await mcp.state.store.semantic_search(
        query=query,
        mode="semantic",
        limit=10,
        offset=0,
    )
    return dict(results=results)


@mcp.tool()
async def get_crate_metadata(crate_id: str) -> dict[str, Any]:
    """Return the top-level ro-crate-metadata.json content for the given crate_id.

    This is a convenience endpoint to retrieve the main metadata without downloading
    and extracting the entire crate.
    """
    entry = await mcp.state.store.get(crate_id)
    if entry is None:
        return {}
    files = mcp.state.backend.get_extracted_file_paths(entry.resource_locator)
    ro_crate_metadata_file_path = [
        a
        for a in mcp.state.backend.get_extracted_file_paths(entry.resource_locator)
        if a.name == "ro-crate-metadata.json"
    ][0]
    with ro_crate_metadata_file_path.open("rb") as f:
        json_metadata_file = json.load(f)
    for file in files:
        file.unlink(missing_ok=True)  # Clean up temp files
    if not json_metadata_file:
        return {}
    return json_metadata_file or {}


@mcp.tool()
async def storage_list_resources(
    prefix: str | None = None,
    suffixes: list[str] | None = (".zip",),
    limit: int = 100,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List raw storage resources from the configured backend (read-only)."""
    backend_local = mcp.state.backend
    if backend_local is None:
        return {"count": 0, "items": []}

    # enforce sensible bounds
    requested_limit = max(1, int(limit or 0))
    max_limit = max(1, int(getattr(mcp.state.settings, "max_list_limit", 1000)))
    hard_cap = 10000
    max_limit = min(max_limit, hard_cap)

    effective_limit = min(requested_limit, max_limit)

    it = backend_local.list_resources(prefix=prefix, suffixes=suffixes)
    # NOTE: don't convert `it` to list() here or you'll exhaust the iterator
    items: list[dict[str, Any]] = []
    skipped = 0
    total_seen = 0
    for res in it:
        total_seen += 1
        if skipped < offset:
            skipped += 1
            continue
        if len(items) >= effective_limit:
            break
        items.append(
            {
                "locator": res.locator,
                "size": res.size,
                "last_modified": str(res.last_modified),
            }
        )

    meta = {
        "requested_limit": requested_limit,
        "effective_limit": effective_limit,
        "max_limit": max_limit,
        "truncated": requested_limit > effective_limit,
        "total_seen": total_seen,
    }
    return {"count": len(items), "items": items, "meta": meta}

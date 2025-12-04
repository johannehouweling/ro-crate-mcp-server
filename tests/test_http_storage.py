from __future__ import annotations

import io
import zipfile
from typing import Iterator

import requests

from rocrate_mcp.rocrate_storage.http_listing import HTTPStorageBackend


class DummyResponse:
    def __init__(self, *, status_code=200, text="", content=b""):
        self.status_code = status_code
        self.text = text
        self.content = content

    def close(self):
        pass


class DummySession:
    def __init__(self, pages: dict[str, DummyResponse]):
        self.pages = pages

    def get(self, url, stream=False, timeout=None):
        # Return the matching page (exact match on url) or a 404 response
        return self.pages.get(url, DummyResponse(status_code=404))

    def head(self, url, timeout=None):
        r = self.pages.get(url, DummyResponse(status_code=404))
        # convert to a minimal head-like response
        class H:
            def __init__(self, headers, status_code):
                self.headers = headers
                self.status_code = status_code

        if r.status_code != 200:
            return H({}, 404)
        # try to read length from content if present
        return H({"Content-Length": str(len(r.content))}, 200)


def make_zip_with_metadata() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("ro-crate-metadata.json", "{\"name\": \"test\"}")
    return bio.getvalue()


def test_list_and_stream_basic():
    base = "https://example.org/data/"
    # Directory listing with two files and one subdir
    html_root = '<html><body>\n<a href="file1.zip">file1.zip</a>\n<a href="subdir/">subdir/</a>\n</body></html>'
    html_sub = '<html><body>\n<a href="nested.zip">nested.zip</a>\n</body></html>'

    zbytes = make_zip_with_metadata()

    pages = {
        base: DummyResponse(status_code=200, text=html_root),
        base + "subdir/": DummyResponse(status_code=200, text=html_sub),
        base + "file1.zip": DummyResponse(status_code=200, content=zbytes),
        base + "subdir/nested.zip": DummyResponse(status_code=200, content=zbytes),
    }

    session = DummySession(pages)
    backend = HTTPStorageBackend(base_url=base, root_prefix=None, session=session)

    items = list(backend.list_resources())
    locators = sorted([it.locator for it in items])
    assert "file1.zip" in locators
    assert "subdir/nested.zip" in locators

    # Stream a file and check extraction
    paths = backend.get_extracted_file_paths("file1.zip")
    assert paths is not None
    assert any(p.name == "ro-crate-metadata.json" for p in paths)

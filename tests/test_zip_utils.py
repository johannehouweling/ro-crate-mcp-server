import io
import zipfile

from rocrate_mcp.utils.zip_utils import list_zip_contents, read_file_from_zip_stream


def _make_zip_bytes(entries: dict[str, bytes]) -> io.BytesIO:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    b.seek(0)
    return b


def test_list_zip_contents():
    entries = {"foo.txt": b"hello", "dir/bar.txt": b"world"}
    stream = _make_zip_bytes(entries)
    res = list_zip_contents(stream)
    names = [e["name"] for e in res]
    assert "foo.txt" in names
    assert "dir/bar.txt" in names

    # sizes reported should match the content lengths
    size_map = {e["name"]: e["size"] for e in res}
    assert size_map["foo.txt"] == len(entries["foo.txt"])
    assert size_map["dir/bar.txt"] == len(entries["dir/bar.txt"])


def test_read_file_from_zip_stream():
    entries = {"a.txt": b"abc", "b.bin": b"\x00\x01"}
    stream = _make_zip_bytes(entries)
    data = read_file_from_zip_stream(stream, "a.txt")
    assert data == entries["a.txt"]

    # requesting a missing member returns None
    stream2 = _make_zip_bytes(entries)
    assert read_file_from_zip_stream(stream2, "missing.txt") is None

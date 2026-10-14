"""Persistent Chroma ontology loading against a cached synthetic tarball.

These tests exercise actual collection creation and copying without downloads.
An extraction guard bounds regressions that would recursively reload the source.
"""

import tarfile

import pytest

chromadb = pytest.importorskip("chromadb")
numpy = pytest.importorskip("numpy")

from lobster.vector.backends import chromadb_backend as cb  # noqa: E402

COLLECTION = "synthetic_v2024_01"
DOCS = 1200
DIM = 32


class _Runaway(BaseException):
    """The loader's fallback exception handler must not swallow the test guard."""


@pytest.fixture(scope="module")
def tarball(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_ontology")
    source = root / "synthetic_openai_32"
    client = chromadb.PersistentClient(path=str(source))
    try:
        collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        vectors = (
            numpy.random.default_rng(3).standard_normal((DOCS, DIM)).astype("float32")
        )
        collection.add(
            ids=[f"SYN:{i:06d}" for i in range(DOCS)],
            embeddings=vectors.tolist(),
            documents=[f"term {i}" for i in range(DOCS)],
            metadatas=[{"ontology_id": f"SYN:{i:06d}"} for i in range(DOCS)],
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    path = root / "synthetic_openai_32.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    return path


@pytest.fixture
def backend(monkeypatch, tmp_path, tarball):
    cache = tmp_path / "ontology_cache"
    cache.mkdir()
    (cache / tarball.name).write_bytes(tarball.read_bytes())
    monkeypatch.setattr(cb, "ONTOLOGY_CACHE_DIR", cache)
    monkeypatch.setattr(
        cb, "ONTOLOGY_TARBALLS", {COLLECTION: f"https://example.invalid/{tarball.name}"}
    )

    def no_download(*args, **kwargs):
        pytest.fail("Synthetic ontology must load from cache without network access")

    monkeypatch.setattr(cb, "_download_with_progress", no_download)
    instance = cb.ChromaDBBackend(persist_path=str(tmp_path / "store"))
    yield instance
    if instance._client is not None:
        close = getattr(instance._client, "close", None)
        if callable(close):
            close()


@pytest.fixture
def extractions(monkeypatch):
    """Count extractions and abort recursion before it can consume large resources."""
    calls = {"n": 0}
    real = tarfile.TarFile.extractall

    def spy(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 3:
            raise _Runaway(
                "ontology loader extracted the tarball more than three times"
            )
        return real(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", spy)
    return calls


def test_first_search_loads_the_tarball_exactly_once(backend, extractions):
    result = backend.search(COLLECTION, [0.1] * DIM, n_results=3)

    assert extractions["n"] == 1
    assert len(result["ids"][0]) == 3
    assert backend.count(COLLECTION) == DOCS
    assert backend._ensure_ontology_data(COLLECTION) is True
    assert extractions["n"] == 1  # Populated collections do not re-extract.


def test_load_uses_bounded_pages(backend, extractions, monkeypatch):
    from chromadb.api.models.Collection import Collection

    rows = []
    real_add = Collection.add

    def counting_add(self, *args, **kwargs):
        rows.append(len(kwargs["ids"]))
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "add", counting_add)
    assert backend._ensure_ontology_data(COLLECTION) is True
    assert extractions["n"] == 1
    assert rows == [500, 500, 200]
    assert max(rows) <= cb._ONTOLOGY_LOAD_PAGE_SIZE
    assert sum(rows) == DOCS


def test_source_client_is_released_after_the_load(backend, extractions):
    from chromadb.api.shared_system_client import SharedSystemClient

    before = len(SharedSystemClient._identifier_to_system)
    assert backend._ensure_ontology_data(COLLECTION) is True
    # Only the backend's destination client may remain in Chroma's shared cache.
    assert len(SharedSystemClient._identifier_to_system) == before + 1


def test_failed_load_discards_partial_collection_and_can_retry(
    backend, extractions, monkeypatch
):
    from chromadb.api.models.Collection import Collection
    from chromadb.api.shared_system_client import SharedSystemClient

    before = len(SharedSystemClient._identifier_to_system)
    real_add = Collection.add
    calls = {"n": 0}

    def failing_add(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "add", failing_add)
    assert backend._ensure_ontology_data(COLLECTION) is False
    assert not backend.collection_exists(COLLECTION)
    assert len(SharedSystemClient._identifier_to_system) == before + 1
    monkeypatch.setattr(Collection, "add", real_add)
    assert backend._ensure_ontology_data(COLLECTION) is True
    assert backend.count(COLLECTION) == DOCS
    assert extractions["n"] == 2
    assert len(SharedSystemClient._identifier_to_system) == before + 1

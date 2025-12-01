from __future__ import annotations

import json
import os
from typing import Any, Iterable, List

from rdflib import ConjunctiveGraph, Graph, URIRef
from rdflib.namespace import RDF


class RDFIndexStore:
    """Simple RDF-backed global index using rdflib.

    - By default this opens a SQLAlchemy-backed ConjunctiveGraph for persistence
      when a sqlite_url is provided (e.g. "sqlite:///data/rdflib_store.db").
    - Falls back to an in-memory ConjunctiveGraph if initialization fails.
    - Each crate's triples are kept in a named graph with identifier urn:rocrate:{crate_id}
      so removal is possible. Global queries run against the ConjunctiveGraph union.
    """

    def __init__(self, sqlite_url: str | None = "sqlite:///data/rdflib_store.db") -> None:
        self.sqlite_url = sqlite_url
        self._cg: ConjunctiveGraph | None = None

        # try to ensure parent dir exists when using sqlite
        if sqlite_url and sqlite_url.startswith("sqlite://"):
            # derive file path if provided as sqlite:///path
            parts = sqlite_url.split("sqlite:///", 1)
            if len(parts) == 2:
                path = parts[1]
                try:
                    parent = os.path.dirname(path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                except Exception:
                    pass

        try:
            if sqlite_url:
                # open a SQLAlchemy-backed ConjunctiveGraph
                cg = ConjunctiveGraph(store="SQLAlchemy")
                cg.open(sqlite_url, create=True)
                self._cg = cg
            else:
                raise RuntimeError("no sqlite_url")
        except Exception:
            # fallback to in-memory graph
            try:
                self._cg = ConjunctiveGraph()
            except Exception:
                self._cg = None

    def is_available(self) -> bool:
        return self._cg is not None

    def _crate_graph_uri(self, crate_id: str) -> URIRef:
        return URIRef(f"urn:rocrate:{crate_id}")

    def insert_from_jsonld(self, crate_id: str, jsonld: Any, base: str | None = None) -> None:
        """Parse provided JSON-LD (dict or string) and add triples into the global graph.

        The parsed triples are added into a named graph identified by urn:rocrate:{crate_id}.
        base is used as the publicID/base IRI when parsing JSON-LD so relative URIs are resolved
        according to RO-Crate rules (caller should provide a sensible base if available).
        """
        if not self.is_available():
            return

        tg = Graph()  # temporary graph to parse JSON-LD
        data = jsonld
        if isinstance(jsonld, dict):
            try:
                data = json.dumps(jsonld)
            except Exception:
                data = str(jsonld)

        try:
            tg.parse(data=data, format="json-ld", publicID=base)
        except Exception:
            # if parsing as JSON-LD fails, try to treat data as already-serialized triples
            try:
                tg.parse(data=data)
            except Exception:
                return

        # add triples to named graph for the crate
        g_uri = self._crate_graph_uri(crate_id)
        target = self._cg.get_context(g_uri)
        for s, p, o in tg:
            target.add((s, p, o))

        # also add a lightweight provenance triple linking the crate resource to any top-level subjects
        # this can help removal or provenance queries
        crate_ref = URIRef(f"urn:rocrate:#{crate_id}")
        for s in set(tg.subjects()):
            try:
                target.add((crate_ref, RDF.type, s))
            except Exception:
                continue

    def remove(self, crate_id: str) -> None:
        """Remove all triples associated with the named graph for the given crate_id."""
        if not self.is_available():
            return
        g_uri = self._crate_graph_uri(crate_id)
        try:
            ctx = self._cg.get_context(g_uri)
            ctx.remove((None, None, None))
        except Exception:
            pass

    def query_sparql(self, query: str, init_bindings: dict | None = None) -> list[dict[str, str]]:
        """Run a SPARQL query over the global ConjunctiveGraph and return list of dict rows."""
        if not self.is_available():
            return []
        try:
            res = self._cg.query(query, initBindings=init_bindings or {})
        except Exception:
            return []

        rows: list[dict[str, str]] = []
        # convert each result row into a dict of var -> str(value)
        for r in res:
            d: dict[str, str] = {}
            if hasattr(r, "labels"):
                # ordered dict style
                for v in r.labels:
                    val = r[v]
                    d[v] = str(val) if val is not None else ""
            else:
                # fallback: use index-based mapping
                for i, val in enumerate(r):
                    d[str(i)] = str(val) if val is not None else ""
            rows.append(d)
        return rows

    def find_crates_by_entity_property(self, prop_iri: str, value: str, exact: bool = True) -> list[str]:
        """Convenience wrapper: find crate ids that contain triples where prop_iri has the given value.

        prop_iri should be a full IRI string (e.g. 'http://schema.org/name').
        """
        if not self.is_available():
            return []
        # craft a SPARQL query that searches the union graph and returns distinct named graphs
        # that contain a subject with the requested property/value
        if exact:
            filter_clause = f"FILTER(str(?o) = \"{value}\")"
        else:
            filter_clause = f"FILTER(CONTAINS(LCASE(str(?o)), LCASE(\"{value}\")))"

        query = f"""
        SELECT DISTINCT ?g WHERE {{
          GRAPH ?g {{ ?s <{prop_iri}> ?o . {filter_clause} }}
        }}
        """
        rows = self.query_sparql(query)
        crate_ids: list[str] = []
        for r in rows:
            g = r.get("g") or r.get("0")
            if not g:
                continue
            # expect urn:rocrate:{crate_id} or similar
            if g.startswith("urn:rocrate:"):
                crate_ids.append(g.split("urn:rocrate:", 1)[1])
            else:
                crate_ids.append(g)
        return crate_ids


__all__ = ["RDFIndexStore"]

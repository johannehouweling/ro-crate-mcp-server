from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TermNode:
    term: str
    fuzzy: bool = False


@dataclass
class PhraseNode:
    phrase: str


@dataclass
class FieldNode:
    field: str
    child: TermNode | PhraseNode


QueryNode = TermNode | PhraseNode | FieldNode

# Very small lexer/parser for a limited lucene subset:
# - field:term
# - field:"quoted phrase"
# - "quoted phrase"
# - bareword
# Tokens separated by whitespace. No boolean operators.

FIELD_RE = re.compile(r"^([A-Za-z0-9_.]+):(.+)$")
QUOTED_RE = re.compile(r'^"(.+)"$')


def _make_term_node(s: str) -> TermNode:
    """Create a TermNode and detect trailing '~' for fuzzy matching.

    Examples:
      'test'  -> TermNode(term='test', fuzzy=False)
      'test~' -> TermNode(term='test', fuzzy=True)
    """
    s = s.strip()
    fuzzy = False
    if s.endswith('~'):
        fuzzy = True
        s = s[:-1]
    return TermNode(term=s, fuzzy=fuzzy)


def _split_tokens(q: str) -> list[str]:
    # split on whitespace but keep quoted phrases together
    parts: list[str] = []
    cur = []
    in_quote = False
    for c in q.strip():
        if c == '"':
            cur.append(c)
            in_quote = not in_quote
        elif c.isspace() and not in_quote:
            if cur:
                parts.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur))
    return parts


def parse_query(q: str) -> list[QueryNode]:
    """Parse the query string into a list of QueryNode items.

    This parser accepts both 'field:term' (no space) and 'field: term' (space)
    forms. It keeps quoted phrases together and supports trailing '~' for
    fuzzy terms.
    """
    q = (q or "").strip()
    if not q:
        return []

    tokens = _split_tokens(q)
    nodes: list[QueryNode] = []
    i = 0
    while i < len(tokens):
        t = tokens[i].strip()
        i += 1
        if not t:
            continue

        # Try immediate field:value (no space after colon)
        m = FIELD_RE.match(t)
        if m:
            field = m.group(1)
            rest = m.group(2).strip()
            # check quoted or create term node
            mq = QUOTED_RE.match(rest)
            if mq:
                nodes.append(FieldNode(field=field, child=PhraseNode(mq.group(1))))
            else:
                if rest.startswith('"') and rest.endswith('"'):
                    nodes.append(FieldNode(field=field, child=PhraseNode(rest[1:-1])))
                else:
                    nodes.append(FieldNode(field=field, child=_make_term_node(rest)))
            continue

        # Support 'field:' followed by separate token 'term' (e.g. 'description: Test~')
        if t.endswith(":"):
            field = t[:-1].strip()
            # get next token as the term (if any)
            rest = ""
            if i < len(tokens):
                rest = tokens[i].strip()
                i += 1
            rest = rest.strip()
            if rest:
                mq = QUOTED_RE.match(rest)
                if mq:
                    nodes.append(FieldNode(field=field, child=PhraseNode(mq.group(1))))
                else:
                    if rest.startswith('"') and rest.endswith('"'):
                        nodes.append(FieldNode(field=field, child=PhraseNode(rest[1:-1])))
                    else:
                        nodes.append(FieldNode(field=field, child=_make_term_node(rest)))
            else:
                # empty field value, treat as an empty TermNode for the field
                nodes.append(FieldNode(field=field, child=TermNode("", fuzzy=False)))
            continue

        # quoted phrase
        mq = QUOTED_RE.match(t)
        if mq:
            nodes.append(PhraseNode(mq.group(1)))
            continue

        # bare term (may include trailing ~ for fuzzy)
        nodes.append(_make_term_node(t))

    return nodes

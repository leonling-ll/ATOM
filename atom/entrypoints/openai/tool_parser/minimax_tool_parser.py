# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 tool-call format.

Every tag is prefixed by the ns_token ``]<]minimax[>[``::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">
    ]<]minimax[>[<pname>value]<]minimax[>[</pname>
    ...
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
not a ``name="..."`` attribute. Values: schema type wins, else JSON, else raw
string.

A nested value is spelled out as elements, each tag again behind the
ns_token, the way the chat template's ``to_xml`` renders a call in the
history: an object as one element per key, an array as ``<item>`` elements::

    ]<]minimax[>[<tags>]<]minimax[>[<item>a]<]minimax[>[</item>]<]minimax[>[</tags>

The ns_token is *not* stripped first, though this said so long after the code
stopped doing it: parsing a copy with it deleted made every offset meaningless
against the bytes that arrived.
"""

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

MINIMAX_NS = "]<]minimax[>["

# The ns_token is matched optionally wherever a tag can appear, as DSML does
# with its own marker, rather than deleted from a copy of the text first.
# Deleting it made every offset in the parsed copy meaningless against the
# bytes that actually arrived, which is why `parse` could only ever report
# "content precedes the call" and never "and this follows it".
_NS = r"(?:" + re.escape(MINIMAX_NS) + r")?"

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <invoke name="NAME">" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
_NOT_NESTED = r"(?:(?!" + _NS + r"<invoke\s).)"
# No leading ns_token: `CALL_FILLERS` declares it and `markup_begin` walks
# back over it, so matching it here too is a second reader of one token -- and
# the expensive kind. A pattern opening with an optional group has no fixed
# first byte, so the scan tries every position rather than skipping to the
# next `<`: on 18 KB, 524 us against 3.8, and 262 -> 3.8 for `_PARAM_RE`. The
# other five formats open on a literal and never paid it.
_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*?)"
    + _NS
    + r"</invoke>|"
    + r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
# Ends at the matching close tag, at the next parameter, at the close of the
# invoke, or at end of input. That last alternative is what keeps a value the
# model was cut off inside; without it a `max_tokens` truncation arrived as a
# call with `{}` arguments while Qwen and GLM returned the partial value.
_PARAM_RE = re.compile(
    r"<([\w-]+)>(.*?)"
    r"(?:" + _NS + r"</\1>"
    r"|(?=" + _NS + r"<[\w-]+>)"
    r"|(?=" + _NS + r"</invoke>)"
    r"|\Z)",
    re.DOTALL,
)
_FIRST_TAG_RE = re.compile(r"^" + _NS + r"<([\w-]+)>")
# The model now and then writes the ns_token twice in a row, before a tag or
# before a closer; between elements, any run of them reads as one.
_ELEMENT_RE = re.compile(r"\s*((?:" + re.escape(MINIMAX_NS) + r")+)?<([\w-]+)>")
_ELEMENTS_TAIL_RE = re.compile(r"(?:\s|" + re.escape(MINIMAX_NS) + r")*\Z")
_CONTAINER_TYPES = ("object", "array")


def _element_end(text: str, name: str, start: int) -> tuple[int, int] | None:
    """Where the element whose ``<name>`` opener ends at ``start`` closes: the
    end of its value and the end of its closer, or ``None`` if it does not.

    Depth-aware, because an array of arrays nests ``<item>`` in ``<item>``.
    """
    depth = 1
    tag = re.compile(r"<(/?)" + re.escape(name) + r">")  # re caches it
    for m in tag.finditer(text, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            close = m.start()
            value_end = close
            if text.endswith(MINIMAX_NS, 0, close):
                value_end -= len(MINIMAX_NS)
            return value_end, m.end()
    return None


def _elements(text: str) -> tuple[list[tuple[str, str]], bool] | None:
    """``text`` as a run of complete elements, ``(tag, value)`` each, and
    whether every tag carried the ns_token; ``None`` when anything else is
    in it, so a value is only taken apart when all of it is markup."""
    out: list[tuple[str, str]] = []
    prefixed = True
    pos = 0
    while not _ELEMENTS_TAIL_RE.match(text, pos):
        m = _ELEMENT_RE.match(text, pos)
        if m is None:
            return None
        end = _element_end(text, m.group(2), m.end())
        if end is None:
            return None
        out.append((m.group(2), text[m.end() : end[0]]))
        prefixed = prefixed and m.group(1) is not None
        pos = end[1]
    return (out, prefixed) if out else None


def _schema_type(schema: Any) -> Any:
    """The declared type, the first non-null one of a type list."""
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def _decode_value(raw: str, schema: Any) -> Any:
    """One parameter value, nested elements included, typed by its schema.

    Elements are taken apart when the schema declares an object or an array,
    or, with no type declared, when they carry the ns_token as the trained
    format writes them; a string-typed value stays the text it arrived as.
    """
    ptype = _schema_type(schema)
    if ptype in _CONTAINER_TYPES and not raw.strip():
        return [] if ptype == "array" else {}  # `to_xml` renders these as nothing
    parsed = _elements(raw) if ptype is None or ptype in _CONTAINER_TYPES else None
    if parsed is not None and (ptype in _CONTAINER_TYPES or parsed[1]):
        elements = parsed[0]
        schema = schema if isinstance(schema, dict) else {}
        as_list = ptype == "array" or (
            ptype is None and all(tag == "item" for tag, _ in elements)
        )
        if as_list:
            return [_decode_value(v, schema.get("items")) for _, v in elements]
        props = schema.get("properties") or {}
        return {tag: _decode_value(v, props.get(tag)) for tag, v in elements}
    return coerce_json_or_raw(raw, ptype)


def _params(body: str) -> list[tuple[str, str]]:
    """The invoke body's parameters as ``(name, raw value)``.

    A closed parameter runs to its own closer, past any nested elements; an
    unclosed one ends where `_PARAM_RE` ends it.
    """
    out: list[tuple[str, str]] = []
    pos = 0
    while (pm := _PARAM_RE.search(body, pos)) is not None:
        end = _element_end(body, pm.group(1), pm.start(2))
        if end is None:
            out.append((pm.group(1), pm.group(2)))
            pos = pm.end()
        else:
            out.append((pm.group(1), body[pm.start(2) : end[0]]))
            pos = end[1]
    return out


def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`. The sweep that added that
    test to Qwen and GLM missed this format and DSML both, and "you emit
    `<invoke name="get_weather">` and then a `<city>Paris</city>` line" was
    still dispatched as a real call with the arguments filled in.

    The follower is a tag, because this format names parameters by the tag
    itself and so has no keyword to look for. Which tag is checked too when
    the request declared a schema for this tool -- otherwise `<br>` reads as
    the start of a call body -- but a tool declared without one still has to
    be callable, so an empty schema falls back to "any tag".
    """
    if not declared_tools_allow(name, param_types):
        return False
    # An undeclared schema falls back to "any tag", the same way an
    # empty one does below.
    types = param_types.get(name) or {}
    rest = body.lstrip()
    if not rest:
        return at_end
    tag = _FIRST_TAG_RE.match(rest)
    if tag is not None:
        return not types or tag.group(1) in types
    # A tag that has not finished arriving. At end of region that is all
    # there will ever be, so a prefix counts -- the rule `continues_a_call`
    # states and the other three formats have applied since it was written.
    # The followers are the declared tags, built from the request, because
    # this format names a parameter by the tag itself.
    #
    # An empty schema falls back to "any tag" here exactly as it does for a
    # complete tag four lines up. Gating on `bool(openers)` instead made a
    # zero-parameter tool the one tool whose truncated call could not be
    # recovered -- the same tool declared with one property recovered it --
    # and a truncation is precisely what this function exists to catch.
    followers = tuple(f"{MINIMAX_NS}<{t}>" for t in types) + tuple(
        f"<{t}>" for t in types
    ) or (f"{MINIMAX_NS}<", "<")
    return continues_a_call(rest, followers, arrived=not at_end)


class MiniMaxParser(ToolCallParser):
    NAME: ClassVar[str] = "minimax"
    # `<invoke name="` too: `_INVOKE_RE` matches it anywhere, so an invoke the
    # model wrote without the ns_token was a call when parsed whole and plain
    # text when streamed -- the read-ahead never opened a region for it. DSML
    # lists the same marker-less malform for the same reason.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        MINIMAX_NS,
        "<tool_call>",
        '<invoke name="',
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)
    CALL_FILLERS: ClassVar[tuple[str, ...]] = (MINIMAX_NS,)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ("</invoke>",)

    # The ns_token opens a region like the other two, rather than being
    # framing the reader drops. It prefixes every tag including ones inside a
    # call, so calling it framing would delete it from an answer that merely
    # *mentions* it while an answer that mentions the tag it prefixes keeps
    # both -- the same token surviving or not depending on what followed it.
    # `parse` had exactly that split (every occurrence deleted from a copy of
    # the text when a call was found, all of them left when one was not); a
    # region that parses to nothing is released whole, so this way both
    # answers keep their bytes.

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f"{MINIMAX_NS}<{k}>{v}{MINIMAX_NS}</{k}>" for k, v in args.items()
        )
        return (
            f'{MINIMAX_NS}<tool_call>{MINIMAX_NS}<invoke name="{name}">'
            f"{body}{MINIMAX_NS}</invoke>{MINIMAX_NS}</tool_call>"
        )

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        return MINIMAX_NS in text

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _INVOKE_RE.finditer(region):
            closed = m.group(1) is not None
            name = m.group(1) if closed else m.group(3)
            body = m.group(2) if closed else (m.group(4) or "")
            # Strip first, then judge. The other order -- `if not name` before
            # `name.strip()` -- let `name="   "` through the guard and out as
            # the empty string.
            name = (name or "").strip()
            if not usable_tool_name(name):
                continue
            if not closed and not _is_truncated_call(
                name, body, param_types, at_end=at_end
            ):
                continue
            types = param_types.get(name, {})
            schemas = getattr(param_types, "schemas", {}).get(name, {})
            args: dict[str, Any] = {}
            for k, raw in _params(body):
                args[k] = _decode_value(raw, schemas.get(k) or {"type": types.get(k)})
            spans.append(
                (cls.markup_begin(region, m.start()), cls.markup_end(region, m.end()))
            )
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        return RegionParse(tuple(tool_calls), tuple(spans))

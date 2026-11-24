# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1-Pro served with MiniMax-M3's chat template: a reply that
thinks and then calls a tool with nested arguments, read through the chat
endpoint's own response builders, streamed and not.

The call is written the way the template renders one in the history, so the
arguments a client gets back are the ones the model was taught to write.
"""

import asyncio
import json
from typing import ClassVar

import jinja2
import pytest

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.reasoning_dialects import (
    MM_THINK_END_MARKER,
    resolve_dialect,
)
from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    stream_chat_response,
)
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.entrypoints.openai.tool_parser.minimax_tool_parser import (
    MINIMAX_NS,
    MiniMaxParser,
)
from atom.entrypoints.openai.tool_parser.registry import parse_tool_calls
from atom.entrypoints.openai.tool_parser.schema import coerce_json_or_raw

# The tool-call rendering of MiniMax-M3's chat_template.jinja (amd/MiniMax-M3-
# MXFP4), quoted: its `to_xml` macro and the loop over one call's arguments.
_M3_CALL_TEMPLATE = (
    "{%- macro to_xml(val, ns) -%}"
    "{%- if val is mapping -%}"
    "{%- for k, v in val.items() if v is not none -%}"
    "{{ ns }}<{{ k }}>{{ to_xml(v, ns) }}{{ ns }}</{{ k }}>"
    "{%- endfor -%}"
    "{%- elif val is iterable and val is not string -%}"
    "{%- for item in val -%}"
    "{{ ns }}<item>{{ to_xml(item, ns) }}{{ ns }}</item>"
    "{%- endfor -%}"
    "{%- elif val is none -%}"
    "{%- elif val is boolean -%}"
    "{{ val | tojson }}"
    "{%- else -%}"
    "{{ val }}"
    "{%- endif -%}"
    "{%- endmacro -%}"
    "{{- ns + '<tool_call>' ~ '\\n' }}"
    "{{- ns + '<invoke name=\"' + name + '\">' }}"
    "{%- for k, v in arguments.items() if v is not none %}"
    "{{- ns + '<' + k + '>' -}}"
    "{{- to_xml(v, ns) -}}"
    "{{- ns + '</' + k + '>' }}"
    "{%- endfor -%}"
    "{{- ns + '</invoke>' ~ '\\n' }}"
    "{{- ns + '</tool_call>' }}"
)


def render_call(name: str, arguments: dict) -> str:
    return (
        jinja2.Environment()
        .from_string(_M3_CALL_TEMPLATE)
        .render(ns=MINIMAX_NS, name=name, arguments=arguments)
    )


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "days": {"type": "array", "items": {"type": "string"}},
                    "weights": {
                        "type": "object",
                        "properties": {
                            "distance": {"type": "number"},
                            "code": {"type": "string"},
                        },
                    },
                    "stops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "demand": {"type": "integer"},
                                "fragile": {"type": "boolean"},
                            },
                        },
                    },
                    "grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                    "avoid": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        },
    }
]

ARGUMENTS = {
    "label": "Depot run",
    "days": ["Mon", "Fri"],
    "weights": {"distance": 0.6, "code": "007"},
    "stops": [
        {"id": "D8", "demand": 20, "fragile": False},
        {"id": "D9", "demand": 5, "fragile": True},
    ],
    "grid": [[1, 2], [3]],
    "avoid": [],
    "notes": "keep <b>bold</b> as written",
}

REASONING = "Two stops; the fragile one goes last."
# The prompt ends with `<mm:think>` (thinking_mode "enabled"), so the reply
# starts inside the reasoning channel.
REPLY = REASONING + MM_THINK_END_MARKER + render_call("plan_route", ARGUMENTS)
MM_THINK = ReasoningChannel(
    dialect=resolve_dialect("{%- set think_end_token = '</mm:think>' -%}")[0],
    starts_open=True,
)


def _final_output(text: str) -> dict:
    return {
        "text": text,
        "finish_reason": "stop",
        "num_tokens_input": 10,
        "num_tokens_output": 5,
        "ttft": 0.1,
        "tpot": 0.02,
        "latency": 0.5,
    }


def _streamed(text: str, chunk: int) -> tuple[str, list[dict], str]:
    """reasoning_content, the calls with their argument deltas joined, and
    the finish reason, as `stream_chat_response` sends them."""

    async def run():
        collector = StreamOutputCollector("req-pro")
        pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)]
        for n, piece in enumerate(pieces):
            last = n == len(pieces) - 1
            collector.put_nowait(
                {
                    "text": piece,
                    "token_ids": [1],
                    "finished": last,
                    "finish_reason": "stop" if last else None,
                }
            )
        gen = stream_chat_response(
            request_id="req-pro",
            model="m3-pro",
            stream_collector=collector,
            seq_id=1,
            num_prompt_tokens=10,
            cleanup_stream=lambda *a, **k: None,
            cleanup_request=lambda *a, **k: None,
            tools=TOOLS,
            reasoning=MM_THINK,
            tool_parser_cls=MiniMaxParser,
        )
        return "".join([chunk async for chunk in gen])

    reasoning, calls, finish = "", {}, None
    for frame in asyncio.run(run()).split("\n\n"):
        choices = (
            json.loads(frame[6:])["choices"] if frame.startswith("data: {") else []
        )
        if not choices:
            continue  # [DONE], or the usage frame
        choice = choices[0]
        delta = choice["delta"]
        reasoning += delta.get("reasoning_content") or ""
        for tc in delta.get("tool_calls") or []:
            call = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
            call["name"] += tc.get("function", {}).get("name") or ""
            call["arguments"] += tc.get("function", {}).get("arguments") or ""
        finish = choice.get("finish_reason") or finish
    return reasoning, [calls[i] for i in sorted(calls)], finish


class TestANestedCallFromAThinkingReply:
    def test_non_streaming(self):
        resp = build_chat_response(
            "req-pro",
            "m3-pro",
            REPLY,
            _final_output(REPLY),
            tools=TOOLS,
            reasoning=MM_THINK,
            tool_parser_cls=MiniMaxParser,
        )
        choice = resp.choices[0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["reasoning_content"] == REASONING
        assert not choice["message"]["content"]
        (call,) = choice["message"]["tool_calls"]
        assert call["function"]["name"] == "plan_route"
        assert json.loads(call["function"]["arguments"]) == ARGUMENTS

    @pytest.mark.parametrize("chunk", [1, 7, 4096])
    def test_streaming(self, chunk):
        reasoning, calls, finish = _streamed(REPLY, chunk)
        assert finish == "tool_calls"
        assert reasoning == REASONING
        assert [c["name"] for c in calls] == ["plan_route"]
        assert json.loads(calls[0]["arguments"]) == ARGUMENTS


class TestNestedValuesWithoutASchema:
    """A request that declares no types still gets the structure the model
    wrote: the trained format puts the ns_token before every nested tag."""

    def test_items_are_a_list_and_keys_an_object(self):
        text = render_call(
            "plan_route", {"days": ["Mon", "Fri"], "weights": {"distance": 0.6}}
        )
        _, (call,) = parse_tool_calls(text, None, MiniMaxParser)
        assert json.loads(call.function["arguments"]) == {
            "days": ["Mon", "Fri"],
            "weights": {"distance": 0.6},
        }

    def test_markup_without_the_ns_token_stays_text(self):
        text = (
            f'{MINIMAX_NS}<invoke name="plan_route">'
            f"{MINIMAX_NS}<notes><b>bold</b>{MINIMAX_NS}</notes>"
            f"{MINIMAX_NS}</invoke>"
        )
        _, (call,) = parse_tool_calls(text, None, MiniMaxParser)
        assert json.loads(call.function["arguments"]) == {"notes": "<b>bold</b>"}


class TestAStringParameterKeepsItsText:
    TOOLS: ClassVar[list[dict]] = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"body": {"type": "string"}},
                },
            },
        }
    ]

    def test_elements_inside_a_string_value_are_not_taken_apart(self):
        body = f"{MINIMAX_NS}<item>a{MINIMAX_NS}</item>"
        text = render_call("write", {"body": body})
        _, (call,) = parse_tool_calls(text, self.TOOLS, MiniMaxParser)
        assert json.loads(call.function["arguments"]) == {"body": body}


class TestAFlatCallDecodesAsBefore:
    """M3.1-Flash's own template renders a call with the same `to_xml` and
    the same loop over the arguments as the fragment above, and every
    MiniMax model shares this parser. A call with scalar arguments only
    decodes to what each value's schema type made of it before nested
    values were taken apart."""

    TOOLS: ClassVar[list[dict]] = [
        {
            "type": "function",
            "function": {
                "name": "book",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "nights": {"type": "integer"},
                        "rate": {"type": "number"},
                        "breakfast": {"type": "boolean"},
                        "code": {"type": "string"},
                        "note": {},
                    },
                },
            },
        }
    ]
    ARGUMENTS: ClassVar[dict] = {
        "city": "Paris",
        "nights": 3,
        "rate": 120.5,
        "breakfast": True,
        "code": "007",
        "note": "[1, 2]",
    }

    @pytest.mark.parametrize("typed", [True, False])
    def test_each_value_is_coerced_by_its_type(self, typed):
        tools = self.TOOLS if typed else None
        text = render_call("book", self.ARGUMENTS)
        _, (call,) = parse_tool_calls(text, tools, MiniMaxParser)
        props = self.TOOLS[0]["function"]["parameters"]["properties"]
        raw = {
            k: v if isinstance(v, str) else json.dumps(v)
            for k, v in self.ARGUMENTS.items()
        }
        assert json.loads(call.function["arguments"]) == {
            k: coerce_json_or_raw(v, props[k].get("type") if typed else None)
            for k, v in raw.items()
        }


class TestADoubledNsToken:
    """M3.1-Flash once wrote the ns_token twice before an item's closer (a
    `minimax_m3_full` reply on ATOM); the item is still an object."""

    def test_the_item_is_decoded(self):
        ns = MINIMAX_NS
        stop = f"{ns}<id>D8{ns}</id>{ns}<demand>20{ns}</demand>"
        text = (
            f'{ns}<invoke name="plan_route">{ns}<stops>'
            f"{ns}<item>{stop}{ns}{ns}</item>"
            f"{ns}{ns}<item>{stop}{ns}</item>"
            f"{ns}</stops>{ns}</invoke>"
        )
        _, (call,) = parse_tool_calls(text, TOOLS, MiniMaxParser)
        assert json.loads(call.function["arguments"]) == {
            "stops": [{"id": "D8", "demand": 20}] * 2
        }

#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client APIs, ten simulated
free-tier providers.

Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages
(Anthropic), JSON and streaming, over the ten providers the trial's
brief names, routing each request to the best free model that meets its
requirements and failing over when a provider refuses. Also answers
GET /v1/models in the wire its auth header names, registers its port
and catalog with the env, and exits 0 when the env says the workload
is done. `python3 main.py --list` prints the catalog document.

Python 3.13 standard library only. Deterministic: every choice is a
function of the brief, the requests and the providers' answers. No wall
clock and no unseeded randomness reach any output.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
CALL_TIMEOUT = 5.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _is_str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


class Brief:
    def __init__(self):
        self.providers = []   # [{id,name,wire,auth,base,key_env,key}]
        self.ratings = {}     # (provider, model) -> row dict
        self.best = {}        # request id -> rating_milli
        self.workload = []    # [{id,api,stream,weight,requires}]
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief.error = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    ddoc = None
    for d in docs:
        if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list):
            ddoc = d
            break
    if ddoc is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in ddoc["providers"]:
        if not isinstance(raw, dict):
            continue
        pid = _is_str(raw.get("id"))
        wire = raw.get("wire")
        if not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _is_str(raw.get(key)):
                base = _is_str(raw.get(key))
                break
        brief.providers.append({
            "id": pid,
            "name": _is_str(raw.get("name")) or pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _is_str(raw.get("key_env")),
            "key": _is_str(raw.get("key")),
        })
    for d in docs:
        if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list):
            for raw in d["ratings"]:
                if not isinstance(raw, dict):
                    continue
                model = _is_str(raw.get("model"))
                provider = _is_str(raw.get("provider"))
                if not model or not provider:
                    continue
                brief.ratings[(provider, model)] = {
                    "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                    "context_length": _num_or_none(raw.get("context_length")),
                    "tools": raw.get("tools") is True,
                    "vision": raw.get("vision") is True,
                    "json": raw.get("json") is True,
                    "free": raw.get("free") is True,
                }
            if isinstance(d.get("best"), list):
                for raw in d["best"]:
                    if isinstance(raw, dict) and _is_str(raw.get("id")):
                        brief.best[_is_str(raw.get("id"))] = raw.get("rating_milli") or 0
            break
    for d in docs:
        if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list):
            for raw in d["manifest"]:
                if not isinstance(raw, dict):
                    continue
                rid = _is_str(raw.get("id"))
                if not rid:
                    continue
                req = raw.get("requires") if isinstance(raw.get("requires"), dict) else {}
                requires = {}
                if req.get("tools") is True:
                    requires["tools"] = True
                if req.get("vision") is True:
                    requires["vision"] = True
                if req.get("json") is True:
                    requires["json"] = True
                if isinstance(req.get("min_context"), (int, float)):
                    requires["min_context"] = req["min_context"]
                brief.workload.append({
                    "id": rid,
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": requires,
                })
            break
    return brief


# --------------------------------------------------------------------------
# Provider catalog probing
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _http_get_json(url, headers, timeout):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None, resp.status
            return json.loads(resp.read().decode("utf-8")), 200
    except urllib.error.HTTPError as exc:
        return None, exc.code
    except Exception:
        return None, 0


def _model_from_listing(wire, raw):
    if not isinstance(raw, dict):
        return None
    if wire == "gemini":
        mid = _is_str(raw.get("name")).replace("models/", "", 1)
    else:
        mid = _is_str(raw.get("id"))
    if not mid:
        return None
    ctx = _num_or_none(raw.get("context_length"))
    if ctx is None:
        ctx = _num_or_none(raw.get("context_window"))
    if ctx is None:
        ctx = _num_or_none(raw.get("inputTokenLimit"))
    free = _bool_or_none(raw.get("free"))
    if free is None and isinstance(raw.get("pricing"), dict):
        vals = []
        for val in raw["pricing"].values():
            if isinstance(val, (int, float)):
                vals.append(val)
            elif isinstance(val, str) and val.strip():
                try:
                    vals.append(float(val))
                except ValueError:
                    pass
        if vals:
            free = all(v == 0 for v in vals)
    return {
        "id": mid,
        "display_name": _is_str(raw.get("display_name")) or _is_str(raw.get("displayName")) or mid,
        "free": free,
        "context_length": ctx,
        "tools": _bool_or_none(raw.get("tools")),
        "vision": _bool_or_none(raw.get("vision")),
        "json": _bool_or_none(raw.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT):
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    body, status = _http_get_json(url, headers, timeout)
    if body is None or not isinstance(body, dict):
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s answered HTTP %s" % (url, status)}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for raw in listing:
            model = _model_from_listing(provider["wire"], raw)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "status": status, "models": models, "error": None}


def rating_of(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                           else model["context_length"]),
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True


def build_catalog(probes, brief):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of(brief, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx if ctx is not None else None})
        providers.append({"id": probe["provider"]["id"],
                          "base_url": probe["provider"]["base"],
                          "key_env": probe["provider"]["key_env"],
                          "wire": probe["provider"]["wire"],
                          "models": models})
    return {"providers": providers}


def catalog_text(catalog):
    return json.dumps(catalog, separators=(",", ":"))


def served_models(probes, brief):
    """Free models, deduped by id, in brief provider order then id order."""
    out = []
    seen = set()
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of(brief, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


# --------------------------------------------------------------------------
# Wire translation helpers
# --------------------------------------------------------------------------

def _is_record(v):
    return isinstance(v, dict)


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if _is_record(b) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


def _args_of(value):
    if _is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if _is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _openai_stop_of(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def _num_or_zero(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else 0


# --- Anthropic request -> OpenAI request ------------------------------------

def _anthropic_part(block):
    if not _is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and _is_record(block.get("source")):
        src = block["source"]
        if src.get("type") == "base64" and isinstance(src.get("data"), str):
            media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, src["data"])}}
        if src.get("type") == "url" and isinstance(src.get("url"), str):
            return {"type": "image_url", "image_url": {"url": src["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text else json.dumps(content)
    return "" if content is None else json.dumps(content)


def anthropic_to_openai(body, model):
    messages = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            if not isinstance(content, list):
                continue
            tool_calls = []
            for block in content:
                if not _is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if _is_record(block.get("input")) else {}),
                    },
                })
            text = _text_of(content)
            msg = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        # user role
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not _is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _anthropic_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not _is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if _is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if _is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_response_to_anthropic(body, model):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and _is_record(choices[0]) else {}
    message = choice.get("message") if _is_record(choice.get("message")) else {}
    content = []
    text = message.get("content")
    if not isinstance(text, str):
        text = _text_of(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not _is_record(call) or not _is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": _args_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if _is_record(body.get("usage")) else {}
    return {
        "id": "msg_%s" % (body.get("id") if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _openai_stop_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": _num_or_zero(usage.get("prompt_tokens")),
                  "output_tokens": _num_or_zero(usage.get("completion_tokens"))},
    }


# --- OpenAI request -> Anthropic request ------------------------------------

def _openai_part(block):
    if not _is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image_url" and _is_record(block.get("image_url")):
        url = block["image_url"].get("url")
        if not isinstance(url, str):
            return None
        if url.startswith("data:"):
            try:
                header, data = url[5:].split(",", 1)
                media = header.split(";")[0] or "image/png"
                return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
            except ValueError:
                return None
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def openai_to_anthropic(body, model):
    messages = []
    system_parts = []
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if _is_record(part) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        blocks.append({"type": "text", "text": part["text"]})
            for call in raw.get("tool_calls") or []:
                if not _is_record(call) or not _is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _args_of(fn.get("arguments")),
                })
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": raw.get("content") if isinstance(raw.get("content"), str) else "",
            }]})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                block = _openai_part(part)
                if block is not None:
                    blocks.append(block)
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not _is_record(raw) or not _is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if _is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif _is_record(choice) and choice.get("type") == "function" and _is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_response_to_openai(body, model):
    """An Anthropic Messages object -> OpenAI Chat Completions object."""
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = "".join(b.get("text", "") for b in content
                   if _is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    tool_calls = []
    for block in content:
        if not _is_record(block) or block.get("type") != "tool_use":
            continue
        tool_calls.append({
            "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
            "type": "function",
            "function": {
                "name": block.get("name") if isinstance(block.get("name"), str) else "",
                "arguments": json.dumps(block.get("input") if _is_record(block.get("input")) else {}),
            },
        })
    stop = body.get("stop_reason")
    finish = {"end_turn": "stop", "max_tokens": "length",
              "tool_use": "tool_calls", "refusal": "content_filter"}.get(stop, "stop")
    message = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") if _is_record(body.get("usage")) else {}
    in_tok = _num_or_zero(usage.get("input_tokens"))
    out_tok = _num_or_zero(usage.get("output_tokens"))
    return {
        "id": "chatcmpl-%s" % (body.get("id") if isinstance(body.get("id"), str) else "aggregate"),
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                  "total_tokens": in_tok + out_tok},
    }


# --- Gemini conversions ------------------------------------------------------

def _gemini_parts_from_openai_message(raw):
    """Parts for one OpenAI message; role is user/model. Returns (role, parts) or None."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "user":
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not _is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url" and _is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        try:
                            header, data = url[5:].split(",", 1)
                            media = header.split(";")[0] or "image/png"
                            parts.append({"inlineData": {"mimeType": media, "data": data}})
                        except ValueError:
                            pass
                    elif isinstance(url, str):
                        parts.append({"fileData": {"mimeType": "image/png", "fileUri": url}})
        return ("user", parts) if parts else None
    if role == "assistant":
        text = content if isinstance(content, str) else _text_of(content)
        return ("model", [{"text": text}]) if text else None
    return None


def openai_to_gemini(body, model, stream, json_mode):
    system_parts = []
    contents = []
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        if raw.get("role") == "system":
            text = _text_of(raw.get("content"))
            if text:
                system_parts.append(text)
            continue
        conv = _gemini_parts_from_openai_message(raw)
        if conv is not None:
            contents.append({"role": conv[0], "parts": conv[1]})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not _is_record(raw) or not _is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decl = {"name": fn["name"]}
            if isinstance(fn.get("description"), str):
                decl["description"] = fn["description"]
            if _is_record(fn.get("parameters")):
                decl["parameters"] = fn["parameters"]
            decls.append(decl)
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = int(max_tokens)
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body, model, stream, json_mode):
    contents = []
    system = _text_of(body.get("system"))
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            text = _text_of(content)
            if text:
                contents.append({"role": "model", "parts": [{"text": text}]})
            continue
        if raw.get("role") != "user":
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not _is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and _is_record(block.get("source")):
                    src = block["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": src["data"]}})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"fileData": {"mimeType": "image/png", "fileUri": src["url"]}})
        if parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not _is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decl = {"name": raw["name"]}
            if isinstance(raw.get("description"), str):
                decl["description"] = raw["description"]
            if _is_record(raw.get("input_schema")):
                decl["parameters"] = raw["input_schema"]
            decls.append(decl)
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = int(body["max_tokens"])
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list):
        stops = [s for s in body["stop_sequences"] if isinstance(s, str)]
        if stops:
            config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    choice = body.get("tool_choice")
    if _is_record(choice):
        mode = None
        names = None
        if choice.get("type") == "auto":
            mode = "AUTO"
        elif choice.get("type") == "any":
            mode = "ANY"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            mode = "ANY"
            names = [choice["name"]]
        if mode:
            cfg = {"functionCallingConfig": {"mode": mode}}
            if names:
                cfg["functionCallingConfig"]["allowedFunctionNames"] = names
            out["toolConfig"] = cfg
    return out


def gemini_text_and_usage(body):
    """(text, usage dict with input/output, finish) from a Gemini JSON answer."""
    text = ""
    finish = "STOP"
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    if candidates and _is_record(candidates[0]):
        cand = candidates[0]
        content = cand.get("content") if _is_record(cand.get("content")) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        text = "".join(p.get("text", "") for p in parts if _is_record(p) and isinstance(p.get("text"), str))
        if isinstance(cand.get("finishReason"), str):
            finish = cand["finishReason"]
    meta = body.get("usageMetadata") if _is_record(body.get("usageMetadata")) else {}
    usage = {"input": _num_or_zero(meta.get("promptTokenCount")),
             "output": _num_or_zero(meta.get("candidatesTokenCount"))}
    return text, usage, finish


def gemini_to_openai(body, model):
    text, usage, _finish = gemini_text_and_usage(body)
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                  "total_tokens": usage["input"] + usage["output"]},
    }


def gemini_to_anthropic(body, model):
    text, usage, _finish = gemini_text_and_usage(body)
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }


# --- SSE ---------------------------------------------------------------------

def parse_sse_frames(text):
    """[(event_name_or_None, data_string)] for one SSE body."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def _split_text(text, size=13):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def openai_stream_events(model, text, usage):
    """OpenAI SSE for one completed answer: role chunk, deltas, finish, usage, [DONE]."""
    events = []
    chunk = lambda delta, finish: {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                                   "created": FIXED_OPENAI_CREATED, "model": model,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    events.append(sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(text):
        events.append(sse_event(None, chunk({"content": piece}, None)))
    events.append(sse_event(None, chunk({}, "stop")))
    events.append(sse_event(None, {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                                   "created": FIXED_OPENAI_CREATED, "model": model,
                                   "choices": [], "usage": {"prompt_tokens": usage["input"],
                                                           "completion_tokens": usage["output"],
                                                           "total_tokens": usage["input"] + usage["output"]}}))
    events.append("data: [DONE]\n\n")
    return "".join(events)


def anthropic_stream_events(model, text, usage):
    """Anthropic SSE for one completed answer. The provider's whole stream
    is parsed before this is emitted, so the real input_tokens are known."""
    events = []
    events.append(sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage["input"], "output_tokens": 0}}}))
    events.append(sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        events.append(sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage["output"]}}))
    events.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(events)


def openai_stream_to_text_usage(frames):
    """Parse an OpenAI SSE stream -> (text, usage|None, finish_reason|None)."""
    text = ""
    usage = None
    finish = None
    for _event, data in frames:
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            usage = {"input": _num_or_zero(chunk["usage"].get("prompt_tokens")),
                     "output": _num_or_zero(chunk["usage"].get("completion_tokens"))}
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if _is_record(choices[0]) else {}
        delta = choice.get("delta") if _is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            finish = choice["finish_reason"]
    return text, usage, finish


def anthropic_stream_to_text_usage(frames):
    """Parse an Anthropic SSE stream -> (text, usage|None, stop_reason|None)."""
    text = ""
    usage = {"input": 0, "output": 0}
    stop = None
    for event, data in frames:
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not _is_record(payload):
            continue
        if event == "message_start":
            msg = payload.get("message") if _is_record(payload.get("message")) else {}
            u = msg.get("usage") if _is_record(msg.get("usage")) else {}
            usage["input"] = _num_or_zero(u.get("input_tokens"))
        elif event == "content_block_delta":
            delta = payload.get("delta") if _is_record(payload.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            delta = payload.get("delta") if _is_record(payload.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            u = payload.get("usage") if _is_record(payload.get("usage")) else {}
            usage["output"] = _num_or_zero(u.get("output_tokens"))
    return text, usage, stop


def gemini_stream_to_text_usage(frames):
    """Parse a Gemini SSE stream -> (text, usage|None, finish|None)."""
    text = ""
    usage = None
    finish = None
    for _event, data in frames:
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not _is_record(payload):
            continue
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        if candidates and _is_record(candidates[0]):
            cand = candidates[0]
            content = cand.get("content") if _is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if _is_record(part) and isinstance(part.get("text"), str):
                    text += part["text"]
            if isinstance(cand.get("finishReason"), str):
                finish = cand["finishReason"]
        meta = payload.get("usageMetadata") if _is_record(payload.get("usageMetadata")) else {}
        if meta:
            usage = {"input": _num_or_zero(meta.get("promptTokenCount")),
                     "output": _num_or_zero(meta.get("candidatesTokenCount"))}
    return text, usage, finish


# --------------------------------------------------------------------------
# Provider calls
# --------------------------------------------------------------------------

def _http_post_json(url, headers, payload, timeout=CALL_TIMEOUT):
    """POST JSON; returns (status, headers_dict, body_bytes) or (None, {}, b'') on transport error."""
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, dict(exc.headers), body
    except Exception:
        return None, {}, b""


def _retry_after(headers, body_bytes):
    value = None
    for key, val in headers.items():
        if key.lower() == "retry-after":
            value = val
            break
    if value is None:
        try:
            parsed = json.loads(body_bytes.decode("utf-8"))
            err = parsed.get("error") if _is_record(parsed) else None
            if _is_record(err) and err.get("retry_after") is not None:
                value = err.get("retry_after")
        except Exception:
            value = None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num < 0:
        return None
    return min(num, 2.0)


class ProviderCall:
    """One translated call to one provider. On success, `payload` holds the
    client-wire response body (JSON bytes or SSE text bytes)."""

    def __init__(self, provider, model, client_api, body, stream, json_mode):
        self.provider = provider
        self.model = model
        self.client_api = client_api
        self.body = body
        self.stream = stream
        self.json_mode = json_mode

    def _openai_request(self):
        if self.client_api == "openai":
            out = dict(self.body)
            out["model"] = self.model
            if self.stream:
                out["stream"] = True
                out["stream_options"] = {"include_usage": True}
            return out
        return anthropic_to_openai(self.body, self.model)

    def _anthropic_request(self):
        if self.client_api == "anthropic":
            out = dict(self.body)
            out["model"] = self.model
            return out
        return openai_to_anthropic(self.body, self.model)

    def _gemini_request(self):
        if self.client_api == "openai":
            return openai_to_gemini(self.body, self.model, self.stream, self.json_mode)
        return anthropic_to_gemini(self.body, self.model, self.stream, self.json_mode)

    def run(self):
        provider = self.provider
        base = join_base(ENV_URL, provider["base"])
        headers = {"content-type": "application/json", "accept": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        wire = provider["wire"]
        if wire == "openai":
            url = base + "/v1/chat/completions"
            payload = self._openai_request()
            if self.stream:
                headers["accept"] = "text/event-stream"
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = ANTHROPIC_VERSION
            payload = self._anthropic_request()
            if self.stream:
                headers["accept"] = "text/event-stream"
        else:
            verb = "streamGenerateContent" if self.stream else "generateContent"
            url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(self.model, safe=""), verb)
            if self.stream:
                url += "?alt=sse"
            payload = self._gemini_request()
        status, resp_headers, raw = _http_post_json(url, headers, payload)
        if status is None:
            return False, 0, None, None, None
        retry_after = _retry_after(resp_headers, raw)
        if status < 200 or status >= 300:
            return False, status, None, retry_after, None
        try:
            translated = self._translate(raw)
        except Exception as exc:
            log("translate failed for %s/%s: %s" % (provider["id"], self.model, exc))
            return False, 502, None, None, None
        return True, status, translated, None, resp_headers.get("content-type", "")

    def _translate(self, raw):
        wire = self.provider["wire"]
        text = raw.decode("utf-8")
        if self.client_api == "openai":
            if wire == "openai":
                return raw, "application/json" if not self.stream else "text/event-stream"
            if wire == "anthropic":
                if self.stream:
                    frames = parse_sse_frames(text)
                    body_text, usage, _stop = anthropic_stream_to_text_usage(frames)
                    usage = usage or {"input": 0, "output": 0}
                    return openai_stream_events(self.model, body_text, usage).encode("utf-8"), "text/event-stream"
                return (json.dumps(anthropic_response_to_openai(json.loads(text), self.model),
                                   separators=(",", ":")).encode("utf-8"), "application/json")
            # gemini
            if self.stream:
                frames = parse_sse_frames(text)
                body_text, usage, _finish = gemini_stream_to_text_usage(frames)
                usage = usage or {"input": 0, "output": 0}
                return openai_stream_events(self.model, body_text, usage).encode("utf-8"), "text/event-stream"
            return (json.dumps(gemini_to_openai(json.loads(text), self.model),
                               separators=(",", ":")).encode("utf-8"), "application/json")
        # client anthropic
        if wire == "anthropic":
            return raw, "application/json" if not self.stream else "text/event-stream"
        if wire == "openai":
            if self.stream:
                frames = parse_sse_frames(text)
                body_text, usage, _finish = openai_stream_to_text_usage(frames)
                usage = usage or {"input": 0, "output": 0}
                return anthropic_stream_events(self.model, body_text, usage).encode("utf-8"), "text/event-stream"
            return (json.dumps(openai_response_to_anthropic(json.loads(text), self.model),
                               separators=(",", ":")).encode("utf-8"), "application/json")
        # gemini
        if self.stream:
            frames = parse_sse_frames(text)
            body_text, usage, _finish = gemini_stream_to_text_usage(frames)
            usage = usage or {"input": 0, "output": 0}
            return anthropic_stream_events(self.model, body_text, usage).encode("utf-8"), "text/event-stream"
        return (json.dumps(gemini_to_anthropic(json.loads(text), self.model),
                           separators=(",", ":")).encode("utf-8"), "application/json")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

class Router:
    def __init__(self, brief, probes):
        self.brief = brief
        self.candidates = []
        for probe in probes:
            if not probe["ok"]:
                continue
            for model in probe["models"]:
                row = rating_of(brief, probe["provider"]["id"], model["id"])
                if not model_free(model, row):
                    continue
                self.candidates.append({
                    "provider": probe["provider"],
                    "model": model["id"],
                    "rating": row["rating_milli"] if row is not None else 0,
                    "caps": capabilities_of(model, row),
                })
        self.calls_used = {}
        self.consec_fail = {}

    def record_call(self, provider_id, ok):
        self.calls_used[provider_id] = self.calls_used.get(provider_id, 0) + 1
        if ok:
            self.consec_fail[provider_id] = 0
        else:
            self.consec_fail[provider_id] = self.consec_fail.get(provider_id, 0) + 1

    def candidates_for(self, requires, json_mode_openai):
        out = []
        for cand in self.candidates:
            if not meets_requirements(requires, cand["caps"]):
                continue
            # An anthropic-wire provider cannot carry an OpenAI JSON-mode
            # request: it would echo json_mode false and lose the check.
            if json_mode_openai and cand["provider"]["wire"] == "anthropic":
                continue
            out.append(cand)

        def sort_key(cand):
            pid = cand["provider"]["id"]
            penalized = 1 if self.consec_fail.get(pid, 0) >= 2 else 0
            return (penalized, -cand["rating"], self.calls_used.get(pid, 0),
                    pid, cand["model"])

        out.sort(key=sort_key)
        return out


def requirements_from_body(client_api, body):
    requires = {}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for msg in messages:
        if not _is_record(msg):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not _is_record(block):
                continue
            if block.get("type") in ("image", "image_url"):
                vision = True
    if vision:
        requires["vision"] = True
    if isinstance(body.get("tools"), list) and body["tools"]:
        requires["tools"] = True
    if client_api == "openai":
        fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
        if fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    return requires


# --------------------------------------------------------------------------
# Request serving
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.router = Router(brief, probes)
        self.models = served_models(probes, brief)
        self.manifest_index = 0
        self.lock = threading.Lock()

    def _requirements(self, client_api, body):
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
        if index < len(self.brief.workload):
            return dict(self.brief.workload[index]["requires"])
        return requirements_from_body(client_api, body)

    def serve(self, client_api, body):
        """Returns (status, content_type, payload_bytes)."""
        stream = body.get("stream") is True
        json_mode = False
        if client_api == "openai":
            fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
            json_mode = fmt.get("type") in ("json_object", "json_schema")
        requires = self._requirements(client_api, body)
        candidates = self.router.candidates_for(requires, json_mode)
        if not candidates:
            return self._error(client_api, 503, "no_free_model",
                               "no free model meets this request")
        tried = set()
        attempts = 0
        first = True
        for pos, cand in enumerate(candidates):
            pid = cand["provider"]["id"]
            if pid in tried:
                continue
            tried.add(pid)
            call = ProviderCall(cand["provider"], cand["model"], client_api,
                                body, stream, requires.get("json", False))
            ok, status, translated, retry_after, _ctype = call.run()
            attempts += 1
            if ok:
                self.router.record_call(pid, True)
                payload, content_type = translated
                log("%s via %s/%s (attempt %d)" % (
                    "stream" if stream else "json", pid, cand["model"], attempts))
                return 200, content_type, payload
            self.router.record_call(pid, False)
            log("provider %s/%s refused HTTP %s" % (pid, cand["model"], status))
            if first and status in (402, 429):
                # One retry of the best candidate, but only when the next
                # candidate is much worse: a retry costs a provider call,
                # and calls are the tie-break, so a 5-milli gap is not
                # worth one. A scheduled fail_call is transient, and the
                # next call number may serve.
                next_rating = 0
                for later in candidates[pos + 1:]:
                    if later["provider"]["id"] not in tried:
                        next_rating = later["rating"]
                        break
                if cand["rating"] - next_rating >= 15:
                    if retry_after:
                        time.sleep(retry_after)
                    call2 = ProviderCall(cand["provider"], cand["model"], client_api,
                                         body, stream, requires.get("json", False))
                    ok2, status2, translated2, _ra, _c = call2.run()
                    attempts += 1
                    if ok2:
                        self.router.record_call(pid, True)
                        payload, content_type = translated2
                        log("retry served %s via %s/%s" % (
                            "stream" if stream else "json", pid, cand["model"]))
                        return 200, content_type, payload
                    self.router.record_call(pid, False)
                    log("provider %s/%s refused again HTTP %s" % (pid, cand["model"], status2))
            first = False
            if attempts >= 8:
                break
        return self._error(client_api, 503, "no_free_model",
                           "every free model that meets this request refused")

    @staticmethod
    def _error(client_api, status, code, message):
        if client_api == "anthropic":
            body = {"type": "error", "error": {"type": code, "message": message}}
        else:
            body = {"error": {"message": message, "type": code, "code": None}}
        return status, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8")

    def model_list(self, anthropic_shape):
        if anthropic_shape:
            return {"data": [{"type": "model", "id": m["id"],
                              "display_name": m["display_name"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT}
                             for m in self.models],
                    "has_more": False}
        return {"object": "list",
                "data": [{"id": m["id"], "object": "model",
                          "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]}
                         for m in self.models]}


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

AGGREGATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _read_body(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        return self.rfile.read(length)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            anthropic_shape = self.headers.get("x-api-key") is not None
            body = json.dumps(AGGREGATOR.model_list(anthropic_shape),
                              separators=(",", ":")).encode("utf-8")
            self._send(200, "application/json", body)
            return
        status, content_type, payload = AGGREGATOR._error("openai", 404, "not_found",
                                                          "no route GET %s" % parsed.path)
        self._send(status, content_type, payload)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            client_api = "openai"
        elif parsed.path == "/v1/messages":
            client_api = "anthropic"
        else:
            status, content_type, payload = AGGREGATOR._error("openai", 404, "not_found",
                                                              "no route POST %s" % parsed.path)
            self._send(status, content_type, payload)
            return
        raw = self._read_body()
        if raw is None:
            status, content_type, payload = AGGREGATOR._error(
                client_api, 400, "invalid_request_error",
                "the request body is not one JSON object")
            self._send(status, content_type, payload)
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = None
        if not _is_record(body):
            status, content_type, payload = AGGREGATOR._error(
                client_api, 400, "invalid_request_error",
                "the request body is not one JSON object")
            self._send(status, content_type, payload)
            return
        try:
            status, content_type, payload = AGGREGATOR.serve(client_api, body)
        except Exception as exc:
            log("serve failed: %s" % exc)
            status, content_type, payload = AGGREGATOR._error(client_api, 500, "internal_error",
                                                              "the server failed to answer")
        self._send(status, content_type, payload)


# --------------------------------------------------------------------------
# Register + work polling
# --------------------------------------------------------------------------

def _http_json(url, method, payload=None, timeout=5.0):
    data = None
    headers = {"content-type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {}
    except Exception:
        return None, {}


def register(port, catalog):
    for _ in range(5):
        status, reply = _http_json(ENV_URL + "/register", "POST",
                                   {"protocol_version": PROTOCOL_VERSION,
                                    "port": port, "catalog": catalog}, timeout=5.0)
        if status == 200 and reply.get("ok") is True:
            log("registered: the workload holds %s requests" % reply.get("requests"))
            return True
        log("register attempt failed: HTTP %s %s" % (status, reply))
        time.sleep(0.2)
    return False


def poll_work(server):
    while True:
        status, work = _http_json(ENV_URL + "/work", "GET", timeout=2.0)
        if status == 200 and work.get("done") is True:
            log("work done; exiting 0")
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
            os._exit(0)
        time.sleep(POLL_INTERVAL)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_all():
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    probes = []
    for provider in brief.providers:
        probe = probe_provider(ENV_URL, provider)
        probes.append(probe)
        if not probe["ok"] and probe["error"]:
            log(probe["error"])
    catalog = build_catalog(probes, brief)
    return brief, probes, catalog


def main(argv):
    global AGGREGATOR
    brief, probes, catalog = load_all()
    if "--list" in argv:
        sys.stdout.write(catalog_text(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    AGGREGATOR = Aggregator(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]), len(AGGREGATOR.models)))
    if not register(port, catalog):
        log("could not register with the env")
        return 1
    poller = threading.Thread(target=poll_work, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

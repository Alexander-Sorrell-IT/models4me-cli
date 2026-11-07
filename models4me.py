#!/usr/bin/env python3
"""Models4me — every model opencode can reach, plus NVIDIA direct, in one CLI.

    models4me                          # list models (opencode catalog + NVIDIA)
    models4me list -p nvidia           # one provider only
    models4me list -f coder --live     # filter, or only models verified live
    models4me providers                # endpoints, key source, model counts
    models4me probe -p nvidia          # test a provider's models with a real call
    models4me ask nvidia/openai/gpt-oss-20b "say pong"
    models4me ask groq/... "hi" --via opencode
    models4me use nvidia/mistralai/mistral-large-4-1t-a52b
    models4me current
    models4me extraFuff -v            # Freebuff's models, levels and per-model criteria

Where the data comes from:
  * models        — `opencode models` (cached) unioned with a LIVE GET /models on every
                    provider a stored key can reach, so a provider's own catalog is
                    listed even when opencode doesn't carry it
  * endpoints     — models.dev (cached) + ~/.config/omnicli/config.json + built-in table
  * credentials   — provider env vars, then omnicli's own keys (config.json `keys` and
                    its keys.env / trio store), then `opencode auth export`. Keys are
                    only read, never written back anywhere
  * liveness      — `models4me probe`, cached in ~/.cache/models4me/model_status.json

Every command accepts --json. Cache dir: MODELS4ME_CACHE, default model config:
MODELS4ME_CONFIG (~/.config/models4me/config.json).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html as html_lib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import namedtuple

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
MODELS_DEV_URL = "https://models.dev/api.json"

HOME = os.path.expanduser("~")
CACHE_DIR = os.environ.get("MODELS4ME_CACHE") or os.path.join(HOME, ".cache", "models4me")
STATUS_PATH = os.path.join(CACHE_DIR, "model_status.json")
PROVIDERS_CACHE = os.path.join(CACHE_DIR, "providers.json")
OPENCODE_CACHE = os.path.join(CACHE_DIR, "opencode_models.json")
CONFIG_PATH = os.environ.get("MODELS4ME_CONFIG") or os.path.join(
    HOME, ".config", "models4me", "config.json"
)
OMNICLI_CONFIG = os.path.join(HOME, ".config", "omnicli", "config.json")
DIRECT_CACHE = os.path.join(CACHE_DIR, "direct_models.json")
DISCOVER_TTL = 24 * 3600

FREEBUFF_README = "https://raw.githubusercontent.com/CodebuffAI/freebuff/main/README.md"
FREEBUFF_PLANS = "https://freebuff.com/plans"
FREEBUFF_CACHE = os.path.join(CACHE_DIR, "freebuff.json")
FREEBUFF_TTL = 24 * 3600

console = Console()
err = Console(stderr=True)

# id = provider/model as opencode prints it; model = the ID the provider's API wants.
Model = namedtuple("Model", "id provider model source category chat tools")

# Used when models.dev is unreachable. Endpoints and env keys for the providers
# opencode exposes, plus NVIDIA direct.
PROVIDER_DEFAULTS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", ["NVIDIA_API_KEY"]),
    "groq": ("https://api.groq.com/openai/v1", ["GROQ_API_KEY"]),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai",
               ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "aihubmix": ("https://aihubmix.com/v1", ["AIHUBMIX_API_KEY"]),
    "edenai": ("https://api.edenai.run/v3", ["EDENAI_API_KEY"]),
    "novita-ai": ("https://api.novita.ai/openai", ["NOVITA_API_KEY"]),
    "chutes": ("https://llm.chutes.ai/v1", ["CHUTES_API_KEY"]),
    "zai": ("https://api.z.ai/api/paas/v4", ["ZHIPU_API_KEY"]),
    "alibaba": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                ["DASHSCOPE_API_KEY"]),
    "orcarouter": ("https://api.orcarouter.ai/v1", ["ORCAROUTER_API_KEY"]),
    "kenari": ("https://kenari.id/v1", ["KENARI_API_KEY"]),
    "tokengo": ("https://api.tokengo.com/v1", ["TOKENGO_API_KEY"]),
    "inferx": ("https://model.inferx.net/endpoints/v1", ["INFERX_API_KEY"]),
    "unorouter": ("https://api.unorouter.com/v1", ["UNOROUTER_API_KEY"]),
    "aki-io": ("https://aki.io/v1", ["AKI_IO_API_KEY"]),
    "agnes": ("https://apihub.agnes-ai.com/v1", ["AGNES_API_KEY"]),
    "huggingface": ("https://router.huggingface.co/v1", ["HF_TOKEN"]),
    "opencode": ("https://opencode.ai/inference/openai/v1", ["OPENCODE_API_KEY"]),
    "ollama": ("http://localhost:11434/v1", []),
    "openrouter": ("https://openrouter.ai/api/v1", ["OPENROUTER_API_KEY"]),
}

# Model-ID -> category / capability rules. First match wins.
CATEGORY_RULES = [
    ("Embedding", ("embed", "nvclip", "nemoretriever", "arctic-embed"), False, False),
    ("Reward Model", ("-reward", "reward-"), False, False),
    ("Biology & Healthcare", ("protein", "gene", "molecule", "boltz", "rfdiffusion",
                              "genmol", "openfold", "dock", "evo2", "proteina"), False, False),
    ("Physical AI & Simulation", ("cuopt", "weather", "fourcastnet", "pose",
                                  "active-speaker", "bevformer", "relighting"), False, False),
    ("Guard / Safety", ("guard", "content-safety", "topic-control", "safety",
                        "video-detector"), True, False),
    ("Translation", ("translate", "riva"), True, False),
    ("Document AI & OCR", ("parse", "ocr", "paddleocr", "element", "table"), True, False),
    ("Image Generation", ("flux", "qwen-image", "diffusiongemma", "image-edit",
                          "stable-diffusion", "sd3", "dall-e"), True, False),
    ("Coding", ("coder", "codegemma", "codellama", "starcoder", "codestral",
                "code-instruct", "laguna", "claude-code", "gpt-5-codex"), True, True),
    ("Reasoning", ("reasoning", "cosmos-reason", "deepseek-v4", "step-5",
                   "o1", "o3", "o4", "r1"), True, True),
    ("Vision / Multimodal", ("vision", "-vl", "vila", "neva", "fuyu", "kosmos",
                             "deplot", "omni", "gemini", "claude", "gpt-4o"), True, True),
]

CATEGORY_ORDER = [
    "Coding", "Reasoning", "Vision / Multimodal", "General Chat",
    "Document AI & OCR", "Image Generation", "Translation",
    "Embedding", "Guard / Safety", "Reward Model",
    "Biology & Healthcare", "Physical AI & Simulation",
]
ORDER_INDEX = {label: i for i, label in enumerate(CATEGORY_ORDER)}

# Outcomes that say "the API was busy/slow", not "the model is gone". Caching
# them would permanently paint a working model as dead.
RETRYABLE = {"429", "500", "502", "503", "504", "timeout"}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# --- small file cache helpers ------------------------------------------------

def read_cache(path, ttl):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if ttl and time.time() - data.get("fetched_at", 0) > ttl:
            return None
        return data.get("payload")
    except (OSError, ValueError):
        return None


def write_cache(path, payload):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "payload": payload}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def load_status():
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"probed_at": None, "models": {}}


def save_status(status):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    os.replace(tmp, STATUS_PATH)


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# --- providers + credentials -------------------------------------------------

def read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def omnicli_config():
    """The user's omnicli config.json (providers + keys), or {} if absent."""
    return read_json_file(OMNICLI_CONFIG) or {}


def omnicli_provider_overrides():
    """base_url/env_key table from the user's omnicli install, if present."""
    out = {}
    for pid, entry in (omnicli_config().get("providers") or {}).items():
        if not isinstance(entry, dict):
            continue
        api = entry.get("base_url")
        env = entry.get("env_key")
        out[pid] = {
            "api": api if isinstance(api, str) and api.startswith("http") else None,
            "env": [env] if isinstance(env, str) and env else [],
        }
    return out


def omnicli_env_files():
    """Env files omnicli itself reads, in its own precedence order."""
    paths = []
    explicit = os.environ.get("OMNICLI_KEYS_FILE", "").strip()
    if explicit:
        paths.append(os.path.expanduser(explicit))
    paths.append(os.path.join(os.path.dirname(OMNICLI_CONFIG), "keys.env"))
    trio = os.environ.get("TRIO_KEYS_FILE", "").strip()
    paths.append(os.path.expanduser(trio) if trio
                 else os.path.join(HOME, ".config", "trio", "keys.env"))
    seen, out = set(), []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def omnicli_credentials(ptable):
    """Keys omnicli holds: config.json `keys` first, then its env files.

    Returns {provider: (key, 'omnicli')}. Read-only: nothing is written back,
    so omnicli stays the owner of its own secrets.
    """
    cfg = omnicli_config()
    creds = {}
    keys = cfg.get("keys") or {}
    if isinstance(keys, dict):
        for pid, key in keys.items():
            if isinstance(key, str) and key.strip() and isinstance(pid, str):
                creds[pid.strip().lower()] = (key.strip(), "omnicli")

    # env-var -> provider index, so keys.env entries land on the right provider.
    var_to_pid = {}
    for pid, entry in (cfg.get("providers") or {}).items():
        if not isinstance(entry, dict):
            continue
        names = [entry.get("env_key")] + list(entry.get("alt_env_keys") or [])
        for var in names:
            if isinstance(var, str) and var:
                var_to_pid.setdefault(var, str(pid).lower())
    for pid, entry in ptable.items():
        for var in (entry or {}).get("env") or []:
            var_to_pid.setdefault(var, pid)

    file_values = {}
    for path in omnicli_env_files():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    var, value = line.split("=", 1)
                    var, value = var.strip(), value.strip().strip("\"'")
                    if var and value:
                        file_values.setdefault(var, value)
        except OSError:
            continue
    for var, value in file_values.items():
        pid = var_to_pid.get(var)
        if pid:
            creds.setdefault(pid, (value, "omnicli"))
    return creds


def provider_table(refresh=False):
    """pid -> {'api': url|None, 'env': [vars], 'models': name|None}."""
    table = {pid: {"api": api, "env": list(envs)} for pid, (api, envs) in PROVIDER_DEFAULTS.items()}

    dev = read_cache(PROVIDERS_CACHE, 0 if refresh else 7 * 86400)
    if dev is None:
        try:
            res = httpx.get(MODELS_DEV_URL, timeout=30.0)
            res.raise_for_status()
            dev = res.json()
            write_cache(PROVIDERS_CACHE, dev)
        except Exception as e:  # noqa: BLE001 - offline is a supported state
            err.print(f"[yellow]models.dev unavailable ({type(e).__name__}); "
                      f"using the built-in provider table.[/yellow]")
            dev = {}

    for pid, pdata in dev.items():
        if not isinstance(pdata, dict):
            continue
        entry = table.setdefault(pid, {"api": None, "env": []})
        api = pdata.get("api")
        # Templated endpoints (${HOST}) cannot be called directly as-is.
        templated = isinstance(api, str) and "{" in api
        if isinstance(api, str) and api.startswith("http") and not templated:
            entry["api"] = api
        envs = [e for e in (pdata.get("env") or []) if isinstance(e, str)]
        entry["env"] = list(dict.fromkeys(entry["env"] + envs))

    for pid, override in omnicli_provider_overrides().items():
        entry = table.setdefault(pid, {"api": None, "env": []})
        if not entry.get("api") and override.get("api"):
            entry["api"] = override["api"]
        entry["env"] = list(dict.fromkeys(entry["env"] + override["env"]))
    return table


def opencode_credentials():
    """Keys opencode already stores, keyed by integration id. Never persisted."""
    exe = shutil.which("opencode")
    if not exe:
        return {}
    try:
        res = subprocess.run([exe, "auth", "export"], capture_output=True,
                             text=True, timeout=60)
        entries = json.loads(res.stdout)
    except Exception:  # noqa: BLE001 - no opencode, or an unexpected format
        return {}
    creds = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("active", True):
            continue
        value = entry.get("value") or {}
        key = value.get("key")
        pid = entry.get("integrationID")
        if isinstance(key, str) and key and isinstance(pid, str):
            creds.setdefault(pid, (key, "opencode"))
    return creds


def credentials(ptable):
    """All stored keys, omnicli first: {provider: (key, source)}."""
    creds = omnicli_credentials(ptable)
    for pid, hit in opencode_credentials().items():
        creds.setdefault(pid, hit)
    return creds


def resolve_key(provider, ptable, creds):
    """(key, source) where source is 'env:VAR', 'omnicli', 'opencode', or None.

    Precedence: the live process environment, then the keys omnicli stores,
    then whatever opencode saved. The user's own shell always wins.
    """
    for var in (ptable.get(provider) or {}).get("env", []):
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    hit = creds.get(provider)
    if hit:
        return hit
    return None, None


def key_source(provider, ptable, creds):
    return resolve_key(provider, ptable, creds)[1]


# --- catalogs ----------------------------------------------------------------

def classify_model(model_id):
    low = model_id.lower()
    for label, needles, chat_ok, tools_ok in CATEGORY_RULES:
        if any(n in low for n in needles):
            return label, chat_ok, tools_ok
    return "General Chat", True, True


def opencode_model_ids(refresh=False):
    """`opencode models`, cached. Falls back to any stale cache when offline."""
    cached = read_cache(OPENCODE_CACHE, 0 if refresh else 6 * 3600)
    if cached:
        return cached

    exe = shutil.which("opencode")
    ids = []
    if exe:
        try:
            res = subprocess.run([exe, "models"], capture_output=True, text=True, timeout=180)
            text = ANSI.sub("", res.stdout)
            ids = sorted({line.strip() for line in text.splitlines()
                          if "/" in line and " " not in line.strip()})
        except Exception as e:  # noqa: BLE001
            err.print(f"[yellow]`opencode models` failed ({e}).[/yellow]")
    if ids:
        write_cache(OPENCODE_CACHE, ids)
        return ids

    stale = read_cache(OPENCODE_CACHE, 0)
    if stale:
        err.print("[yellow]Using the last cached opencode catalog.[/yellow]")
        return stale
    err.print("[yellow]No opencode catalog available; run `opencode models` once.[/yellow]")
    return []


def discoverable_providers(ptable, creds, only=None):
    """Providers whose own /models endpoint we can query with the keys we hold."""
    out = []
    for pid, entry in ptable.items():
        api = (entry or {}).get("api")
        if not api or "{" in api:  # templated endpoints need extra config
            continue
        if only and pid not in only:
            continue
        if resolve_key(pid, ptable, creds)[0]:
            out.append(pid)
    return sorted(out)


def strip_model_prefix(model_id):
    """Google-style `models/gemini-2.5-pro` -> `gemini-2.5-pro`.

    Gemini's /models endpoint prefixes every ID with `models/`; without this the
    catalog grows bogus `google/models/...` rows and no ID ever matches opencode's.
    """
    while model_id.startswith("models/"):
        model_id = model_id[len("models/"):]
    return model_id


def direct_models(ptable, creds, refresh=False, quiet=False):
    """provider -> model IDs taken straight from that provider's /models call.

    This is the "ask the key what it can reach" half of the catalog: whatever
    the stored keys unlock is listed, whether or not opencode knows about it.
    Cached for a day in DIRECT_CACHE; --refresh re-asks every provider.
    """
    cached = read_cache(DIRECT_CACHE, 0 if refresh else DISCOVER_TTL)
    if not isinstance(cached, dict):
        cached = {}
    targets = [pid for pid in discoverable_providers(ptable, creds)
               if refresh or pid not in cached]
    if targets and not quiet:
        err.print(f"[dim]Asking {len(targets)} provider(s) for their model list "
                  f"with the stored keys...[/dim]")

    def fetch(pid):
        api = (ptable.get(pid) or {}).get("api")
        key, _ = resolve_key(pid, ptable, creds)
        try:
            res = httpx.get(f"{api.rstrip('/')}/models",
                            headers={"Authorization": f"Bearer {key}"}, timeout=30.0)
            if res.status_code != 200:
                return pid, {"models": [], "error": f"HTTP {res.status_code}"}
            data = res.json()
            rows = data.get("data") if isinstance(data, dict) else data
            ids = sorted({strip_model_prefix(row["id"]) for row in rows or []
                          if isinstance(row, dict) and isinstance(row.get("id"), str)
                          and " " not in row["id"]})
            return pid, {"models": ids, "error": None}
        except Exception as e:  # noqa: BLE001 - one dead gateway must not stop the rest
            return pid, {"models": [], "error": type(e).__name__}

    if targets:
        with cf.ThreadPoolExecutor(max_workers=12) as pool:
            for pid, result in pool.map(fetch, targets):
                cached[pid] = result
        write_cache(DIRECT_CACHE, cached)

    failed = sum(1 for row in cached.values()
                 if isinstance(row, dict) and row.get("error"))
    if targets and failed and not quiet:
        err.print(f"[yellow]{failed} provider catalog(s) refused or unreachable; "
                  f"their opencode entries are still listed.[/yellow]")

    return {pid: (row.get("models") or [])
            for pid, row in cached.items() if isinstance(row, dict)}


def build_catalog(ptable, creds, include_direct=True, refresh=False, quiet=False):
    """opencode's catalog unioned with every provider's own /models result."""
    entries = {}
    for mid in opencode_model_ids(refresh=refresh):
        provider, _, model = mid.partition("/")
        category, chat_ok, tools_ok = classify_model(mid)
        entries[mid] = Model(mid, provider, model, "opencode", category, chat_ok, tools_ok)

    if include_direct:
        for pid, ids in direct_models(ptable, creds, refresh=refresh, quiet=quiet).items():
            for model in ids:
                full = f"{pid}/{model}"
                existing = entries.get(full)
                if existing is not None:
                    # Reachable both ways: opencode knows it, the key confirms it.
                    entries[full] = existing._replace(source="both")
                    continue
                category, chat_ok, tools_ok = classify_model(full)
                entries[full] = Model(full, pid, model, "direct", category, chat_ok, tools_ok)

    return sorted(entries.values(), key=lambda m: (m.provider, m.model))


def catalog_context(refresh=False, quiet=False):
    ptable = provider_table(refresh=refresh)
    creds = credentials(ptable)
    return ptable, creds, build_catalog(ptable, creds, refresh=refresh, quiet=quiet)


# --- helpers -----------------------------------------------------------------

def resolve_model(choice, entries, explain=True):
    """'#', exact ID, or unique substring -> Model. None on failure."""
    choice = (choice or "").strip()
    say = err.print if explain else (lambda *a, **k: None)
    if not choice:
        return None

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(entries):
            return entries[idx]
        say(f"[red]'{choice}' is out of range (1-{len(entries)}).[/red]")
        return None

    for m in entries:
        if choice.lower() == m.id.lower():
            return m

    matches = [m for m in entries if choice.lower() in m.id.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        say(f"[yellow]'{choice}' is ambiguous:[/yellow] "
            + ", ".join(m.id for m in matches[:5])
            + (" ..." if len(matches) > 5 else ""))
        return None
    say(f"[red]No model matches '{choice}'. Try 'models4me list -f {choice}'.[/red]")
    return None


def status_badge(model_id, status_map):
    verdict = status_map.get(model_id)
    if verdict == "live":
        return "[green]● live[/green]"
    if verdict in (None, "skip"):
        return "[dim]? unknown[/dim]"
    return f"[red]✗ {verdict}[/red]"


def key_badge(provider, ptable, creds):
    source = key_source(provider, ptable, creds)
    if not source:
        return "[red]no key[/red]"
    if source.startswith("env:"):
        return f"[green]✓ {source[4:]}[/green]"
    if source == "omnicli":
        return "[green]✓ omnicli[/green]"
    return "[green]✓ opencode[/green]"


def capability_note(model):
    if not model.chat:
        return "[red]not a chat model[/red]"
    if not model.tools:
        return "[yellow]chat only (no tools)[/yellow]"
    return "[green]tools + chat[/green]"


def filter_entries(entries, keyword=None, provider=None, category=None,
                   live=None, status_map=None, source=None):
    out = entries
    if provider:
        prov = provider.lower()
        out = [m for m in out if m.provider.lower() == prov or prov in m.provider.lower()]
    if keyword:
        kw = keyword.lower()
        out = [m for m in out if kw in m.id.lower() or kw in m.category.lower()]
    if category:
        cat = category.lower()
        out = [m for m in out if cat in m.category.lower() or m.category.lower() in cat]
    if source:
        want = source.lower()
        out = [m for m in out if m.source == want or (want == "key" and m.source in ("direct", "both"))]
    if live:
        out = [m for m in out if (status_map or {}).get(m.id) == "live"]
    return out


def apply_limit(entries, limit):
    return entries if not limit or limit <= 0 else entries[:limit]


# --- commands ----------------------------------------------------------------

def source_badge(source):
    """Where a row came from: opencode's catalog, the provider's /models, or both."""
    if source == "both":
        return "[green]both[/green]"
    if source == "direct":
        return "[cyan]key[/cyan]"
    if source == "opencode":
        return "[dim]oc[/dim]"
    return "[dim]?[/dim]"


def cmd_list(args):
    ptable, creds, catalog = catalog_context(refresh=args.refresh, quiet=args.json)
    status = load_status()
    status_map = status.get("models", {})
    default_model = load_config().get("default_model")

    entries = filter_entries(catalog, args.filter, args.provider, args.category,
                             args.live, status_map, args.source)

    if args.json:
        limit = args.limit or 0
        payload = [
            {
                "index": i,
                "model": m.id,
                "provider": m.provider,
                "api_model": m.model,
                "source": m.source,
                "category": m.category,
                "chat": m.chat,
                "tools": m.tools,
                "status": status_map.get(m.id) or "unknown",
                "key": key_source(m.provider, ptable, creds),
                "default": m.id == default_model,
            }
            for i, m in enumerate(apply_limit(entries, limit), 1)
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if not entries:
        console.print("[yellow]No models match those filters.[/yellow]")
        return 0

    limit = 40 if args.limit is None else args.limit
    shown = apply_limit(entries, limit)

    scope = args.provider or args.category or args.filter or "All sources"
    title = f"Models4me — {scope} ({len(entries)} models)"
    # The ID already carries the provider (provider/model), so no separate
    # provider column: keeping the table narrow stops rich from squeezing the
    # model column to nothing on an 80-column terminal.
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan", expand=False)
    table.add_column("#", style="cyan", justify="right", no_wrap=True, width=3)
    table.add_column("Model", style="bold green", no_wrap=True,
                     max_width=30, overflow="ellipsis")
    table.add_column("From", no_wrap=True, width=4)
    table.add_column("Category", style="bright_yellow", no_wrap=True,
                     max_width=10, overflow="ellipsis")
    table.add_column("Key", no_wrap=True, max_width=9, overflow="ellipsis")
    table.add_column("Status", no_wrap=True, max_width=9, overflow="ellipsis")

    for i, m in enumerate(shown, 1):
        label = m.id + (" [magenta]★[/magenta]" if m.id == default_model else "")
        table.add_row(str(i), label, source_badge(m.source), m.category,
                      key_badge(m.provider, ptable, creds),
                      status_badge(m.id, status_map))

    console.print(table)
    if limit and len(entries) > limit:
        console.print(f"[dim]Showing {limit} of {len(entries)} — use --limit 0 for all, "
                      f"--json for scripting.[/dim]")
    by_key = sum(1 for m in catalog if m.source in ("direct", "both"))
    both = sum(1 for m in catalog if m.source == "both")
    console.print(f"[dim]{len(catalog)} models — From: [dim]oc[/dim] = opencode only "
                  f"({len(catalog) - by_key}), [cyan]key[/cyan] = the provider returned it "
                  f"for your key ({by_key - both}), [green]both[/green] ({both}).[/dim]")
    console.print(f"[dim]Every opencode model plus everything the stored keys reach; "
                  f"'--source opencode|direct|both|key' filters.[/dim]")
    if status.get("probed_at"):
        console.print(f"[dim]Status from the last probe ({status['probed_at']}).[/dim]")
    else:
        console.print("[yellow]Status unknown — run 'models4me probe -p <provider>'.[/yellow]")
    console.print("[dim]'models4me use <#>' remembers a default; 'models4me ask <model> \"...\"' "
                  "talks to one.[/dim]\n")
    return 0


def cmd_providers(args):
    ptable, creds, catalog = catalog_context(refresh=args.refresh, quiet=args.json)
    counts, direct_counts = {}, {}
    for m in catalog:
        counts[m.provider] = counts.get(m.provider, 0) + 1
        if m.source in ("direct", "both"):
            direct_counts[m.provider] = direct_counts.get(m.provider, 0) + 1

    rows = []
    for pid in sorted(counts, key=lambda p: (-counts[p], p)):
        api = (ptable.get(pid) or {}).get("api")
        rows.append({
            "provider": pid,
            "models": counts[pid],
            "from_key": direct_counts.get(pid, 0),
            "endpoint": api,
            "key": key_source(pid, ptable, creds),
            "direct": bool(api) and bool(key_source(pid, ptable, creds)),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    table = Table(title=f"Providers behind Models4me ({len(rows)})",
                  box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider", style="magenta", no_wrap=True, max_width=18)
    table.add_column("Models", justify="right", style="bold green")
    table.add_column("From key", justify="right", style="cyan")
    table.add_column("Key", no_wrap=True, max_width=11, overflow="ellipsis")
    table.add_column("Endpoint", style="dim", overflow="fold")
    for row in rows:
        table.add_row(row["provider"], str(row["models"]), str(row["from_key"]),
                      key_badge(row["provider"], ptable, creds),
                      row["endpoint"] or "— via opencode")
    console.print(table)
    with_key = sum(1 for r in rows if r["key"])
    console.print(f"[dim]{with_key}/{len(rows)} providers have a key (omnicli, opencode, "
                  f"or the environment). 'From key' counts models the provider itself "
                  f"returned for that key; everything else comes from opencode.\n"
                  f"--refresh re-asks the providers.[/dim]\n")
    return 0


def catalog_diff(ptable, creds, refresh=False, quiet=False):
    """Per provider: opencode's IDs vs the IDs the provider returned for your key."""
    opencode = {}
    for mid in opencode_model_ids(refresh=refresh):
        provider, _, model = mid.partition("/")
        opencode.setdefault(provider, set()).add(strip_model_prefix(model))

    direct = direct_models(ptable, creds, refresh=refresh, quiet=quiet)
    rows = []
    for pid in sorted(set(opencode) | set(direct)):
        a, b = opencode.get(pid, set()), set(direct.get(pid, []))
        rows.append({
            "provider": pid,
            "opencode": len(a),
            "key": len(b),
            "shared": len(a & b),
            "only_opencode": sorted(a - b),
            "only_key": sorted(b - a),
        })
    return rows


def cmd_diff(args):
    ptable, creds, _ = catalog_context(refresh=args.refresh, quiet=args.json)
    rows = catalog_diff(ptable, creds, refresh=args.refresh, quiet=args.json)
    if args.provider:
        want = args.provider.lower()
        rows = [r for r in rows if want == r["provider"] or want in r["provider"]]
        if not rows:
            err.print(f"[red]No provider matches {args.provider!r}.[/red]")
            return 1

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    table = Table(title="opencode's list vs the providers' own answers",
                  box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider", style="magenta", no_wrap=True, max_width=18)
    table.add_column("opencode", justify="right", style="green")
    table.add_column("key", justify="right", style="cyan")
    table.add_column("shared", justify="right", style="bold green")
    table.add_column("oc only", justify="right", style="yellow")
    table.add_column("key only", justify="right", style="yellow")
    for r in rows:
        table.add_row(r["provider"], str(r["opencode"]), str(r["key"]),
                      str(r["shared"]), str(len(r["only_opencode"])),
                      str(len(r["only_key"])))
    console.print(table)

    limit = args.limit
    if len(rows) == 1:
        r = rows[0]
        for label, ids in (("listed by opencode, not returned by the provider",
                            r["only_opencode"]),
                           ("returned by the provider, not listed by opencode",
                            r["only_key"])):
            if not ids:
                continue
            shown = ids if limit == 0 else ids[:limit]
            console.print(f"\n[bold]{label}[/bold] ({len(ids)})")
            for mid in shown:
                console.print(f"  {r['provider']}/{mid}")
            if len(shown) < len(ids):
                console.print(f"  [dim]… {len(ids) - len(shown)} more (--limit 0)[/dim]")

    same = sum(1 for r in rows if r["opencode"] and r["key"] and r["shared"] == r["opencode"] == r["key"])
    console.print(f"\n[dim]'shared' is the same model ID returned by both sides. "
                  f"{same}/{len(rows)} providers agree exactly. Pass a provider name "
                  f"to list its differing IDs.[/dim]\n")
    return 0


def probe_model(entry, ptable, creds, timeout=90.0):
    """One minimal request straight at the provider. Returns 'live' or a short reason.

    Reasoning models can take minutes to emit even a short answer, so a slow
    provider is reported as 'timeout' rather than 'down' and never cached.
    """
    api = (ptable.get(entry.provider) or {}).get("api")
    if not entry.chat:
        return "skip"
    if not api:
        return "no-endpoint"
    key, _ = resolve_key(entry.provider, ptable, creds)
    if not key:
        return "no-key"
    try:
        res = httpx.post(
            f"{api.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": entry.model,
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 16},
            timeout=timeout,
        )
        if res.status_code == 200:
            return "live"
        return str(res.status_code)
    except httpx.TimeoutException:
        return "timeout"
    except Exception:  # noqa: BLE001
        return "down"


def cmd_probe(args):
    ptable, creds, catalog = catalog_context()
    entries = filter_entries(catalog, args.filter, args.provider, args.category, False, None)

    if args.models:
        picked, seen = [], set()
        for sel in args.models:
            hits = [m for m in entries if sel.lower() in m.id.lower()]
            if not hits:
                err.print(f"[yellow]No model matches '{sel}'.[/yellow]")
            for m in hits:
                if m.id not in seen:
                    seen.add(m.id)
                    picked.append(m)
        entries = picked

    if not args.all and not (args.provider or args.filter or args.category or args.models):
        err.print("[red]Refusing to probe every model.[/red] Narrow it down with "
                  "-p/--provider, -f, or a model name — or pass --all.")
        return 1
    if not entries:
        err.print("[red]Nothing to probe.[/red]")
        return 1

    if not args.json:
        console.print(f"Probing [bold cyan]{len(entries)}[/bold cyan] model(s) directly...")
    results = {}
    quiet = args.json

    def run(entry):
        results[entry.id] = probe_model(entry, ptable, creds, args.timeout)

    if quiet:
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, entries))
    else:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("Probing models...", total=len(entries))
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                for _ in pool.map(run, entries):
                    progress.advance(task)

    status = load_status()
    status.setdefault("models", {})
    status["models"].update({k: v for k, v in results.items() if v not in RETRYABLE})
    status["probed_at"] = time.strftime("%Y-%m-%d %H:%M")
    save_status(status)

    live = sum(1 for v in results.values() if v == "live")
    if args.json:
        print(json.dumps({"probed_at": status["probed_at"], "live": live,
                          "total": len(results), "results": results}, indent=2,
                         sort_keys=True))
        return 0

    console.print(f"[bold green]{live}/{len(results)} live.[/bold green] "
                  f"[dim]Cached in {STATUS_PATH}[/dim]")
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Model", style="bold green")
    table.add_column("Provider", style="magenta")
    table.add_column("Status")
    for m in entries:
        table.add_row(m.id, m.provider, status_badge(m.id, results))
    console.print(table)
    return 0


def call_direct(entry, messages, max_tokens, temperature, timeout, ptable, creds):
    api = (ptable.get(entry.provider) or {}).get("api")
    if not api:
        raise RuntimeError(f"no direct endpoint known for provider '{entry.provider}'")
    key, source = resolve_key(entry.provider, ptable, creds)
    if not key:
        raise RuntimeError(f"no API key for provider '{entry.provider}' "
                           f"(set {', '.join((ptable.get(entry.provider) or {}).get('env') or ['a key'])})")
    res = httpx.post(
        f"{api.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": entry.model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout,
    )
    try:
        data = res.json()
    except ValueError:
        raise RuntimeError(f"HTTP {res.status_code}: {res.text[:300]}")
    if res.status_code != 200:
        detail = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        raise RuntimeError(f"HTTP {res.status_code}: {detail or res.text[:300]}")
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or message.get("reasoning_content") or ""
    return {"text": text.strip(), "via": "direct", "key": source,
            "usage": data.get("usage"), "finish_reason": choice.get("finish_reason"),
            "model": entry.id}


def call_opencode(entry, prompt, timeout):
    exe = shutil.which("opencode")
    if not exe:
        raise RuntimeError("opencode is not on PATH")
    try:
        res = subprocess.run(
            [exe, "run", "-m", entry.id, "--format", "json", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"opencode timed out after {timeout}s")

    parts, failure = [], None
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "error":
            failure = ((event.get("error") or {}).get("message")) or "unknown error"
        part = event.get("part") or {}
        if part.get("type") == "text" and part.get("text"):
            parts.append(part["text"])
    text = "".join(parts).strip()
    if not text and failure:
        raise RuntimeError(f"opencode: {failure}")
    if not text:
        text = ANSI.sub("", res.stdout).strip()
    if not text:
        detail = ANSI.sub("", (res.stderr or "")).strip()
        raise RuntimeError(f"opencode returned nothing{f': {detail[:300]}' if detail else ''}")
    return {"text": text, "via": "opencode", "key": "opencode", "model": entry.id}


def cmd_ask(args):
    ptable, creds, catalog = catalog_context()
    entry = resolve_model(args.model, catalog, explain=args.json is False)
    if entry is None:
        return 1
    if not entry.chat:
        err.print(f"[red]{entry.id} is a {entry.category.lower()} model; it cannot chat.[/red]")
        return 1

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        err.print("[red]Nothing to ask — pass a prompt.[/red]")
        return 1

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    api = (ptable.get(entry.provider) or {}).get("api")
    has_key = bool(resolve_key(entry.provider, ptable, creds)[0])
    via = args.via
    if via == "auto":
        via = "direct" if (api and has_key) else "opencode"

    try:
        if via == "direct":
            result = call_direct(entry, messages, args.max_tokens, args.temperature,
                                 args.timeout, ptable, creds)
        else:
            result = call_opencode(entry, prompt, args.timeout)
    except Exception as e:  # noqa: BLE001 - surfaced as a CLI error
        # A model opencode lists is not always served by the provider for your
        # key (and the reverse), so a failed direct call retries through opencode.
        if via == "direct" and args.via == "auto":
            if not args.json:
                err.print(f"[yellow]Direct call failed ({e}); trying opencode…[/yellow]")
            try:
                result = call_opencode(entry, prompt, args.timeout)
                result["direct_error"] = str(e)
            except Exception as e2:  # noqa: BLE001
                if args.json:
                    print(json.dumps({"error": str(e2), "direct_error": str(e),
                                      "model": entry.id, "via": "opencode"}, indent=2))
                else:
                    err.print(f"[red]Both paths failed.[/red] direct: {e} | opencode: {e2}")
                return 1
        else:
            if args.json:
                print(json.dumps({"error": str(e), "model": entry.id, "via": via}, indent=2))
            else:
                err.print(f"[red]{via} call failed:[/red] {e}")
            return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    console.print(f"[dim]→ {result['model']} via {result['via']}[/dim]")
    if result["text"]:
        console.print(result["text"])
    elif result.get("finish_reason") == "length":
        console.print("[yellow](no visible text: the model spent the whole token budget "
                      "thinking — raise --max-tokens)[/yellow]")
    else:
        console.print("[yellow](empty response)[/yellow]")
    return 0


def cmd_show(args):
    ptable, creds, catalog = catalog_context()
    status_map = load_status().get("models", {})
    default_model = load_config().get("default_model")

    model = resolve_model(args.model, catalog, explain=False)
    known = model is not None
    if model is None:
        if not args.model.strip():
            return 1
        category, chat_ok, tools_ok = classify_model(args.model)
        provider, _, api_model = args.model.partition("/")
        model = Model(args.model.strip(), provider, api_model or args.model, "unknown",
                      category, chat_ok, tools_ok)

    api = (ptable.get(model.provider) or {}).get("api")
    key = key_source(model.provider, ptable, creds)

    if args.json:
        print(json.dumps({
            "model": model.id,
            "provider": model.provider,
            "api_model": model.model,
            "source": model.source,
            "category": model.category,
            "chat": model.chat,
            "tools": model.tools,
            "endpoint": api,
            "key": key,
            "status": status_map.get(model.id) or "unknown",
            "default": model.id == default_model,
            "in_catalog": known,
            "confirmed_by_key": model.source in ("direct", "both"),
        }, indent=2))
        return 0

    body = (
        f"[bold cyan]Model:[/bold cyan]      {model.id}\n"
        f"[bold cyan]Provider:[/bold cyan]   {model.provider} [dim]({model.source})[/dim]\n"
        f"[bold cyan]API model:[/bold cyan]  {model.model}\n"
        f"[bold cyan]Category:[/bold cyan]   {model.category}\n"
        f"[bold cyan]Usable as:[/bold cyan]  {capability_note(model)}\n"
        f"[bold cyan]Endpoint:[/bold cyan]   {api or '[dim]— use --via opencode[/dim]'}\n"
        f"[bold cyan]Key:[/bold cyan]        {key or '[red]not configured[/red]'}\n"
        f"[bold cyan]Status:[/bold cyan]     {status_badge(model.id, status_map)}\n"
        f"[bold cyan]Default:[/bold cyan]    {'yes' if model.id == default_model else 'no'}"
    )
    if not known:
        body += "\n\n[yellow]Not in the catalog — opencode or the provider may not serve it.[/yellow]"
    elif model.source == "opencode" and api and key:
        body += ("\n\n[yellow]opencode lists this, but the provider's /models did not "
                 "return it for your key — a direct call may be rejected; ask falls "
                 "back to opencode.[/yellow]")
    console.print(Panel(body, title=f"Models4me — {model.id}", border_style="cyan"))
    return 0


def cmd_use(args):
    _, _, catalog = catalog_context()
    model = resolve_model(args.model, catalog)
    if model is None:
        return 1
    if not model.chat:
        err.print(f"[red]{model.id} is a {model.category.lower()} model and cannot be "
                  f"a default chat model.[/red]")
        return 1

    config = load_config()
    previous = config.get("default_model")
    config["default_model"] = model.id
    config["set_at"] = time.strftime("%Y-%m-%d %H:%M")
    save_config(config)

    if args.json:
        print(json.dumps({"default_model": model.id, "previous": previous}, indent=2))
        return 0
    console.print(f"[bold green]Default model set to {model.id}[/bold green]")
    if previous and previous != model.id:
        console.print(f"[dim]Was: {previous}[/dim]")
    console.print(f"[dim]Saved in {CONFIG_PATH}[/dim]\n")
    return 0


def cmd_current(args):
    config = load_config()
    if args.json:
        print(json.dumps({"default_model": config.get("default_model"),
                          "set_at": config.get("set_at")}, indent=2))
        return 0
    if not config.get("default_model"):
        console.print("[yellow]No default model set. Pick one with 'models4me use <#>'.[/yellow]")
        return 0
    console.print(f"[bold green]{config['default_model']}[/bold green]"
                  + (f" [dim](set {config['set_at']})[/dim]" if config.get("set_at") else ""))
    return 0


def cmd_categories(args):
    _, _, catalog = catalog_context()
    counts = {}
    for m in catalog:
        counts[m.category] = counts.get(m.category, 0) + 1
    ordered = sorted(counts, key=lambda c: (ORDER_INDEX.get(c, len(ORDER_INDEX)), c))
    if args.json:
        print(json.dumps([{"category": c, "models": counts[c]} for c in ordered], indent=2))
        return 0
    table = Table(title="Categories", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Category", style="bright_yellow")
    table.add_column("Models", justify="right", style="bold green")
    for c in ordered:
        table.add_row(c, str(counts[c]))
    console.print(table)
    return 0


# --- Freebuff mode -----------------------------------------------------------
#
# Freebuff has no public model API (`/api/models` and friends all 404), so the
# two pages it keeps current are the source: the README's model table carries
# each model's access level and what it is for, and /plans carries the plan
# ladder. Both are fetched live, cached for a day, and fall back to the
# snapshot below when neither is reachable.

FREEBUFF_FALLBACK_MODELS = [
    {"model": "MiMo v2.6 Flash Fast", "access": "Full and limited access",
     "best_for": "The default everywhere; balanced performance with image support"},
    {"model": "GLM 5.3 Flash", "access": "Full and limited access",
     "best_for": "Deepest reasoning; replaced DeepSeek V4 Pro as the reasoning pick"},
    {"model": "DeepSeek V4.1 Flash 10/17", "access": "Full and limited access",
     "best_for": "Fast coding and tool use"},
    {"model": "DeepSeek V4.1 Flash Fast 10/17",
     "access": "Full access; paid plans at limited access",
     "best_for": "Fast mode: the same model on DeepSeek's own API, splitting work across "
                 "parallel subagents and verifying the result"},
    {"model": "GPT-6 Luna", "access": "Full access",
     "best_for": "Strong all-around with native images; runs on OpenAI flex capacity"},
    {"model": "MiMo 2.6 Pro", "access": "Full access",
     "best_for": "Xiaomi's stronger reasoning model, with image support"},
    {"model": "Solar Mini 4", "access": "Full and limited access",
     "best_for": "Upstage's fast, compact model; 524K context, text only"},
    {"model": "Solar Pro 4", "access": "Full and limited access",
     "best_for": "Upstage's larger, stronger model; 524K context, text only"},
    {"model": "Gemini 3.8 Flash", "access": "Paid plans",
     "best_for": "1M context, and the only model that accepts audio, video and PDF"},
    {"model": "Muse Spark 1.3", "access": "Paid plans",
     "best_for": "Meta's agentic coding model; 1M context; answers on DeepSeek V4.1 "
                 "Flash when it is busy"},
    {"model": "Gemini 4 Argon", "access": "US, or paid plans elsewhere",
     "best_for": "Google's flagship reasoning model; 1M context, images"},
    {"model": "GPT-6.1 Terra", "access": "Full access; paid plans at limited access",
     "best_for": "OpenAI's model at a temporary promotional price; 1M context, images"},
    {"model": "GPT-6.1 Sol", "access": "US, or paid plans elsewhere",
     "best_for": "OpenAI's flagship; 1M context, images"},
]

# Offered by Freebuff but not in the main picker: the specialist model, the
# Unlimited section, and the 1x baselines the plans page prices by.
FREEBUFF_FALLBACK_EXTRAS = [
    {"model": "Gemini 3.1 Flash Lite", "access": "specialist", "listed_in": "docs",
     "best_for": "Powers file finding and research rather than appearing in the picker"},
    {"model": "Claude Haiku 5.5", "access": "Paid plans", "listed_in": "plans page",
     "best_for": "Anthropic's fast small model; 100k context"},
    {"model": "Glyph Cluster", "access": "Unlimited section", "listed_in": "plans page",
     "best_for": "Spends none of your daily hours"},
    {"model": "Nemotron 3.5 Lightning", "access": "Unlimited section",
     "listed_in": "plans page", "best_for": "Fast NVIDIA model; long hours per day"},
    {"model": "MiMo v2.6 Flash Slow", "access": "Full and limited access",
     "listed_in": "plans page", "best_for": "The 1x baseline: an hour here costs one hour"},
    {"model": "DeepSeek V4.1 Flash Slow", "access": "Full and limited access",
     "listed_in": "plans page", "best_for": "Cheapest DeepSeek route: ~25 hrs/day on free"},
]

FREEBUFF_UNLOCK_NOTE = ("GPT-6 Luna, DeepSeek V4.1 Flash Fast 10/17, Muse Spark 1.3, "
                        "Claude Haiku 5.5, GPT-6.1 Terra")
FREEBUFF_FALLBACK_PLANS = [
    {"plan": "Free", "price": 0, "hours": 30, "bonus": 0, "unlocks": None},
    {"plan": "Starter", "price": 8, "hours": 50, "bonus": 30,
     "unlocks": FREEBUFF_UNLOCK_NOTE},
    {"plan": "Plus", "price": 25, "hours": 80, "bonus": 50,
     "unlocks": FREEBUFF_UNLOCK_NOTE},
    {"plan": "Pro", "price": 60, "hours": 130, "bonus": 100,
     "unlocks": FREEBUFF_UNLOCK_NOTE},
]


def html_to_text(markup):
    """Tags out, text in: enough to read a marketing page's numbers."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text)


def parse_freebuff_readme(text):
    """Freebuff's model table: name, access level, what it is for.

    Only the "## Models" section counts — the product table higher up has the
    same three-column shape but lists Freebuff Desktop/CLI/Web/Cloud/Chat.
    """
    models = []
    in_models = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            in_models = stripped.lower().startswith("## models")
            continue
        if not in_models or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 3:
            continue
        name, access, best_for = cells
        if "**" not in name:  # header and separator rows are not bold
            continue
        name = name.replace("**", "").strip()
        if not name or not access:
            continue
        models.append({"model": name, "access": access,
                       "best_for": best_for, "listed_in": "picker"})
    return models


def parse_freebuff_plans(markup):
    """[{plan, price, hours, bonus, unlocks}] from the plans page."""
    text = html_to_text(markup)
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r"\b(Free|Starter|Plus|Pro)\b\s*\$\s*(\d+)", text)]
    plans = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else min(len(text), pos + 900)
        segment = text[pos:end]
        price = re.search(r"\$\s*(\d+)", segment)
        hours = re.search(r"(\d+)\s*hours refilled daily", segment)
        if not price or not hours:
            continue
        bonus = re.search(r"(\d+)\s*hours monthly bonus", segment)
        unlocks = re.search(r"Unlocks ([A-Z][^\n]{3,200}?)\s*(?:Per day|$)", segment)
        plans.append({"plan": name, "price": int(price.group(1)),
                      "hours": int(hours.group(1)),
                      "bonus": int(bonus.group(1)) if bonus else 0,
                      "unlocks": unlocks.group(1).strip() if unlocks else None})
    return plans


def freebuff_level(access):
    """One short label for a model's access line."""
    low = (access or "").lower()
    if "unlimited" in low:
        return "unlimited"
    if low.strip() == "specialist":
        return "specialist"
    if "paid plans at limited" in low:
        return "free, paid in limited"
    if "us" in low and "paid" in low:
        return "US free, else paid"
    if "paid" in low:
        return "paid only"
    if "full and limited" in low:
        return "free + limited"
    if "full access" in low:
        return "free, full only"
    return "—"


def freebuff_plan_label(level):
    """How to read the level as a plan requirement."""
    return {
        "free + limited": "Free",
        "free, full only": "Full mode",
        "free, paid in limited": "Free/paid",
        "US free, else paid": "US / paid",
        "paid only": "Starter+",
        "unlimited": "0 hours",
        "specialist": "in-app",
    }.get(level, "—")


def freebuff_aliases(name):
    """Catalog ID fragments a Freebuff display name is likely to appear as.

    "MiMo v2.6 Flash" -> mimo-v2.6-flash; "MiMo 2.6 Pro" -> mimo-2.6-pro and
    mimo-v2.6-pro; "Solar Mini 4" -> solar-mini-4 and solar-mini4.
    """
    base = name.lower().replace("/", " ").replace("·", " ")
    base = re.sub(r"\(.*?\)", " ", base)
    # Version first: with [a-z]+ ahead of it, "V4.1" tokenises as v + 4.1 and the
    # short fallback becomes "deepseek-v", which matches deepseek-v3.
    toks = re.findall(r"v?\d+(?:\.\d+)?|[a-z]+", base)
    out = []

    def add(seq):
        slug = "-".join(t for t in seq if t)
        if slug and slug not in out:
            out.append(slug)

    core = toks[:3]
    add(core)
    if len(core) >= 2 and re.match(r"^\d", core[1]):
        add([core[0], "v" + core[1]] + core[2:])
    if len(core) >= 3:
        # solar-mini4 / gpt-6luna: the catalog drops the last separator sometimes.
        add([core[0] + "-" + core[1] + core[2]])
    if len(toks) >= 4:
        add(toks[:4])
    return out


def freebuff_matches(name, catalog):
    """Catalog entries for a Freebuff model, closest ID first."""
    aliases = freebuff_aliases(name)
    hits = []
    for entry in catalog:
        low = entry.id.lower()
        if any(alias in low for alias in aliases):
            hits.append(entry)
    hits.sort(key=lambda e: (len(e.id), e.id))
    return hits


def freebuff_data(refresh=False):
    """Freebuff's models + levels, from its own pages when reachable."""
    # read_cache treats ttl 0 as "never expires", so --refresh skips the read.
    cached = None if refresh else read_cache(FREEBUFF_CACHE, FREEBUFF_TTL)
    if isinstance(cached, dict) and len(cached.get("models") or []) >= 5:
        return cached

    live = {"readme": False, "plans": False}
    models, plans = [], []
    try:
        res = httpx.get(FREEBUFF_README, timeout=25.0, follow_redirects=True)
        res.raise_for_status()
        models = parse_freebuff_readme(res.text)
        live["readme"] = bool(models)
    except Exception:  # noqa: BLE001 - offline is a supported state
        pass
    try:
        res = httpx.get(FREEBUFF_PLANS, timeout=25.0, follow_redirects=True)
        res.raise_for_status()
        plans = parse_freebuff_plans(res.text)
        live["plans"] = len(plans) >= 4
    except Exception:  # noqa: BLE001
        pass

    if not live["readme"]:
        models = [dict(row) for row in FREEBUFF_FALLBACK_MODELS]
    if not live["plans"]:
        plans = [dict(row) for row in FREEBUFF_FALLBACK_PLANS]

    seen = {m["model"].lower() for m in models}
    for row in FREEBUFF_FALLBACK_EXTRAS:
        if row["model"].lower() not in seen:
            models.append(dict(row))
            seen.add(row["model"].lower())

    # Limited mode covers exactly the models whose access line says so.
    limited = [row["model"] for row in models
               if "full and limited" in (row.get("access") or "").lower()]
    limited_set = {name.lower() for name in limited}
    for row in models:
        row["limited"] = row["model"].lower() in limited_set
        row["level"] = freebuff_level(row.get("access"))
        row["plans"] = freebuff_plan_label(row["level"])

    payload = {"models": models, "plans": plans, "limited": limited,
               "live": live, "fetched_at": time.strftime("%Y-%m-%d %H:%M")}
    if live["readme"]:
        write_cache(FREEBUFF_CACHE, payload)
    return payload


def cmd_extrafuff(args):
    data = freebuff_data(refresh=args.refresh)
    try:
        catalog = catalog_context(quiet=True)[2]
    except Exception:  # noqa: BLE001 - the Freebuff answer stands without our catalog
        catalog = []

    rows = []
    for row in data["models"]:
        hits = freebuff_matches(row["model"], catalog)
        rows.append(dict(row, catalog_count=len(hits),
                         catalog_ids=[e.id for e in hits[:8]],
                         catalog_sample=hits[0].id if hits else None))

    wanted = (args.model or "").strip().lower()
    if wanted:
        rows = [r for r in rows
                if wanted in r["model"].lower() or wanted in r["level"].lower()
                or wanted in r["plans"].lower()]
        if not rows:
            err.print(f"[red]No Freebuff model matches {args.model!r}.[/red]")
            return 1

    if args.json:
        print(json.dumps({"fetched_at": data["fetched_at"], "live": data["live"],
                          "limited_mode": data["limited"], "plans": data["plans"],
                          "models": rows}, indent=2))
        return 0

    if not rows:
        console.print("[yellow]Freebuff lists no models right now.[/yellow]")
        return 0

    noun = "model" if len(rows) == 1 else "models"
    table = Table(title=f"Freebuff — {len(rows)} {noun}, levels and criteria",
                  box=box.ROUNDED, header_style="bold cyan", expand=False)
    table.add_column("Model", style="bold green", no_wrap=True, max_width=26,
                     overflow="ellipsis")
    table.add_column("Level", no_wrap=True, max_width=22, overflow="ellipsis")
    table.add_column("Plan", no_wrap=True, max_width=11, overflow="ellipsis")
    table.add_column("Keys", justify="right", no_wrap=True)

    for row in rows:
        keys = (f"[green]{row['catalog_count']}[/green]" if row["catalog_count"]
                else "[dim]0[/dim]")
        table.add_row(row["model"], row["level"], row["plans"], keys)

    if not args.verbose:
        console.print(table)
    else:
        for row in rows:
            console.print(f"[bold green]{row['model']}[/bold green]  "
                          f"[dim]({row['level']}, {row['plans']}, "
                          f"{row.get('listed_in', 'picker')})[/dim]")
            console.print(f"  [cyan]Access:[/cyan]   {row['access']}")
            console.print(f"  [cyan]Best for:[/cyan] {row['best_for'] or '—'}")
            if row["catalog_count"]:
                console.print(f"  [cyan]Catalog:[/cyan]  {row['catalog_count']} IDs, e.g. "
                              + ", ".join(row["catalog_ids"][:4]))
            else:
                console.print("  [cyan]Catalog:[/cyan]  [yellow]no matching ID for "
                              "your keys[/yellow]")
            console.print()

    levels = Table(title="How the levels work", box=box.SIMPLE,
                   header_style="bold cyan", expand=False)
    levels.add_column("Plan", style="magenta", no_wrap=True)
    levels.add_column("$", justify="right")
    levels.add_column("Hours/day", justify="right", style="bold green")
    levels.add_column("Monthly bonus", justify="right", style="cyan")
    levels.add_column("Unlocks", style="dim", overflow="fold", max_width=34)
    for plan in data["plans"]:
        levels.add_row(plan["plan"], f"{plan['price']}", str(plan["hours"]),
                       f"{plan['bonus']} h" if plan.get("bonus") else "—",
                       plan.get("unlocks") or "—")
    console.print(levels)

    with_key = sum(1 for r in rows if r["catalog_count"])
    console.print(f"[dim]Level = where it is offered: 'free + limited' reaches limited "
                  f"mode, 'Full mode' needs a full-access country, 'Free/paid' is free "
                  f"with full access but paid plans in limited mode, 'US / paid' is US "
                  f"accounts only, 'Starter+' needs any paid plan, '0 hours' spends none "
                  f"of your daily balance.[/dim]")
    console.print(f"[dim]Limited mode (other regions and VPNs): 5 h/day, 4 on a VPN, "
                  f"restricted to {', '.join(data['limited'])}.[/dim]")
    console.print(f"[dim]One session is one hour with unlimited messages; daily hours "
                  f"reset at local midnight and do not roll over; 1 hour = 5 Freebucks; "
                  f"models in the Unlimited section spend none.[/dim]")
    console.print(f"[dim]{with_key}/{len(rows)} of these also exist in your own catalog "
                  f"(the Keys column) — `extraFuff -v` shows the matching IDs, "
                  f"`ask <id>` calls them with your keys.[/dim]")
    source = ("README + plans page" if all(data["live"].values())
              else "README" if data["live"]["readme"]
              else "plans page" if data["live"]["plans"]
              else "built-in snapshot (offline)")
    console.print(f"[dim]Source: {source}, read {data['fetched_at']}. Per-model "
                  f"multipliers (1x…3x, Unlimited) appear only in Freebuff's own "
                  f"picker, so they are not listed here.[/dim]\n")
    return 0


# --- entry point -------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="models4me",
        description="Browse, probe and talk to every model opencode can reach, "
                    "plus every model your own API keys can reach directly.",
        epilog="Run 'models4me COMMAND --help' for command options.",
    )
    parser.add_argument("-V", "--version", action="version", version="models4me 3.2.0")
    sub = parser.add_subparsers(dest="command")

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        return p

    p_list = add("list", "list models from opencode + the providers your keys unlock")
    p_list.add_argument("-p", "--provider", metavar="NAME", help="only this provider")
    p_list.add_argument("-f", "--filter", metavar="KW", help="match a keyword in model or category")
    p_list.add_argument("-c", "--category", metavar="CAT", help="only this category")
    p_list.add_argument("-s", "--source", metavar="SRC",
                        choices=["opencode", "direct", "both", "key"],
                        help="only models from this source ('key' = confirmed by a stored key)")
    p_list.add_argument("-l", "--live", action="store_true", help="only models verified live")
    p_list.add_argument("-n", "--limit", type=int, default=None,
                        help="rows to show (0 = all; default 40, all with --json)")
    p_list.add_argument("--refresh", action="store_true",
                        help="re-ask opencode and every provider instead of using caches")
    p_list.set_defaults(func=cmd_list)

    p_prov = add("providers", "providers, endpoints, keys and model counts")
    p_prov.add_argument("--refresh", action="store_true",
                        help="re-ask every provider for its model list")
    p_prov.set_defaults(func=cmd_providers)

    p_diff = add("diff", "compare opencode's model list with what your keys reach")
    p_diff.add_argument("provider", nargs="?", help="provider to list differing IDs for")
    p_diff.add_argument("-n", "--limit", type=int, default=15,
                        help="IDs to print per side (0 = all; default 15)")
    p_diff.add_argument("--refresh", action="store_true",
                        help="re-ask opencode and every provider")
    p_diff.set_defaults(func=cmd_diff)

    p_probe = add("probe", "test models with a real request and cache the verdict")
    p_probe.add_argument("models", nargs="*", metavar="MODEL",
                         help="limit the probe to these models")
    p_probe.add_argument("-p", "--provider", metavar="NAME", help="only this provider")
    p_probe.add_argument("-f", "--filter", metavar="KW", help="match a keyword")
    p_probe.add_argument("-c", "--category", metavar="CAT", help="only this category")
    p_probe.add_argument("--all", action="store_true", help="probe every matching model")
    p_probe.add_argument("--timeout", type=float, default=90.0,
                         help="per-model timeout in seconds (default 90)")
    p_probe.set_defaults(func=cmd_probe)

    p_ask = add("ask", "send one prompt to one model, direct or through opencode")
    p_ask.add_argument("model", help="model #, full ID (provider/model), or unique substring")
    p_ask.add_argument("prompt", nargs="*", help="prompt text (reads stdin when omitted)")
    p_ask.add_argument("--via", choices=["auto", "direct", "opencode"], default="auto",
                       help="transport (default: direct when a key+endpoint exist)")
    p_ask.add_argument("--system", help="optional system prompt")
    p_ask.add_argument("--max-tokens", type=int, default=512)
    p_ask.add_argument("--temperature", type=float, default=0.7)
    p_ask.add_argument("--timeout", type=float, default=180.0)
    p_ask.set_defaults(func=cmd_ask)

    p_show = add("show", "details for one model")
    p_show.add_argument("model", help="model #, ID, or unique substring")
    p_show.set_defaults(func=cmd_show)

    p_use = add("use", "remember a default model")
    p_use.add_argument("model", help="model #, ID, or unique substring")
    p_use.set_defaults(func=cmd_use)

    p_current = add("current", "print the remembered default model")
    p_current.set_defaults(func=cmd_current)

    p_cats = add("categories", "categories with model counts")
    p_cats.set_defaults(func=cmd_categories)

    p_ff = add("extraFuff", "Freebuff's models: access level, plan and criteria for each")
    p_ff.add_argument("model", nargs="?",
                      help="only models matching this name, level or plan")
    p_ff.add_argument("-v", "--verbose", action="store_true",
                      help="print every model's full criteria, not just the table")
    p_ff.add_argument("--refresh", action="store_true",
                      help="re-read Freebuff's own pages instead of using the cache")
    p_ff.set_defaults(func=cmd_extrafuff)

    return parser


def main(argv=None):
    # Behave like a normal Unix filter when the reader goes away (e.g. `| head`).
    if hasattr(signal, "SIGPIPE"):
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"list", "providers", "diff", "probe", "ask", "show", "use",
                "current", "categories", "extraFuff"}

    if argv and argv[0].lower() == "extrafuff":
        argv[0] = "extraFuff"  # the mode has a camelCase name; accept both

    if not argv:
        argv = ["list"]
    elif argv[0] not in commands and not argv[0].startswith("-"):
        # `models4me coder` reads as a listing filter.
        argv = ["list", "-f", argv[0]] + argv[1:]

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        err.print("\n[dim]Interrupted.[/dim]")
        return 130
    except BrokenPipeError:
        # Reader went away (`models4me list | head`): exit clean, no traceback.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())

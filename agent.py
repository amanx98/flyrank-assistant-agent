"""Internship assistant that reads your GitHub repo and Hugging Face dataset.

It can use several FREE online model providers and automatically fails over
to the next one when a provider hits its quota or is overloaded.

.env keys (you need at least ONE model key; add more for more free quota):
  GEMINI_API_KEY      https://aistudio.google.com/apikey
  GROQ_API_KEY        https://console.groq.com/keys
  OPENROUTER_API_KEY  https://openrouter.ai/keys
  NVIDIA_API_KEY      https://build.nvidia.com  (sign in, then "Get API Key")
  GITHUB_TOKEN        optional for public repos (needed for private ones)
  HF_TOKEN            only needed if the dataset is private

Install:  pip install google-genai openai PyGithub datasets huggingface_hub python-dotenv
Run:      python agent.py
"""
import difflib
import inspect
import json
import os
import re
import sys
import time
from itertools import islice

from dotenv import load_dotenv
from github import Auth, Github, GithubException
from datasets import load_dataset
from huggingface_hub import HfApi
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # avoid Windows unicode crashes

# ----------------------------- config ---------------------------------
GEMINI_FIRST = "gemini-3.8-flash"
DEFAULT_REPO = "amanx98/Flyrank-internship-notebook"
DEFAULT_DATASET = "FlyRank/internship-warehouse"
MAX_CHARS = 6000       # cap on any single tool output (free tiers limit tokens/minute)
MAX_TOOL_STEPS = 6     # max tool calls per question on the OpenAI-style providers
MAX_ANSWER_TOKENS = 2500  # cap per reply so free tiers don't reserve huge token budgets
HISTORY_TURNS = 8      # how many past question/answer pairs to keep

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
gh = Github(auth=Auth.Token(GITHUB_TOKEN)) if GITHUB_TOKEN else Github()
hf_api = HfApi(token=HF_TOKEN)


# ----------------------------- helpers --------------------------------
def _truncate(text: str, offset: int = 0) -> str:
    chunk = text[offset : offset + MAX_CHARS]
    remaining = len(text) - (offset + MAX_CHARS)
    if remaining > 0:
        chunk += f"\n\n[... {remaining} more characters. Call again with offset={offset + MAX_CHARS} to continue.]"
    return chunk


def _notebook_to_text(raw: str) -> str:
    nb = json.loads(raw)
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        parts.append(f"[{cell.get('cell_type', 'cell')}]\n{src}")
    return "\n\n".join(parts)


_tree_cache: dict[str, list[str]] = {}


def _repo_paths(repo: str) -> list[str]:
    """All file paths in a repo, fetched once and cached."""
    if repo not in _tree_cache:
        r = gh.get_repo(repo)
        tree = r.get_git_tree(r.default_branch, recursive=True)
        _tree_cache[repo] = [t.path for t in tree.tree if t.type == "blob"]
    return _tree_cache[repo]


def _not_found_hint(repo: str, path: str) -> str:
    """When a path doesn't exist (e.g. a typo), suggest the closest real paths."""
    try:
        paths = _repo_paths(repo)
        close = difflib.get_close_matches(path, paths, n=4, cutoff=0.5)
        if not close:  # try matching on the file name alone
            names = {p.rsplit("/", 1)[-1]: p for p in paths}
            close = [names[n] for n in difflib.get_close_matches(
                path.rsplit("/", 1)[-1], list(names), n=4, cutoff=0.5)]
        if close:
            return "Not found. Did you mean one of these exact paths? " + ", ".join(close)
    except Exception:
        pass
    return "Not found. Call repo_tree to see the exact file paths."


# ------------------------------ tools ---------------------------------
# Models read these docstrings to decide when and how to call each tool.

def repo_tree(repo: str) -> str:
    """List every file path in a GitHub repo (full folder structure). repo is 'owner/name'."""
    try:
        return _truncate("\n".join(_repo_paths(repo)))
    except Exception as e:
        return f"Error: {e}"


def read_repo_file(repo: str, path: str, offset: int) -> str:
    """Read one file from a GitHub repo. repo is 'owner/name'. path is the file path
    (Jupyter notebooks are converted to plain cell text). If path is a folder, its contents
    are listed. Use offset 0 to start; for long files, call again with the offset given in the truncation note."""
    try:
        item = gh.get_repo(repo).get_contents(path)
        if isinstance(item, list):
            return "\n".join(f"{i.path}{'/' if i.type == 'dir' else ''}" for i in item)
        text = item.decoded_content.decode("utf-8", errors="replace")
        if path.endswith(".ipynb"):
            text = _notebook_to_text(text)
        return _truncate(text, offset)
    except GithubException as e:
        if e.status == 404:
            return _not_found_hint(repo, path)
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def list_dataset_files(name: str) -> str:
    """List the files inside a Hugging Face dataset repo. name is 'owner/dataset'.
    Use this first to learn how the dataset is organised (configs, splits, file types)."""
    try:
        return _truncate("\n".join(hf_api.list_repo_files(name, repo_type="dataset")))
    except Exception as e:
        return f"Error: {e}"


def inspect_dataset(name: str, config: str, split: str) -> str:
    """Peek at a Hugging Face dataset: returns the column names and the first 3 rows.
    name is 'owner/dataset'. config is the subset name, or '' for the default.
    split is usually 'train'."""
    try:
        ds = load_dataset(name, config or None, split=split or "train", streaming=True, token=HF_TOKEN)
        rows = list(islice(ds, 3))
        if not rows:
            return "The dataset (or split) is empty."
        return _truncate(f"Columns: {list(rows[0].keys())}\nFirst rows: {rows}")
    except Exception as e:
        return f"Error: {e}. Try list_dataset_files to see how the data is organised."


FUNCS = {f.__name__: f for f in (repo_tree, read_repo_file, list_dataset_files, inspect_dataset)}


def run_tool(name: str, args: dict) -> str:
    """Used by the OpenAI-style providers. Fills in defaults the model may leave out."""
    fn = FUNCS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        args = dict(args)
        if name in ("repo_tree", "read_repo_file"):
            args.setdefault("repo", DEFAULT_REPO)
        else:
            args.setdefault("name", DEFAULT_DATASET)
        if name == "read_repo_file":
            args["offset"] = int(args.get("offset") or 0)
        if name == "inspect_dataset":
            args.setdefault("config", "")
            args.setdefault("split", "train")
        allowed = inspect.signature(fn).parameters
        return fn(**{k: v for k, v in args.items() if k in allowed})
    except Exception as e:
        return f"Error: {e}"


def _spec(fn, props: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": " ".join(fn.__doc__.split()),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_S, _I = {"type": "string"}, {"type": "integer"}
TOOLS_SPEC = [
    _spec(repo_tree, {"repo": _S}, []),
    _spec(read_repo_file, {"repo": _S, "path": _S, "offset": _I}, ["path"]),
    _spec(list_dataset_files, {"name": _S}, []),
    _spec(inspect_dataset, {"name": _S, "config": _S, "split": _S}, []),
]


# ------------------------------ prompt --------------------------------
def _preload_tree() -> str:
    """Put the repo's file list in the system prompt so the agent needn't spend a request on it."""
    try:
        return "\n".join(_repo_paths(DEFAULT_REPO))[:3000]
    except Exception:
        return "(could not load the file list; use the repo_tree tool)"


SYSTEM = f"""You are my internship assistant.
My main GitHub repo is {DEFAULT_REPO}.
My main Hugging Face dataset is {DEFAULT_DATASET}.
When I say "my repo", "the repo", "my data" or "the dataset", use those.
Use your tools instead of guessing about file contents or data. If you don't know how the
dataset is organised, call list_dataset_files first. Each tool call is expensive, so read
only the files you actually need. Be concise and say clearly when a tool returned an error.

Rules:
- I make typos. Never copy a path from my message; use the exact paths from the file list
  below (for example "notboks" means "notebooks").
- Only state facts about my repo or data that you actually read with a tool. If you haven't
  read something, say so and read it. Never invent steps, buttons, links or requirements.
- If a tool says "Not found", use one of its suggested paths instead of retrying the same one.

Files currently in my main repo:
{_preload_tree()}"""


# ----------------------------- providers ------------------------------
BAD_WORDS = ("whisper", "guard", "tts", "embed", "vision", "-vl", "audio", "image",
             "rerank", "reward", "safety", "moderation", "orpheus", "compound", "live",
             "robotics", "computer", "learnlm")

# OpenAI-compatible providers. "prefer" = exact model ids to try first (if your key can see
# them); "keywords" = otherwise pick any listed model whose id contains one of these.
OPENAI_PROVIDERS = {
    "groq": dict(
        env="GROQ_API_KEY", base="https://api.groq.com/openai/v1",
        prefer=["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"],
        keywords=("gpt-oss", "llama-3.3", "llama-3.1", "qwen"),
    ),
    "nvidia": dict(
        env="NVIDIA_API_KEY", base="https://integrate.api.nvidia.com/v1",
        prefer=["meta/llama-3.3-70b-instruct"],
        keywords=("llama-3.3", "gpt-oss", "qwen"),
    ),
    "openrouter": dict(
        env="OPENROUTER_API_KEY", base="https://openrouter.ai/api/v1",
        prefer=[],
        keywords=("gpt-oss", "llama-3.3", "qwen", "gemma", "mistral"),
    ),
}

clients: dict[str, OpenAI] = {}
gemini_client = None
chain: list[tuple[str, str]] = []  # (provider, model) in order of preference


def discover_gemini() -> list[str]:
    found = []
    try:
        for m in gemini_client.models.list():
            name = m.name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(w in name for w in BAD_WORDS):
                found.append(name)
    except Exception:
        pass
    found = sorted(set(found), reverse=True)  # newer version numbers first
    found.sort(key=lambda n: "lite" in n)     # full Flash before Lite
    return [GEMINI_FIRST] + [n for n in found if n != GEMINI_FIRST][:2]


def pick_models(name: str, cfg: dict, client: OpenAI) -> list[str]:
    ids = []
    try:
        for m in client.models.list():
            mid = m.id
            if name == "openrouter":
                if not mid.endswith(":free"):
                    continue
                params = (getattr(m, "model_extra", None) or {}).get("supported_parameters")
                if params is not None and "tools" not in params:
                    continue
            if any(b in mid.lower() for b in BAD_WORDS):
                continue
            ids.append(mid)
    except Exception:
        return list(cfg["prefer"])[:2]  # couldn't list models: try the preferred ones blindly
    chosen = [p for p in cfg["prefer"] if p in ids]
    for kw in cfg["keywords"]:
        chosen += [i for i in ids if kw in i.lower() and i not in chosen]
    return chosen[:3]


def setup_providers() -> None:
    global gemini_client
    gem_models: list[str] = []
    if os.environ.get("GEMINI_API_KEY"):
        gemini_client = genai.Client()
        gem_models = discover_gemini()
    else:
        print("(no GEMINI_API_KEY: Gemini disabled)")

    openai_models: list[tuple[str, str]] = []
    for name, cfg in OPENAI_PROVIDERS.items():
        key = os.environ.get(cfg["env"])
        if not key:
            print(f"(no {cfg['env']}: {name} disabled)")
            continue
        clients[name] = OpenAI(api_key=key, base_url=cfg["base"], timeout=60, max_retries=0)
        openai_models += [(name, m) for m in pick_models(name, cfg, clients[name])]

    # Gemini's first model, then the other providers, then Gemini's extra models.
    if gem_models:
        chain.append(("gemini", gem_models[0]))
    chain.extend(openai_models)
    chain.extend(("gemini", m) for m in gem_models[1:])
    if not chain:
        sys.exit("No usable model keys found. Add at least one key to .env (see the top of this file).")


# ------------------------------ calling -------------------------------
def call_gemini(model: str, history: list[dict], q: str) -> str:
    config = types.GenerateContentConfig(system_instruction=SYSTEM, tools=list(FUNCS.values()))
    past = [
        types.Content(role="user" if h["role"] == "user" else "model", parts=[types.Part(text=h["content"])])
        for h in history
    ]
    chat = gemini_client.chats.create(model=model, config=config, history=past)
    return chat.send_message(q).text or ""


def call_openai(provider: str, model: str, history: list[dict], q: str) -> str:
    client = clients[provider]
    messages = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": q}]
    for _ in range(MAX_TOOL_STEPS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS_SPEC, tool_choice="auto",
            max_tokens=MAX_ANSWER_TOKENS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": run_tool(tc.function.name, args)})
    # Tool budget used up: make the model answer from what it has gathered.
    messages.append({
        "role": "user",
        "content": "Stop calling tools. Using only what you have read so far, answer my question "
                   "now. If something is missing, say exactly what you could not read.",
    })
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS_SPEC, tool_choice="none",
        max_tokens=MAX_ANSWER_TOKENS,
    )
    return resp.choices[0].message.content or "(no answer)"


def cooldown_for(e: Exception) -> float:
    """How long (seconds) to avoid a model after it failed, based on the kind of error."""
    msg = str(e)
    low = msg.lower()
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code in (401, 403):
        return 6 * 3600  # bad or unauthorised key
    if code == 429 or "rate limit" in low or "quota" in low or "resource_exhausted" in low:
        if any(w in low for w in ("perday", "per day", "(rpd)", "(tpd)", "daily")):
            return 6 * 3600  # daily quota used up
        m = re.search(r"(?:try again|retry) in (?:(\d+)m)?\s*([\d.]+)s", low)
        if m:
            return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 3600)
        return 60
    if isinstance(code, int) and code >= 500:
        return 120
    if code is None:
        return 60  # network problem or timeout
    return 1800  # model unavailable, unsupported tool format, etc.


cooldown: dict[tuple[str, str], float] = {}
history: list[dict] = []


def ask(q: str):
    """Returns (answer, 'provider/model') or (None, None) if every model failed."""
    for attempt in range(2):
        for key in chain:
            if cooldown.get(key, 0) > time.time():
                continue
            provider, model = key
            try:
                if provider == "gemini":
                    text = call_gemini(model, history, q)
                else:
                    text = call_openai(provider, model, history, q)
            except Exception as e:
                wait = cooldown_for(e)
                cooldown[key] = time.time() + wait
                reason = " ".join(str(e).split())[:450]
                print(f"({provider}/{model} failed: {reason}. Skipping it for {max(1, round(wait / 60))} min.)")
                continue
            history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": text}])
            del history[: -2 * HISTORY_TURNS]
            return text or "(empty reply)", f"{provider}/{model}"
        # Everything failed. If a short cooldown ends soon, wait once and retry.
        soonest = min(cooldown.get(k, 0) for k in chain) - time.time()
        if attempt == 0 and 0 < soonest <= 65:
            print(f"(all models busy, waiting {soonest:.0f}s...)")
            time.sleep(soonest + 1)
            continue
        break
    return None, None


if __name__ == "__main__":
    setup_providers()
    print("Model order:", " -> ".join(f"{p}/{m}" for p, m in chain))
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        answer, used = ask(q)
        if answer is None:
            print("agent: Every model is out of free quota or unavailable right now. "
                  "Add more provider keys to .env or try again later.\n")
        else:
            print(f"agent [{used}]: {answer}\n")
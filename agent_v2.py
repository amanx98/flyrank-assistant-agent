"""
Internship Assistant v2 — full rebuild.

What it does now:
  * Reads your GitHub repos and Hugging Face datasets (same free tools as before)
  * KEEPS STATE between runs in a local SQLite file: tracked internship listings,
    deadlines, application statuses, notes, and a saved "profile" of your skills
  * Tells you what to do next: ask "what should I do?" and it inspects your real
    projects + dataset + tracker and returns a prioritized action plan
  * Morning briefing mode:  python agent.py digest
  * Multi-provider failover across FREE model tiers (unchanged idea, fixed bugs)

Bug fixes vs v1:
  * Tool functions have REAL default arguments -> Gemini's automatic function
    calling no longer crashes with TypeError on missing args
  * Tool budget raised 6 -> 20 steps, file cap 6000 -> 12000 chars
  * Empty/filtered model replies are treated as failures (with short cooldown),
    not stored as successful answers
  * Cooldowns: 401/403 -> 15 min (not 6h); unknown errors no longer ban models
  * System prompt now forces a plan-first protocol instead of one-shot guessing

.env keys (at least ONE model key; more = more free quota):
  GEMINI_API_KEY      https://aistudio.google.com/apikey
  GROQ_API_KEY        https://console.groq.com/keys
  OPENROUTER_API_KEY  https://openrouter.ai/keys
  NVIDIA_API_KEY      https://build.nvidia.com
  GITHUB_TOKEN        optional for public repos (REQUIRED for my_repos / private)
  HF_TOKEN            only if the dataset is private

Install:  pip install google-genai openai PyGithub datasets huggingface_hub python-dotenv
Run:      python agent.py            (chat)
          python agent.py digest     (one-shot morning briefing, no tools needed)
"""
import argparse
import inspect
import json
import os
import re
import sqlite3
import sys
import time
from datetime import date, timedelta
from itertools import islice

from dotenv import load_dotenv
from github import Auth, Github
from datasets import load_dataset
from huggingface_hub import HfApi
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------- config ---------------------------------
GEMINI_FIRST = "gemini-3.8-flash"
DEFAULT_REPO = "amanx98/Flyrank-internship-notebook"
DEFAULT_DATASET = "FlyRank/internship-warehouse"
MAX_CHARS = 12000      # cap on any single tool output
MAX_TOOL_STEPS = 20    # max tool calls per question on OpenAI-style providers
HISTORY_TURNS = 10     # past question/answer pairs kept in context
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "internship_agent.db")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
gh = Github(auth=Auth.Token(GITHUB_TOKEN)) if GITHUB_TOKEN else Github()
hf_api = HfApi(token=HF_TOKEN)

STATUSES = ("watching", "applied", "interviewing", "offer", "rejected", "accepted")


# ------------------------------ database -------------------------------
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS listings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL, role TEXT NOT NULL, url TEXT DEFAULT '',
            deadline TEXT DEFAULT '', status TEXT DEFAULT 'watching',
            notes TEXT DEFAULT '', added_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS kv(
            key TEXT PRIMARY KEY, value TEXT DEFAULT '');
    """)
    con.commit()
    con.close()


def kv_get(key: str) -> str:
    con = db()
    row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    con.close()
    return row["value"] if row else ""


def kv_set(key: str, value: str) -> str:
    con = db()
    con.execute("INSERT INTO kv(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    con.commit()
    con.close()
    return f"Saved ({len(value)} chars)."


# ----------------------------- helpers --------------------------------
def _truncate(text: str, offset: int = 0) -> str:
    chunk = text[offset : offset + MAX_CHARS]
    remaining = len(text) - (offset + MAX_CHARS)
    if remaining > 0:
        chunk += (f"\n\n[... {remaining} more characters. Call again with "
                  f"offset={offset + MAX_CHARS} to continue.]")
    return chunk


def _notebook_to_text(raw: str) -> str:
    nb = json.loads(raw)
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        parts.append(f"[{cell.get('cell_type', 'cell')}]\n{src}")
    return "\n\n".join(parts)


def _norm_deadline(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return "INVALID"


# ------------------------------ tools ---------------------------------
# NOTE: every parameter has a real default. Gemini executes these functions
# directly (automatic function calling), so defaults must live here, not in
# run_tool. Every function returns a string and never raises.

def repo_tree(repo: str = DEFAULT_REPO) -> str:
    """List every file path in a GitHub repo (full folder structure). repo is 'owner/name'.
    Use this to understand how a project is organised before reading files."""
    try:
        r = gh.get_repo(repo)
        tree = r.get_git_tree(r.default_branch, recursive=True)
        return _truncate("\n".join(t.path for t in tree.tree if t.type == "blob"))
    except Exception as e:
        return f"Error: {e}"


def read_repo_file(repo: str = DEFAULT_REPO, path: str = "", offset: int = 0) -> str:
    """Read one file from a GitHub repo. repo is 'owner/name', defaults to the user's main repo.
    path is the file path (Jupyter notebooks are converted to plain cell text). If path is a
    folder, its contents are listed. For long files, call again with the offset from the truncation note."""
    try:
        item = gh.get_repo(repo).get_contents(path)
        if isinstance(item, list):
            return "\n".join(f"{i.path}{'/' if i.type == 'dir' else ''}" for i in item)
        text = item.decoded_content.decode("utf-8", errors="replace")
        if path.endswith(".ipynb"):
            text = _notebook_to_text(text)
        return _truncate(text, offset)
    except Exception as e:
        return f"Error: {e}"


def my_repos(limit: int = 15) -> str:
    """List the authenticated user's own GitHub repos with language, stars and last update,
    most recently updated first. Use to build an overview of the user's projects.
    Requires GITHUB_TOKEN in .env."""
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN is required for my_repos (add it to .env)."
    try:
        repos = gh.get_user().get_repos(sort="updated")[: int(limit)]
        lines = [f"{r.full_name} | {r.language or '?'} | stars {r.stargazers_count} | "
                 f"updated {r.updated_at:%Y-%m-%d} | {(r.description or '')[:80]}"
                 for r in repos]
        return _truncate("\n".join(lines))
    except Exception as e:
        return f"Error: {e}"


def search_repos(query: str = "", limit: int = 10) -> str:
    """Search GitHub for public repositories matching a query. Use to find example projects
    or libraries relevant to an internship task."""
    try:
        results = gh.search_repositories(query)[: int(limit)]
        lines = [f"{r.full_name} | {r.language or '?'} | stars {r.stargazers_count} | {(r.description or '')[:90]}"
                 for r in results]
        return _truncate("\n".join(lines) or "No results.")
    except Exception as e:
        return f"Error: {e}"


def list_dataset_files(name: str = DEFAULT_DATASET) -> str:
    """List the files inside a Hugging Face dataset repo. name is 'owner/dataset'.
    Use this first to learn how the dataset is organised (configs, splits, file types)."""
    try:
        return _truncate("\n".join(hf_api.list_repo_files(name, repo_type="dataset")))
    except Exception as e:
        return f"Error: {e}"


def inspect_dataset(name: str = DEFAULT_DATASET, config: str = "", split: str = "train") -> str:
    """Peek at a Hugging Face dataset: returns the column names and the first 3 rows.
    name is 'owner/dataset'. config is the subset name, or '' for the default.
    split is usually 'train'."""
    try:
        ds = load_dataset(name, config or None, split=split or "train",
                          streaming=True, token=HF_TOKEN)
        rows = list(islice(ds, 3))
        if not rows:
            return "The dataset (or split) is empty."
        return _truncate(f"Columns: {list(rows[0].keys())}\nFirst rows: {rows}")
    except Exception as e:
        return f"Error: {e}. Try list_dataset_files to see how the data is organised."


def add_listing(company: str = "", role: str = "", url: str = "", deadline: str = "", notes: str = "") -> str:
    """Add an internship listing to the tracker. company and role are required.
    url is the posting link, deadline is 'YYYY-MM-DD' or empty, notes is anything
    worth remembering (contacts, requirements, referral names). Always use this
    when the user mentions a company or role they are interested in."""
    company, role = company.strip(), role.strip()
    if not company or not role:
        return "Error: company and role are required."
    dl = _norm_deadline(deadline)
    if dl == "INVALID":
        return "Error: deadline must be YYYY-MM-DD (or empty)."
    con = db()
    con.execute("INSERT INTO listings(company,role,url,deadline,notes) VALUES(?,?,?,?,?)",
                (company, role, url.strip(), dl, notes.strip()))
    con.commit()
    con.close()
    return f"Tracked: {company} — {role}" + (f" (deadline {dl})" if dl else "")


def update_status(company_or_url: str = "", status: str = "") -> str:
    """Update the status of a tracked listing. company_or_url matches the company name or URL
    (partial match works). status must be one of: watching, applied, interviewing,
    offer, rejected, accepted."""
    s = status.strip().lower()
    if s not in STATUSES:
        return f"Error: status must be one of {', '.join(STATUSES)}."
    con = db()
    row = con.execute("SELECT * FROM listings WHERE company LIKE ? OR url LIKE ? ORDER BY id DESC LIMIT 1",
                      (f"%{company_or_url}%", f"%{company_or_url}%")).fetchone()
    if not row:
        con.close()
        return f"Error: no tracked listing matches '{company_or_url}'. Use list_tracked to see them."
    con.execute("UPDATE listings SET status=? WHERE id=?", (s, row["id"]))
    con.commit()
    con.close()
    return f"Updated: {row['company']} — {row['role']} -> {s}"


def list_tracked(status: str = "") -> str:
    """List all tracked internship listings with status and deadline, most recent first.
    Optionally filter by status (watching, applied, interviewing, offer, rejected, accepted)."""
    con = db()
    if status.strip():
        rows = con.execute("SELECT * FROM listings WHERE status=? ORDER BY id DESC", (status.strip().lower(),)).fetchall()
    else:
        rows = con.execute("SELECT * FROM listings ORDER BY id DESC").fetchall()
    con.close()
    if not rows:
        return "No tracked listings yet."
    return "\n".join(f"[{r['status']:11}] {r['company']} — {r['role']}"
                     + (f" | deadline {r['deadline']}" if r["deadline"] else "")
                     + (f" | {r['url']}" if r["url"] else "") for r in rows)


def deadlines_due(days: int = 14) -> str:
    """List tracked listings with deadlines in the next N days (default 14) that are not
    closed (offer/rejected/accepted). Use at the start of any planning question."""
    cutoff = (date.today() + timedelta(days=int(days))).isoformat()
    con = db()
    rows = con.execute("SELECT * FROM listings WHERE deadline != '' AND deadline <= ? "
                       "AND status NOT IN ('offer','rejected','accepted') ORDER BY deadline", (cutoff,)).fetchall()
    con.close()
    if not rows:
        return f"No deadlines in the next {int(days)} days."
    return "\n".join(f"{r['deadline']} ({(date.fromisoformat(r['deadline']) - date.today()).days}d) "
                     f"[{r['status']}] {r['company']} — {r['role']}" for r in rows)


def save_profile(text: str = "") -> str:
    """Save a summary of the user's skills, projects, and goals (persisted across sessions).
    Call this after analysing the user's repos/dataset so future advice is grounded in it."""
    if not text.strip():
        return "Error: profile text is empty."
    return "Profile saved. " + kv_set("profile", text.strip())


def get_profile() -> str:
    """Read the saved user profile (skills, projects, goals). ALWAYS call this before
    answering 'what should I do', 'help me plan', or any advice question."""
    return kv_get("profile") or "No profile saved yet."


def add_note(text: str = "") -> str:
    """Append a free-form note (persisted). Use for decisions, follow-ups, or things to remember."""
    if not text.strip():
        return "Error: note is empty."
    con = db()
    con.execute("INSERT INTO kv(key,value) VALUES(?, datetime('now') || '  ' || ?)",
                (f"note:{time.time()}", text.strip()))
    con.commit()
    con.close()
    return "Note saved."


def get_notes(limit: int = 20) -> str:
    """Read the most recent notes (default 20)."""
    con = db()
    rows = con.execute("SELECT value FROM kv WHERE key LIKE 'note:%' "
                       "ORDER BY key DESC LIMIT ?", (int(limit),)).fetchall()
    con.close()
    if not rows:
        return "No notes yet."
    return _truncate("\n".join(r["value"] for r in reversed(rows)))


TOOL_FUNCS = [repo_tree, read_repo_file, my_repos, search_repos,
              list_dataset_files, inspect_dataset,
              add_listing, update_status, list_tracked, deadlines_due,
              save_profile, get_profile, add_note, get_notes]
FUNCS = {f.__name__: f for f in TOOL_FUNCS}


def run_tool(name: str, args: dict) -> str:
    """Execute a tool by name. Coerces numeric args and never raises."""
    fn = FUNCS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        params = inspect.signature(fn).parameters
        clean = {}
        for k, v in args.items():
            if k not in params:
                continue
            if params[k].annotation is int:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    v = 0
            clean[k] = v
        return fn(**clean)
    except Exception as e:
        return f"Error: {e}"


def build_spec(fn) -> dict:
    """Build an OpenAI tool spec from a function's real signature + docstring,
    so the OpenAI path and the Gemini path always agree on required args."""
    props, required = {}, []
    for pname, p in inspect.signature(fn).parameters.items():
        props[pname] = {"type": "integer" if p.annotation is int else "string"}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": " ".join((fn.__doc__ or "").split()),
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS_SPEC = [build_spec(f) for f in TOOL_FUNCS]


# ------------------------------ prompt --------------------------------
def _preload_tree() -> str:
    try:
        r = gh.get_repo(DEFAULT_REPO)
        tree = r.get_git_tree(r.default_branch, recursive=True)
        return "\n".join(t.path for t in tree.tree if t.type == "blob")[:3000]
    except Exception:
        return "(could not load the file list; use the repo_tree tool)"


SYSTEM = f"""You are an internship coach assistant with real memory across sessions.

My main GitHub repo is {DEFAULT_REPO} (its file list is at the bottom).
My main Hugging Face dataset is {DEFAULT_DATASET}.
My projects, tracker, notes and profile persist in a database — use the tools, never guess.

How to work:
1. PLAN FIRST. For anything non-trivial, say your plan in one short sentence, then act.
2. Ground every claim about my projects/data in tool results. If a tool errors, say so and
   try a different angle — never invent file contents or dataset rows.
3. When I ask for advice ("what should I do", "help me plan", "am I ready"):
   a) call get_profile, deadlines_due and list_tracked first;
   b) if the profile is missing or thin, inspect my repo/dataset with the tools, then call
      save_profile with a 5-10 line summary of my skills, projects and goals;
   c) give a numbered, prioritized action list. Deadlines first, then highest-impact work.
4. Whenever I mention a company or role I'm interested in, call add_listing immediately.
   When I say I applied / got an interview / was rejected, call update_status immediately.
5. Be concrete and honest. If my portfolio has a gap for the roles I want, say exactly what
   to build or document next. Be concise.

Files currently in my main repo:
{_preload_tree()}"""


# ----------------------------- providers ------------------------------
class FailoverError(Exception):
    """A model replied without usable text; retry on another model soon."""
    def __init__(self, msg: str, cooldown: float = 90):
        super().__init__(msg)
        self.cooldown = cooldown


BAD_WORDS = ("whisper", "guard", "tts", "embed", "vision", "-vl", "audio", "image",
             "rerank", "reward", "safety", "moderation", "orpheus", "compound", "live",
             "robotics", "computer", "learnlm")

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
chain: list[tuple[str, str]] = []


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
    found = sorted(set(found), reverse=True)
    found.sort(key=lambda n: "lite" in n)
    return [GEMINI_FIRST] + [n for n in found if n != GEMINI_FIRST][:2]


def pick_models(name: str, cfg: dict, client: OpenAI) -> list[str]:
    ids = []
    try:
        for m in client.models.list():
            mid = m.id
            if name == "openrouter":
                if not mid.endswith(":free"):
                    continue
                extra = (getattr(m, "model_extra", None) or {})
                params = extra.get("supported_parameters")
                if params is not None and "tools" not in params:
                    continue
            if any(b in mid.lower() for b in BAD_WORDS):
                continue
            ids.append(mid)
    except Exception:
        return list(cfg["prefer"])[:2]
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

    if gem_models:
        chain.append(("gemini", gem_models[0]))
    chain.extend(openai_models)
    chain.extend(("gemini", m) for m in gem_models[1:])
    if not chain:
        sys.exit("No usable model keys found. Add at least one key to .env (see the top of this file).")


# ------------------------------ calling -------------------------------
def call_gemini(model: str, history: list[dict], q: str) -> str:
    config = types.GenerateContentConfig(system_instruction=SYSTEM, tools=TOOL_FUNCS)
    past = [types.Content(role="user" if h["role"] == "user" else "model",
                          parts=[types.Part(text=h["content"])]) for h in history]
    chat = gemini_client.chats.create(model=model, config=config, history=past)
    resp = chat.send_message(q)
    text = (resp.text or "").strip()
    if text:
        return text
    fr = resp.candidates[0].finish_reason if resp.candidates else None
    raise FailoverError(f"empty reply (finish_reason={fr})", cooldown=90)


def call_openai(provider: str, model: str, history: list[dict], q: str) -> str:
    client = clients[provider]
    messages = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": q}]
    for _ in range(MAX_TOOL_STEPS):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              tools=TOOLS_SPEC, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            if text:
                return text
            raise FailoverError("empty reply", cooldown=90)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": run_tool(tc.function.name, args)})
    return ("(Stopped: hit the tool-call budget for one question. "
            "Break it into smaller steps or raise MAX_TOOL_STEPS.)")


def cooldown_for(e: Exception) -> float:
    msg = str(e)
    low = msg.lower()
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code in (401, 403):
        return 900  # bad key or model not visible: 15 min, not 6 hours
    if code == 429 or "rate limit" in low or "quota" in low or "resource_exhausted" in low:
        if any(w in low for w in ("perday", "per day", "(rpd)", "(tpd)", "daily", "exceeded your current quota")):
            return 6 * 3600
        m = re.search(r"(?:try again|retry) in (?:(\d+)m)?\s*([\d.]+)s", low)
        if m:
            return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 3600)
        return 60
    if isinstance(code, int) and code >= 500:
        return 120
    if code is None:
        return 60  # network problem / timeout
    return 900  # model unavailable, unsupported tool format, etc.


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
                text = call_gemini(model, history, q) if provider == "gemini" \
                    else call_openai(provider, model, history, q)
            except FailoverError as e:
                cooldown[key] = time.time() + e.cooldown
                print(f"({provider}/{model}: {e}. Cooling down {e.cooldown:.0f}s.)")
                continue
            except Exception as e:
                wait = cooldown_for(e)
                cooldown[key] = time.time() + wait
                reason = " ".join(str(e).split())[:110]
                print(f"({provider}/{model} failed: {reason}. Skipping it for {max(1, round(wait / 60))} min.)")
                continue
            history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": text}])
            del history[: -2 * HISTORY_TURNS]
            return text, f"{provider}/{model}"
        soonest = min(cooldown.get(k, 0) for k in chain) - time.time()
        if attempt == 0 and 0 < soonest <= 65:
            print(f"(all models busy, waiting {soonest:.0f}s...)")
            time.sleep(soonest + 1)
            continue
        break
    return None, None


# ------------------------------ digest --------------------------------
def digest() -> None:
    """One-shot morning briefing built straight from the local DB (no tools, cheap)."""
    today = date.today().isoformat()
    prompt = f"""Today is {today}. Prepare my morning internship briefing in this exact format:
1. URGENT — deadlines in the next 7 days (from the data below) with days remaining.
2. PIPELINE — one line per non-watching listing and its status.
3. SUGGESTED MOVES — 3-5 numbered, concrete actions for today, grounded in my profile
   and pipeline. If the profile is thin, say what to build/document next instead.
Be brief. Data follows.

MY PROFILE:
{kv_get('profile') or '(none saved yet)'}

DEADLINES (next 14 days):
{deadlines_due(14)}

TRACKED LISTINGS:
{list_tracked()}

RECENT NOTES:
{get_notes(10)}"""
    answer, used = ask(prompt)
    if answer is None:
        print("Every model is out of free quota or unavailable right now. Try again later.")
    else:
        print(f"--- morning briefing [{used}] ---\n{answer}")


# -------------------------------- main ---------------------------------
def chat_loop() -> None:
    print("Model order:", " -> ".join(f"{p}/{m}" for p, m in chain))
    print("Tip: ask 'what should I do?' for a plan, or 'digest' for a briefing. 'exit' to quit.\n")
    while True:
        try:
            q = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "digest":
            digest()
            continue
        answer, used = ask(q)
        if answer is None:
            print("agent: Every model is out of free quota or unavailable right now. "
                  "Add more provider keys to .env or try again later.\n")
        else:
            print(f"agent [{used}]: {answer}\n")


if __name__ == "__main__":
    init_db()
    parser = argparse.ArgumentParser(description="Internship Assistant v2")
    parser.add_argument("mode", nargs="?", default="chat", choices=["chat", "digest"])
    args = parser.parse_args()
    setup_providers()
    if args.mode == "digest":
        digest()
    else:
        chat_loop()

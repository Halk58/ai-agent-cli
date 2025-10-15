#!/bin/bash
# install_ai_agent.sh - Deploy the AI server agent on a Debian 12/13 system
#
# One-file installer: copies the Python agent to /opt/ai-agent, creates a venv,
# and installs a wrapper /usr/local/bin/ai-agent. Safe to re-run for updates.
#
set -euo pipefail

# ========= User-tunable defaults (used by wrapper as fallbacks) =========
DEFAULT_MODEL="gpt-5-mini"          # override per-run: AI_AGENT_DEFAULT_MODEL="gpt-4o" ai-agent ...
DEFAULT_MAX_STEPS=24
DEFAULT_MAX_CMDS_PER_STEP=24
DEFAULT_WEB_MODE="auto"             # auto|on|off

# ===== Pre-flight: require root or sudo =====
if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] Necesitas ser root o tener sudo instalado." >&2
  exit 1
fi

# ===== Lock to avoid concurrent installs =====
LOCK_FILE="/tmp/ai-agent.install.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE" || true
  if ! flock -n 9; then
    echo "[ERROR] Otra instalación está en curso. Aborto."
    exit 1
  fi
fi

# ===== Versioning and self-update =====
SCRIPT_VERSION="1.4.0"
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

# Distro check (warn only)
if [ -r /etc/os-release ]; then
  . /etc/os-release
  if [ "${ID:-}" != "debian" ]; then
    echo "[WARN] Script pensado para Debian; detectado ${ID:-?} ${VERSION_ID:-?}."
  fi
fi

# Auto-update: fetch UPDATE_URL and compare SCRIPT_VERSION
if [ -n "$UPDATE_URL" ] && command -v curl >/dev/null 2>&1; then
  tmp_update_file="$(mktemp)"
  trap 'rm -f "$tmp_update_file"' EXIT
  if curl -fsSL --connect-timeout 5 -m 15 -H 'Cache-Control: no-cache' "$UPDATE_URL" -o "$tmp_update_file"; then
    remote_version="$(grep -Eo 'SCRIPT_VERSION=\"?[0-9]+\.[0-9]+\.[0-9]+\"?' "$tmp_update_file" | head -n1 | grep -Eo '([0-9]+\.){2}[0-9]+')"
    if [ -n "$remote_version" ]; then
      newest="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
      if [ "$newest" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
        echo "[INFO] A newer version ($remote_version) is available. Updating self from $UPDATE_URL ..."
        script_path="$(readlink -f -- "$0")"
        ${SUDO} install -m 755 "$tmp_update_file" "$script_path"
        echo "[INFO] Re-executing the updated installer..."
        exec "$script_path" "$@"
      fi
    else
      echo "[WARN] No pude extraer SCRIPT_VERSION del archivo remoto."
    fi
  else
    echo "[WARN] No se pudo descargar $UPDATE_URL. Continuo con la versión local."
  fi
fi

echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y

echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip

DEST_DIR="/opt/ai-agent"
echo "[INFO] Creating installation directory at $DEST_DIR"
${SUDO} mkdir -p "$DEST_DIR"
${SUDO} mkdir -p /etc/ai-agent
${SUDO} chmod 755 /etc/ai-agent

echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
"""
ai_server_agent.py - Autonomous admin agent for Debian using OpenAI.

Goals:
- Single prompt per task (no pre-prompt visible). Web-search decision is transparent and done once per task.
- Faster: concise outputs, strict command cap, robust token-cap compatibility.
- Resilient: auto-batching with bash -lc when state is needed; tokenized dangerous-command guard.
- Self-healing guidance: detects common failure patterns and injects DIAGNOSTIC_HINTS (no blind mutations).
- ENV snapshot + network check shared with the model to reduce wrong steps.
- Sliding window context (summarize old turns) to keep latency/cost low.

"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from glob import glob
from typing import Any, Dict, List, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Falta la librería 'openai'. Instala con: pip install 'openai>=1.60,<2'", file=sys.stderr)
    sys.exit(1)

# ===== Defaults via environment =====
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
DEFAULT_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto|on|off

# ===== Utilities =====
def run_shell_command(command: str, timeout: int = 900) -> Tuple[int, str, str]:
    """Run a shell command with sane env + timeout; return rc, stdout, stderr."""
    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, env=env
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or f"Timeout after {timeout}s")
    except Exception as e:
        return 1, "", str(e)

SAFE_RM_PATTERNS = {"/var/lib/apt/lists/*"}

def _tokenize(cmd: str) -> List[str]:
    try:
        return shlex.split(cmd, posix=True)
    except Exception:
        return []

def is_really_dangerous(cmd: str) -> bool:
    parts = _tokenize(cmd)
    if not parts:
        return False
    head = parts[0]
    if head == "rm":
        flags = {p for p in parts[1:] if p.startswith("-")}
        targets = [p for p in parts[1:] if not p.startswith("-")]
        has_rf = any(("r" in f and "f" in f) for f in flags)
        if has_rf:
            for t in targets:
                if t in ("/", "/*"):
                    return True
                if t in SAFE_RM_PATTERNS:
                    continue
    if any(p.startswith("mkfs") for p in parts):
        return True
    if head == "dd" and any(x.startswith("of=/dev/") for x in parts[1:]):
        return True
    if head in {"shutdown", "reboot", "halt", "poweroff"}:
        return True
    if ":(){ :|:& };:" in cmd:
        return True
    # Ultra-dangerous wildcards
    if cmd.strip().startswith("chmod -R") and " /" in cmd:
        return True
    return False

def clip(text: str, limit: int = 2500) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head, tail = text[:1200], text[-700:]
    return f"{head}\n...<clipped {len(text)-1900} chars>...\n{tail}"

def build_system_prompt() -> str:
    return (
        "You are a server automation agent on Debian Linux.\n"
        "- Respond ONLY with strict JSON: {\"commands\": [\"...\"], \"explanation\":\"...\", \"finished\": true|false}.\n"
        "- Keep explanation <= 280 chars. Propose the FEWEST commands (<= 5) to achieve the step.\n"
        "- Avoid HTTP fetches (curl/wget) unless WEB_CONTEXT explicitly supports it. Prefer native OS tools.\n"
        "- Each command runs in a fresh shell. If state is needed (variables, set -e), propose ONE multi-line bash -lc script.\n"
        "- Prefer non-interactive flags; avoid destructive ops and interactive prompts."
    )

# ----- Chat Completions with token-cap compatibility + retries -----
def chat_request_with_backoff(client, params_builder, max_attempts=4, first_wait=0.6):
    wait = first_wait
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return params_builder()
        except Exception as exc:
            last_exc = exc
            s = str(exc).lower()
            # retry on rate limits / transient errors
            if any(code in s for code in (" 429", " 500", " 502", " 503", "timeout")):
                time.sleep(wait)
                wait = min(wait * 2, 5.0)
                continue
            break
    raise last_exc

def call_chat_json(messages: List[Dict[str, str]], model: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Chat Completions JSON; cap via max_completion_tokens → max_tokens → none."""
    client = OpenAI()

    def _call(msgs, with_temp: bool, cap_key: str | None):
        params = dict(model=model, messages=msgs, response_format={"type": "json_object"})
        if with_temp:
            # Algunos modelos no aceptan temperature; se reintenta sin él más abajo
            params["temperature"] = 0.0
        if cap_key:
            params[cap_key] = 600  # salida concisa
        def builder():
            return client.chat.completions.create(**params)
        return chat_request_with_backoff(client, builder)

    try:
        resp = _call(messages, with_temp=True, cap_key="max_completion_tokens")
    except Exception as e1:
        try:
            resp = _call(messages, with_temp=True, cap_key="max_tokens")
        except Exception as e2:
            try:
                resp = _call(messages, with_temp=True, cap_key=None)
            except Exception as e3:
                # último recurso: sin temperature
                try:
                    resp = _call(messages, with_temp=False, cap_key=None)
                except Exception:
                    raise RuntimeError(f"OpenAI API call failed: {e1}\nThen: {e2}\nFinally: {e3}") from e3

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}\nRaw content:\n{content}") from e

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    return data, usage_dict

# ----- Responses API one-shot web-search -----
def responses_available() -> bool:
    try:
        _ = OpenAI().responses
        return True
    except Exception:
        return False

def responses_web_autoplan(task: str, model: str, mode: str) -> str:
    """One call: decide+fetch web context. Returns summary or empty."""
    if mode == "off" or not responses_available():
        return ""
    prompt = (
        "Decide if a web search would significantly help execute the following server task.\n"
        "If yes, PERFORM it using the web_search tool and then return ONLY:\n"
        "WEB_CONTEXT:\\n<<=2000 chars concise summary with key URLs and commands>\n"
        "If not needed, return ONLY:\n"
        "NO_WEB: <short reason>\n\n"
        f"TASK:\n{task}\n"
    )
    client = OpenAI()
    try:
        resp = client.responses.create(model=model, input=prompt, tools=[{"type":"web_search"}])
    except Exception:
        return ""
    text = getattr(resp, "output_text", "") or str(resp)
    if not isinstance(text, str):
        try: text = str(text)
        except Exception: return ""
    if "WEB_CONTEXT:" in text:
        return text.split("WEB_CONTEXT:", 1)[1].strip()
    return ""  # NO_WEB or any other response → no web

# ----- ENV snapshot to help the model -----
def read_os_release() -> Dict[str, str]:
    out = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if "=" in line and not line.startswith("#"):
                    k,v=line.split("=",1)
                    out[k]=v.strip().strip('"')
    except Exception:
        pass
    return out

def net_status() -> str:
    # very light check
    rc,_,_ = run_shell_command("getent hosts deb.debian.org >/dev/null 2>&1 || getent hosts 1.1.1.1 >/dev/null 2>&1", timeout=5)
    if rc!=0: return "degraded/no-dns"
    rc,_,_ = run_shell_command("ping -n -c1 -W1 1.1.1.1 >/dev/null 2>&1", timeout=3)
    return "online" if rc==0 else "dns-only"

def env_snapshot() -> str:
    osr = read_os_release()
    root = os.geteuid()==0
    rc_df,out_df,_ = run_shell_command("df -h / | awk 'NR==2{print $4\" free\"}'", timeout=3)
    disk = out_df.strip() if rc_df==0 else "n/a"
    rc_mem,out_mem,_ = run_shell_command("awk '/MemAvailable/{printf \"%.0f MB\", $2/1024}' /proc/meminfo", timeout=3)
    mem = out_mem.strip() if rc_mem==0 else "n/a"
    net = net_status()
    return (
        f"OS={osr.get('PRETTY_NAME','unknown')} CODENAME={osr.get('VERSION_CODENAME','?')} "
        f"ROOT={'yes' if root else 'no'} NET={net} DISK_FREE={disk} MEM_FREE={mem}"
    )

# ----- Diagnostic hints injected when failures are seen -----
def diag_hints_for(cmd: str, rc: int, stderr: str, stdout: str) -> str:
    s = (stderr or "") + "\n" + (stdout or "")
    s_low = s.lower()
    hints = []

    if "dpkg was interrupted" in s or "run 'dpkg --configure -a'" in s_low:
        hints.append("Run: dpkg --configure -a (non-interactive).")
    if "could not get lock" in s_low and ("dpkg" in s_low or "apt" in s_low):
        hints.append("Another APT/dpkg is running. Wait/kill process; then retry apt-get update.")
    if "hash sum mismatch" in s_low:
        hints.append("Try: apt-get clean && apt-get update.")
    if "temporary failure resolving" in s_low or "name resolution" in s_low:
        hints.append("DNS/network issue. Check resolv.conf or wait for connectivity.")
    if "InRelease" in s and "is not signed" in s_low:
        hints.append("Repository not signed; ensure correct keyring and signed-by= in sources.list.d/*.list.")
    if "missing key" in s_low or ("no pubkey" in s_low):
        hints.append("Missing repo key; import key to /etc/apt/trusted.gpg.d or /etc/apt/keyrings and use signed-by= option.")
    if "permission denied" in s_low:
        hints.append("Permission denied; if not root, use sudo or adjust file perms/ownership.")
    if "command not found" in s_low:
        hints.append("Command not found; install the package providing it (apt-get install <pkg>).")
    if "no space left on device" in s_low:
        hints.append("Disk full; free space on / or target fs.")
    if rc==124:
        hints.append("Command timeout; consider splitting, reducing verbosity, or increasing timeout.")

    if not hints:
        return ""
    return "DIAGNOSTIC_HINTS:\n- " + "\n- ".join(hints)

# ----- Sliding window summarization (lightweight heuristic) -----
def maybe_summarize_history(messages: List[Dict[str,str]], max_keep: int = 18) -> None:
    """If the chat gets long, summarize old user/assistant turns into one system note."""
    if len(messages) <= max_keep:
        return
    # Keep: system prompt, latest 12 turns, and collapse the middle
    head = [messages[0]]
    tail = messages[-12:]
    middle = messages[1:-12]
    # Basic compress: keep last assistant tool outputs; drop big stdout/stderr blocks
    summaries = []
    for m in middle:
        role = m.get("role","")
        content = m.get("content","")
        if role in ("user","assistant"):
            # strip massive outputs
            summaries.append(f"{role}: {clip(content, 400)}")
    collapsed = {"role":"system","content":"SUMMARY_OF_PRIOR_CONTEXT:\n" + "\n".join(summaries[:40])}
    messages[:] = head + [collapsed] + tail

# ===== Main =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous server agent: natural language -> shell commands on Debian.")
    parser.add_argument("--task", type=str, help="Initial instruction.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max reasoning cycles per task.")
    parser.add_argument("--max-commands-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP, help="Max commands per step.")
    parser.add_argument("--timeout", type=int, default=900, help="Per-command timeout (sec).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    parser.add_argument("--confirm", action="store_true", help="Ask before executing each command.")
    parser.add_argument("--log-file", type=str, default="", help="Optional log file.")
    parser.add_argument("--web", type=str, default=DEFAULT_WEB_MODE, choices=["auto","on","off"], help="Web search via Responses API: auto|on|off")
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY no está definida. Exporta tu clave o usa 'ai-agent --set-key'.", file=sys.stderr, flush=True)
        sys.exit(1)

    print("[INFO] Agent starting (interactive).", flush=True)

    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    # Provide environment snapshot to the model to reduce missteps
    messages.append({"role":"system","content": "ENV_SNAPSHOT: " + env_snapshot()})

    # Single prompt for the first task
    if args.task:
        first_task = args.task.strip()
    else:
        first_task = input("Enter task (or press Enter to exit): ").strip() if sys.stdin.isatty() else ""
        if not first_task:
            print("Exiting interactive session.", flush=True)
            return

    # One-shot web context
    if args.web in ("on","auto"):
        web_ctx = responses_web_autoplan(first_task, args.model, args.web)
        if web_ctx:
            messages.append({"role": "system", "content": "WEB_CONTEXT:\n" + clip(web_ctx, 2000)})

    total_prompt_tokens = 0
    total_completion_tokens = 0

    last_apt_sig = apt_context_signature()
    failed_commands_count: Dict[str, int] = {}
    command_errors: Dict[str, str] = {}

    pending_task = first_task

    while True:
        user_task = pending_task if pending_task is not None else input("Enter next task (or press Enter to exit): ").strip()
        pending_task = None
        if not user_task:
            print("Exiting interactive session.", flush=True)
            break

        messages.append({"role": "user", "content": user_task})
        finished = False

        for step_num in range(1, args.max_steps + 1):
            apt_sig_before = apt_context_signature()

            # Trim history if needed
            maybe_summarize_history(messages)

            try:
                data, usage = call_chat_json(messages, args.model)
            except Exception as e:
                print(f"Error during API call: {e}", file=sys.stderr, flush=True)
                finished = True
                break

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            # Validate JSON
            if not isinstance(data, dict) or not all(k in data for k in ("commands", "explanation", "finished")):
                print("Model returned an unexpected JSON structure:", data, file=sys.stderr, flush=True)
                finished = True
                break

            commands = data.get("commands", []) or []
            explanation = (data.get("explanation", "") or "")[:500]
            finished_flag = bool(data.get("finished", False))

            if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
                print("'commands' must be a list of strings. Received:", commands, file=sys.stderr, flush=True)
                finished = True
                break

            if len(commands) > args.max_commands_per_step:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Limiting commands from {len(commands)} to {args.max_commands_per_step}", flush=True)
                commands = commands[: args.max_commands_per_step]

            print(f"\n=== Step {step_num} ===", flush=True)
            print(f"AI explanation: {explanation}", flush=True)
            if commands:
                print("Proposed commands:", flush=True)
                for c in commands:
                    print(f"  $ {c}", flush=True)
            else:
                print("No commands proposed.", flush=True)

            # Decide if we should auto-batch into one script (more aggressive)
            combine = False
            if len(commands) > 1:
                joined = "\n".join(commands)
                # heuristics: signs of shared state OR chaining
                kws = [" set -e", "set -o pipefail", "export ", "=", "&&", ";", ". /etc/os-release", "source /etc/os-release", "cd "]
                combine = any(k in joined for k in kws)

            if combine:
                # Safety check
                if any(is_really_dangerous(c) for c in commands):
                    print("Blocked batch due to dangerous line.", flush=True)
                else:
                    script = "\n".join(commands)
                    if args.confirm:
                        ans = (input("Execute batch script? [y/N]: ").strip().lower() if sys.stdin.isatty() else "n")
                        if ans not in {"y", "yes", "s", "si", "sí"}:
                            print("Batch skipped by user.", flush=True)
                            messages.append({"role": "assistant", "content": json.dumps(data)})
                            messages.append({"role": "user", "content": "User declined batch execution. Propose one minimal safe command instead."})
                            continue
                    if args.dry_run:
                        print("[DRY-RUN] bash -lc <<SCRIPT\n" + script + "\nSCRIPT", flush=True)
                        rc, out, err = 0, "", ""
                    else:
                        rc, out, err = run_shell_command(f"bash -lc {shlex.quote(script)}", timeout=args.timeout)

                    if out: print("-- STDOUT --", flush=True); print(out.rstrip(), flush=True)
                    if err: print("-- STDERR --", file=sys.stderr, flush=True); print(err.rstrip(), file=sys.stderr, flush=True)

                    messages.append({"role": "assistant", "content": json.dumps(data)})

                    # If failed, inject diagnostic hints to guide next step
                    if rc != 0:
                        hints = diag_hints_for("BATCH", rc, err, out)
                        if hints:
                            messages.append({"role":"system","content": hints})

                    messages.append({"role": "user", "content": f"BATCH EXECUTION\nReturn code: {rc}\nSTDOUT:\n{clip(out)}\nSTDERR:\n{clip(err)}"})

                    if finished_flag:
                        print("\nTask complete.\nSummary:", flush=True)
                        print(explanation, flush=True)
                        finished = True
                        break
                    continue

            skip_remaining = False
            for idx, cmd in enumerate(commands, start=1):
                previously_failed = cmd in failed_commands_count

                # allow limited retry patterns if APT context changed
                allow_retry = any(cmd.strip().startswith(k) for k in ("apt-get update","apt update","apt-get -f install","apt -f install","dpkg --configure -a"))
                apt_sig_now = apt_context_signature()
                apt_changed = (apt_sig_now != last_apt_sig) or (apt_sig_now != apt_sig_before)

                if previously_failed and not (allow_retry and apt_changed and failed_commands_count[cmd] < 3):
                    prev_err = command_errors.get(cmd, "")
                    print(f"Skipping repeated failing command: {cmd}", flush=True)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": f"The command '{cmd}' already failed with:\n{prev_err}\nPropose an alternative or explain next steps."})
                    skip_remaining = True
                    break

                if is_really_dangerous(cmd):
                    msg = f"Blocked dangerous command: {cmd}"
                    print(msg, flush=True)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "system", "content": "DIAGNOSTIC_HINTS:\n- Proposed command blocked as dangerous. Provide a safer alternative."})
                    skip_remaining = True
                    break

                if args.confirm:
                    ans = (input(f"Execute command {idx}/{len(commands)}? [y/N]: ").strip().lower() if sys.stdin.isatty() else "n")
                    if ans not in {"y","yes","s","si","sí"}:
                        print("Skipped by user.", flush=True)
                        messages.append({"role":"assistant","content":json.dumps(data)})
                        messages.append({"role":"user","content":f"User declined: {cmd}. Propose a different minimal safe command."})
                        continue

                print(f"\nExecuting command {idx}/{len(commands)}: {cmd}", flush=True)
                if args.dry_run:
                    rc, stdout, stderr = 0, "", ""
                else:
                    rc, stdout, stderr = run_shell_command(cmd, timeout=args.timeout)

                if stdout: print("-- STDOUT --", flush=True); print(stdout.rstrip(), flush=True)
                if stderr: print("-- STDERR --", file=sys.stderr, flush=True); print(stderr.rstrip(), file=sys.stderr, flush=True)

                messages.append({"role": "assistant", "content": json.dumps(data)})

                # On failure, inject hints before asking the model again
                if rc != 0:
                    failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                    command_errors[cmd] = (stderr or stdout).strip()
                    hint = diag_hints_for(cmd, rc, stderr, stdout)
                    if hint:
                        messages.append({"role":"system","content": hint})

                messages.append({"role": "user", "content": f"Command: {cmd}\nReturn code: {rc}\nSTDOUT:\n{clip(stdout)}\nSTDERR:\n{clip(stderr)}"})

                new_sig = apt_context_signature()
                if new_sig != last_apt_sig:
                    last_apt_sig = new_sig

            if skip_remaining:
                continue

            if finished_flag:
                print("\nTask complete.\nSummary:", flush=True)
                print(explanation, flush=True)
                finished = True
                break

            time.sleep(0.2)

        if not finished and step_num >= args.max_steps:
            print("Maximum number of steps reached. The task may not be complete.", flush=True)

    # Cost (approx)
    if total_prompt_tokens or total_completion_tokens:
        pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0003},
            "gpt-5-mini": {"input": 0.00025, "output": 0.0020},
        }
        model_key = args.model.lower()
        if model_key in pricing:
            rates = pricing[model_key]
            cost = (total_prompt_tokens * rates["input"] + total_completion_tokens * rates["output"]) / 1000.0
            print(f"\nApproximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.", flush=True)
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)", flush=True)
        else:
            print(f"\nToken usage (chat): prompt={total_prompt_tokens}, completion_tokens={total_completion_tokens}.", flush=True)

# ----- APT context signature -----
def apt_context_signature() -> str:
    paths = ["/etc/apt/sources.list"]
    paths.extend(sorted(glob("/etc/apt/sources.list.d/*.list")))
    paths.extend(sorted(glob("/etc/apt/trusted.gpg.d/*.gpg")))
    sig_parts = []
    for p in paths:
        try:
            st = os.stat(p)
            sig_parts.append(f"{p}:{int(st.st_mtime)}:{st.st_size}")
        except FileNotFoundError:
            sig_parts.append(f"{p}:absent")
        except Exception:
            pass
    return "|".join(sig_parts)

if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# Agente autónomo Debian con OpenAI (v1.4.0)
- Un único prompt de tarea; web-search (Responses) se decide **una vez** y de forma transparente.
- Salidas concisas; cap compatible con modelos nuevos/antiguos.
- **Auto-batching** con `bash -lc` cuando se detecta estado compartido.
- **DIAGNOSTIC_HINTS** ante errores típicos (APT, permisos, DNS, espacio, timeouts).
- **ENV_SNAPSHOT** y chequeo de red para reducir pasos fallidos.
- **Ventana deslizante**: resume historial antiguo para mantener coste/latencia.
- **Guardas** contra comandos destructivos tokenizados.
- Wrapper con clave persistente y `config.env` para defaults.
REPORTDOC
${SUDO} chmod 644 "$DEST_DIR/report.md"

VENV_DIR="$DEST_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    ${SUDO} python3 -m venv "$VENV_DIR"
fi
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir 'openai>=1.60,<2' >/dev/null

WRAPPER="/usr/local/bin/ai-agent"
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<'EOF'
#!/bin/bash
# Wrapper for launching the AI server agent (with API key persistence)
set -euo pipefail

AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
AGENT_SCRIPT="$AGENT_DIR/ai_server_agent.py"
KEY_FILE="/etc/ai-agent/agent.env"
CONF_FILE="/etc/ai-agent/config.env"

print_help() {
  cat <<'HLP'
ai-agent - Ejecuta el agente de automatización para Debian

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en /etc/ai-agent/agent.env
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent -h | --help             # Ayuda del wrapper (no requiere API)

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  /etc/ai-agent/agent.env y, si hay TTY, la pedirá e incluso permitirá guardarla.
- Variables por defecto (puedes persistirlas en /etc/ai-agent/config.env):
  AI_AGENT_DEFAULT_MODEL (por defecto: gpt-5-mini)
  AI_AGENT_DEFAULT_MAX_STEPS (24)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP (24)
  AI_AGENT_WEB_MODE (auto|on|off; por defecto auto)
- Para ver ayuda del agente Python:
  ai-agent --help-agent
HLP
}

mask_key() {
  local key="$1"; local len=${#key}
  if [ "$len" -le 8 ]; then echo "********"; else echo "${key:0:4}********${key: -4}"; fi
}

cmd_set_key() {
  local new="${1:-}"
  if [ -z "$new" ]; then
    if [ -t 0 ]; then read -r -s -p "Introduce tu OPENAI_API_KEY: " new; echo
    else echo "[ERROR] No hay TTY y no se proporcionó clave. Usa: ai-agent --set-key TUCLAVE" >&2; exit 1
    fi
  fi
  install -m 700 -d /etc/ai-agent
  umask 077
  printf 'OPENAI_API_KEY=%s\n' "$new" > "$KEY_FILE"
  echo "[OK] Clave guardada en $KEY_FILE"
}

cmd_clear_key() {
  if [ -f "$KEY_FILE" ]; then rm -f "$KEY_FILE"; echo "[OK] Clave eliminada de $KEY_FILE"
  else echo "[INFO] No hay clave persistida."; fi
}

cmd_show_key() {
  if [ -f "$KEY_FILE" ]; then . "$KEY_FILE"; if [ -n "${OPENAI_API_KEY:-}" ]; then
      echo "OPENAI_API_KEY=$(mask_key "$OPENAI_API_KEY")"; else echo "[WARN] El archivo existe pero no tiene OPENAI_API_KEY."; fi
  else echo "[INFO] No hay clave persistida en $KEY_FILE"; fi
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then print_help; exit 0; fi
if [ "${1:-}" = "--help-agent" ]; then exec "$PYTHON_BIN" "$AGENT_SCRIPT" --help; fi

case "${1:-}" in
  --set-key) shift; cmd_set_key "${1:-}"; exit 0 ;;
  --clear-key) cmd_clear_key; exit 0 ;;
  --show-key) cmd_show_key; exit 0 ;;
esac

# Defaults (can be overridden by config/env)
: "${AI_AGENT_DEFAULT_MODEL:=${AI_AGENT_DEFAULT_MODEL:-gpt-5-mini}}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=${AI_AGENT_DEFAULT_MAX_STEPS:-24}}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:-24}}"
: "${AI_AGENT_WEB_MODE:=${AI_AGENT_WEB_MODE:-auto}}"

# Load persistent config if present (overrides the above)
if [ -r "$CONF_FILE" ]; then . "$CONF_FILE"; fi

# Try to load persistent key if env is empty
if [ -z "${OPENAI_API_KEY:-}" ] && [ -r "$KEY_FILE" ]; then . "$KEY_FILE"; fi

# Prompt for key if still missing
if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
  read -r -s -p "Introduce tu OPENAI_API_KEY: " OPENAI_API_KEY; echo
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    read -r -p "¿Guardar esta clave en $KEY_FILE para usos futuros? [y/N]: " ans
    case "${ans,,}" in
      y|yes|s|si|sí) install -m 700 -d /etc/ai-agent; umask 077; printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" > "$KEY_FILE"; echo "[OK] Clave guardada." ;;
      *) ;;
    esac
  fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[ERROR] OPENAI_API_KEY no está definida. Usa 'ai-agent --set-key' o expórtala." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
if [ "${AI_AGENT_DEBUG:-0}" != "0" ]; then set -x; fi

exec env AI_AGENT_DEFAULT_MODEL="$AI_AGENT_DEFAULT_MODEL" \
        AI_AGENT_DEFAULT_MAX_STEPS="$AI_AGENT_DEFAULT_MAX_STEPS" \
        AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="$AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP" \
        AI_AGENT_WEB_MODE="$AI_AGENT_WEB_MODE" \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
    "$PYTHON_BIN" "$AGENT_SCRIPT" "$@"
EOF
${SUDO} chmod +x "$WRAPPER"

echo "[SUCCESS] Installation complete."
echo ""
if [ -t 0 ]; then
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " WANT_SAVE
  case "${WANT_SAVE,,}" in
    y|yes|s|si|sí)
      read -r -s -p "Introduce tu OPENAI_API_KEY: " K; echo
      if [ -n "${K:-}" ]; then
        ${SUDO} install -m 700 -d /etc/ai-agent
        umask 077
        echo "OPENAI_API_KEY=$K" | ${SUDO} tee /etc/ai-agent/agent.env >/dev/null
        echo "[OK] Clave guardada en /etc/ai-agent/agent.env"
      else
        echo "[INFO] No se ha introducido ninguna clave; puedes usar 'ai-agent --set-key' más tarde."
      fi
      ;;
    *) echo "[INFO] Puedes usar 'ai-agent --set-key' para guardarla más adelante." ;;
  esac
fi

printf "\nListo. Prueba ahora:\n  ai-agent --help-agent\n  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\"\n"

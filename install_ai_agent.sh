#!/bin/bash
# install_ai_agent.sh - Deploy the AI server agent on a Debian 12/13 system
#
# This script installs Python dependencies, copies the agent code to
# /opt/ai-agent, sets up a virtual environment, and creates a wrapper
# executable in /usr/local/bin called `ai-agent` so that you can run the
# agent from anywhere.  It must be run with sufficient privileges to
# install software and create directories under /opt and /usr/local/bin.
#
# Usage:
#   sudo bash install_ai_agent.sh
#
# After running this script you can invoke the agent with:
#   ai-agent --task "update the system"
#
set -euo pipefail

# ========= User-tunable defaults =========
DEFAULT_MODEL="gpt-5-mini"          # Override per run: AI_AGENT_DEFAULT_MODEL="gpt-4o" ai-agent ...
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

# Versioning and update mechanism
SCRIPT_VERSION="1.3.6"
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

# Determine whether sudo is available. Some minimal containers do not have sudo installed.
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# Distro check (warn only)
if [ -r /etc/os-release ]; then
  . /etc/os-release
  if [ "${ID:-}" != "debian" ]; then
    echo "[WARN] Script pensado para Debian; detectado ${ID:-?} ${VERSION_ID:-?}."
  fi
fi

# Auto-update: fetch the script at UPDATE_URL and compare versions
if [ -n "$UPDATE_URL" ]; then
    if command -v curl >/dev/null 2>&1; then
        tmp_update_file="$(mktemp)"
        cleanup_update_tmp() { rm -f "$tmp_update_file"; }
        trap cleanup_update_tmp EXIT
        if curl -fsSL --connect-timeout 5 -m 15 -H 'Cache-Control: no-cache' "$UPDATE_URL" -o "$tmp_update_file"; then
            remote_version="$(grep -Eo 'SCRIPT_VERSION=\"?[0-9]+\.[0-9]+\.[0-9]+\"?' "$tmp_update_file" | head -n1 | grep -Eo '([0-9]+\.){2}[0-9]+')"
            if [ -n "$remote_version" ]; then
                newest_version="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
                if [ "$newest_version" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
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
    else
        echo "[WARN] UPDATE_URL definido pero 'curl' no está disponible; omito auto-actualización."
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
ai_server_agent.py - Automate server tasks on Debian using OpenAI's API

Behavior goals:
- Single prompt: "Enter task..." — the agent decides transparently whether to fetch web context ONCE (auto mode) and proceeds.
- Keep things fast: one Responses call for web decision+fetch; concise outputs; minimal commands.
- Avoid gratuitous network curls; prefer local commands. If web info is needed, use WEB_CONTEXT.
- Hardened: tokenized dangerous command detection, batching, context-aware APT retries, TTY-robust input.
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
    print("La librería 'openai' no está instalada. Instálala con: pip install 'openai>=1.60,<2'", file=sys.stderr)
    sys.exit(1)

# ===== Defaults configurable via environment =====
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
DEFAULT_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto|on|off

# ===== Utilities =====
def run_shell_command(command: str, timeout: int = 900) -> Tuple[int, str, str]:
    """Run a shell command with a timeout and sane environment; return (rc, stdout, stderr)."""
    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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
    if parts[0] == "rm":
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
    if parts[0] == "dd" and any(x.startswith("of=/dev/") for x in parts[1:]):
        return True
    if parts[0] in {"shutdown", "reboot", "halt"}:
        return True
    if ":(){ :|:& };:" in cmd:
        return True
    return False

def clip(text: str, limit: int = 3000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head, tail = text[:1400], text[-800:]
    return f"{head}\n...<clipped {len(text)-2200} chars>...\n{tail}"

def build_system_prompt() -> str:
    return (
        "You are a server automation agent on Debian Linux.\n"
        "- Respond ONLY with strict JSON: {\"commands\": [\"...\"], \"explanation\":\"...\", \"finished\": true|false}.\n"
        "- Keep 'explanation' concise (<= 280 chars). Propose the FEWEST commands needed (<= 5).\n"
        "- Avoid HTTP fetching with curl/wget for general info; rely on provided WEB_CONTEXT when present.\n"
        "- Each command runs in a fresh shell. If you need state, propose ONE multi-line script via `bash -lc`.\n"
        "- Prefer non-interactive flags; avoid destructive ops and interactive prompts.\n"
    )

def call_chat_json(messages: List[Dict[str, str]], model: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Chat Completions en JSON, con compatibilidad de límite de tokens:
       intenta max_completion_tokens → max_tokens → sin límite."""
    client = OpenAI()

    def _call_with_cap(msgs, with_temp: bool, cap_key: str | None):
        params = dict(model=model, messages=msgs, response_format={"type": "json_object"})
        if with_temp:
            params["temperature"] = 0.0
        if cap_key:
            params[cap_key] = 700  # cap conservador para respuestas concisas
        return client.chat.completions.create(**params)

    # 1) Intento con max_completion_tokens (modelos nuevos)
    try:
        resp = _call_with_cap(messages, with_temp=True, cap_key="max_completion_tokens")
    except Exception as e1:
        msg1 = str(e1).lower()
        # 2) Intento con max_tokens (modelos antiguos)
        try:
            # si el error fue “unsupported parameter” para max_completion_tokens, probamos max_tokens
            resp = _call_with_cap(messages, with_temp=True, cap_key="max_tokens")
        except Exception as e2:
            msg2 = str(e2).lower()
            # 3) Último recurso: sin límite explícito
            try:
                resp = _call_with_cap(messages, with_temp=True, cap_key=None)
            except Exception as e3:
                # Reintento sin temperatura por si el modelo no permite temperature
                try:
                    resp = _call_with_cap(messages, with_temp=False, cap_key=None)
                except Exception:
                    raise RuntimeError(
                        f"OpenAI API call failed: {e1}\nThen: {e2}\nFinally: {e3}"
                    ) from e3

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

# ===== Responses API Web Search (ONE call, one-off per task) =====
def responses_available() -> bool:
    try:
        _ = OpenAI().responses
        return True
    except Exception:
        return False

def responses_web_autoplan(task: str, model: str, mode: str) -> str:
    """
    Single Responses call. The model must:
      - If web search helps, USE the web_search tool and return 'WEB_CONTEXT:\\n<summary>'
      - Otherwise return 'NO_WEB: <short reason>'
    Returns just the extracted summary (or empty string).
    """
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
        try:
            text = str(text)
        except Exception:
            return ""
    if "WEB_CONTEXT:" in text:
        return text.split("WEB_CONTEXT:", 1)[1].strip()
    return ""  # NO_WEB or malformed -> no web context

def tty_input(prompt: str) -> str:
    """Read a line; if no TTY, try /dev/tty; if still no TTY, return empty."""
    try:
        if sys.stdin and sys.stdin.isatty():
            return input(prompt)
        with open("/dev/tty", "r", encoding="utf-8", errors="ignore") as tty:
            print(prompt, end="", flush=True)
            return tty.readline().rstrip("\n")
    except Exception:
        return ""

def apt_context_signature() -> str:
    """Signature of APT context; changes when sources/keys change (used for smart retries)."""
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

RETRY_WHITELIST = {"apt-get update", "apt update", "apt-get -f install", "apt -f install", "dpkg --configure -a"}

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

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY no está definida. Exporta tu clave o usa 'ai-agent --set-key'.", file=sys.stderr, flush=True)
        sys.exit(1)

    print("[INFO] Agent starting (interactive).", flush=True)

    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]

    # First task (single prompt)
    if args.task:
        first_task = args.task.strip()
    else:
        first_task = tty_input("Enter task (or press Enter to exit): ").strip()
        if not first_task:
            print("Exiting interactive session.", flush=True)
            return

    # Decide and fetch optional WEB_CONTEXT (transparent for the user)
    if args.web == "on":
        web_ctx = responses_web_autoplan(first_task, args.model, "on")
        if web_ctx:
            messages.append({"role": "system", "content": "WEB_CONTEXT:\n" + clip(web_ctx, 2000)})
    elif args.web == "auto":
        web_ctx = responses_web_autoplan(first_task, args.model, "auto")
        if web_ctx:
            messages.append({"role": "system", "content": "WEB_CONTEXT:\n" + clip(web_ctx, 2000)})
    # args.web == "off" => skip

    total_prompt_tokens = 0
    total_completion_tokens = 0

    last_apt_sig = apt_context_signature()
    failed_commands_count: Dict[str, int] = {}
    command_errors: Dict[str, str] = {}

    # Process first task immediately (no extra prompt)
    pending_task = first_task

    while True:
        if pending_task is not None:
            user_task = pending_task
            pending_task = None
        else:
            user_task = tty_input("Enter next task (or press Enter to exit): ").strip()
            if not user_task:
                print("Exiting interactive session.", flush=True)
                break

        messages.append({"role": "user", "content": user_task})
        finished = False

        for step_num in range(1, args.max_steps + 1):
            apt_sig_before = apt_context_signature()
            try:
                data, usage = call_chat_json(messages, args.model)
            except Exception as e:
                print(f"Error during API call: {e}", file=sys.stderr, flush=True)
                finished = True
                break
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            if not isinstance(data, dict) or not all(k in data for k in ("commands", "explanation", "finished")):
                print("Model returned an unexpected JSON structure:", data, file=sys.stderr, flush=True)
                finished = True
                break
            commands = data.get("commands", []) or []
            explanation = data.get("explanation", "")[:500]
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
                for cmd in commands:
                    print(f"  $ {cmd}", flush=True)
            else:
                print("No commands proposed.", flush=True)

            # Batch if shared state is likely
            batcheable = False
            if len(commands) > 1:
                text = "\n".join(commands)
                hints = ["set -e", "set -o pipefail", ". /etc/os-release", "source /etc/os-release", "export ", "&&", ";"]
                batcheable = any(h in text for h in hints)

            if batcheable:
                if any(is_really_dangerous(c) for c in commands):
                    print("Blocked batch due to dangerous line.", flush=True)
                else:
                    script = "\n".join(commands)
                    if args.confirm:
                        ans = tty_input("Execute batch script? [y/N]: ").strip().lower()
                        if ans not in {"y", "yes", "s", "si", "sí"}:
                            print("Batch skipped by user.", flush=True)
                            # ask for alternative
                            messages.append({"role": "assistant", "content": json.dumps(data)})
                            messages.append({"role": "user", "content": "User declined batch execution. Propose minimal single command instead."})
                            continue
                    if args.dry_run:
                        print("[DRY-RUN] bash -lc <<SCRIPT\n" + script + "\nSCRIPT", flush=True)
                        rc, out, err = 0, "", ""
                    else:
                        rc, out, err = run_shell_command(f"bash -lc {shlex.quote(script)}", timeout=args.timeout)
                    if out:
                        print("-- STDOUT --", flush=True); print(out.rstrip(), flush=True)
                    if err:
                        print("-- STDERR --", file=sys.stderr, flush=True); print(err.rstrip(), file=sys.stderr, flush=True)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": f"BATCH EXECUTION\nReturn code: {rc}\nSTDOUT:\n{clip(out)}\nSTDERR:\n{clip(err)}"})
                    if finished_flag:
                        print("\nTask complete.\nSummary:", flush=True)
                        print(explanation, flush=True)
                        finished = True
                        break
                    continue

            skip_remaining = False
            for idx, cmd in enumerate(commands, start=1):
                allow_retry = any(cmd.strip().startswith(k) for k in RETRY_WHITELIST)
                previously_failed = cmd in failed_commands_count
                apt_sig_now = apt_context_signature()
                apt_changed = (apt_sig_now != last_apt_sig) or (apt_sig_now != apt_sig_before)

                if previously_failed and not (allow_retry and apt_changed and failed_commands_count[cmd] < 3):
                    prev_err = command_errors.get(cmd, "")
                    msg = (
                        f"The command '{cmd}' was already attempted and failed with:\n{prev_err}\n"
                        "Please propose a different command or explain why the task cannot be completed."
                    )
                    print(f"Skipping repeated failing command: {cmd}", flush=True)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": msg})
                    skip_remaining = True
                    break

                if is_really_dangerous(cmd):
                    msg = f"Blocked dangerous command: {cmd}"
                    print(msg, flush=True)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": msg})
                    skip_remaining = True
                    break

                if args.confirm:
                    ans = tty_input(f"Execute command {idx}/{len(commands)}? [y/N]: ").strip().lower()
                    if ans not in {"y","yes","s","si","sí"}:
                        print("Skipped by user.", flush=True)
                        messages.append({"role":"assistant","content":json.dumps(data)})
                        messages.append({"role":"user","content":f"User declined: {cmd}. Propose a different minimal command."})
                        continue

                print(f"\nExecuting command {idx}/{len(commands)}: {cmd}", flush=True)
                if args.dry_run:
                    rc, stdout, stderr = 0, "", ""
                else:
                    rc, stdout, stderr = run_shell_command(cmd, timeout=args.timeout)
                if stdout:
                    print("-- STDOUT --", flush=True); print(stdout.rstrip(), flush=True)
                if stderr:
                    print("-- STDERR --", file=sys.stderr, flush=True); print(stderr.rstrip(), file=sys.stderr, flush=True)
                messages.append({"role": "assistant", "content": json.dumps(data)})
                output_summary = f"Command: {cmd}\nReturn code: {rc}\nSTDOUT:\n{clip(stdout)}\nSTDERR:\n{clip(stderr)}"
                messages.append({"role": "user", "content": output_summary})
                if rc != 0:
                    failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                    command_errors[cmd] = (stderr or stdout).strip()

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

            time.sleep(0.5)  # small pause to avoid hammering

        if not finished and step_num >= args.max_steps:
            print("Maximum number of steps reached. The task may not be complete.", flush=True)

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

if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# Agente autónomo Debian con OpenAI (v1.3.6)
- **Un único prompt de tarea**. La decisión de web-search (Responses API) es transparente y se realiza *una vez* por tarea.
- **Rápido por defecto**: una llamada Responses (decisión+fetch), Chat Completions con límite de salida y explicaciones concisas.
- **Evita curls innecesarios**: si hace falta información de Internet, se incorpora como `WEB_CONTEXT`.
- **Batching** con `bash -lc` cuando hay estado compartido; **guardas** contra comandos peligrosos.
- **APT** con reintentos contextuales (cuando cambian repos/keys).
- **TTY robusto**: si no hay TTY, falla de forma explícita en lugar de quedarse “mudo”.
- **Clave persistente**: `/etc/ai-agent/agent.env` (`--set-key`, `--show-key`, `--clear-key`).
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
  local key="$1"
  local len=${#key}
  if [ "$len" -le 8 ]; then
    echo "********"
  else
    echo "${key:0:4}********${key: -4}"
  fi
}

cmd_set_key() {
  local new="${1:-}"
  if [ -z "$new" ]; then
    if [ -t 0 ]; then
      read -r -s -p "Introduce tu OPENAI_API_KEY: " new
      echo
    else
      echo "[ERROR] No hay TTY y no se proporcionó clave. Usa: ai-agent --set-key TUCLAVE" >&2
      exit 1
    fi
  fi
  install -m 700 -d /etc/ai-agent
  umask 077
  printf 'OPENAI_API_KEY=%s\n' "$new" > "$KEY_FILE"
  echo "[OK] Clave guardada en $KEY_FILE"
}

cmd_clear_key() {
  if [ -f "$KEY_FILE" ]; then
    rm -f "$KEY_FILE"
    echo "[OK] Clave eliminada de $KEY_FILE"
  else
    echo "[INFO] No hay clave persistida."
  fi
}

cmd_show_key() {
  if [ -f "$KEY_FILE" ]; then
    # shellcheck disable=SC1090
    . "$KEY_FILE"
    if [ -n "${OPENAI_API_KEY:-}" ]; then
      echo "OPENAI_API_KEY=$(mask_key "$OPENAI_API_KEY")"
    else
      echo "[WARN] El archivo existe pero no tiene OPENAI_API_KEY."
    fi
  else
    echo "[INFO] No hay clave persistida en $KEY_FILE"
  fi
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  print_help
  exit 0
fi

if [ "${1:-}" = "--help-agent" ]; then
  exec "$PYTHON_BIN" "$AGENT_SCRIPT" --help
fi

case "${1:-}" in
  --set-key)
    shift
    cmd_set_key "${1:-}"
    exit 0
    ;;
  --clear-key)
    cmd_clear_key
    exit 0
    ;;
  --show-key)
    cmd_show_key
    exit 0
    ;;
esac

# Defaults (can be overridden by config/env)
: "${AI_AGENT_DEFAULT_MODEL:=gpt-5-mini}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=24}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=24}"
: "${AI_AGENT_WEB_MODE:=auto}"

# Load persistent config if present (overrides the above)
if [ -r "$CONF_FILE" ]; then
  # shellcheck disable=SC1090
  . "$CONF_FILE"
fi

# Try to load persistent key if env is empty
if [ -z "${OPENAI_API_KEY:-}" ] && [ -r "$KEY_FILE" ]; then
  # shellcheck disable=SC1090
  . "$KEY_FILE"
fi

# If still missing and TTY, prompt
if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
  read -r -s -p "Introduce tu OPENAI_API_KEY: " OPENAI_API_KEY
  echo
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    read -r -p "¿Guardar esta clave en $KEY_FILE para usos futuros? [y/N]: " ans
    case "${ans,,}" in
      y|yes|s|si|sí)
        install -m 700 -d /etc/ai-agent
        umask 077
        printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" > "$KEY_FILE"
        echo "[OK] Clave guardada."
        ;;
      *) ;;
    esac
  fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[ERROR] OPENAI_API_KEY no está definida. Usa 'ai-agent --set-key' o expórtala." >&2
  exit 1
fi

# Ensure unbuffered output from Python
export PYTHONUNBUFFERED=1

# Optional debug
if [ "${AI_AGENT_DEBUG:-0}" != "0" ]; then
  set -x
fi

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
      read -r -s -p "Introduce tu OPENAI_API_KEY: " K
      echo
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

printf "\nListo. Prueba ahora:\n  ai-agent --help-agent\n  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\" \n"

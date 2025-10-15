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
#   OPENAI_API_KEY=<your-api-key> ai-agent --task "update the system"
#
set -euo pipefail

# ========= User-tunable defaults =========
# Default model for the agent if none is specified at runtime.
# You can override at launch with: AI_AGENT_DEFAULT_MODEL="gpt-4o" ai-agent ...
DEFAULT_MODEL="gpt-5-mini"

# Higher defaults for complex tasks
DEFAULT_MAX_STEPS=20
DEFAULT_MAX_CMDS_PER_STEP=20

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
SCRIPT_VERSION="1.2.0"
# Set UPDATE_URL to the address of the latest install script.  Leave empty to disable auto-update.
# Example:
# UPDATE_URL="https://example.com/install_ai_agent.sh"
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

# Determine whether sudo is available.  Some minimal containers do not have sudo installed.
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
        # ensure cleanup
        cleanup_update_tmp() { rm -f "$tmp_update_file"; }
        trap cleanup_update_tmp EXIT
        if curl -fsSL "$UPDATE_URL" -o "$tmp_update_file"; then
            remote_version="$(grep -Eo 'SCRIPT_VERSION=\"[0-9.]+\"' "$tmp_update_file" | head -n1 | cut -d'"' -f2)"
            if [ -n "$remote_version" ]; then
                newest_version="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
                if [ "$newest_version" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
                    echo "[INFO] A newer version ($remote_version) is available at $UPDATE_URL. Updating..."
                    script_path="$(readlink -f -- "$0")"
                    # Atomic replace with correct mode
                    ${SUDO} install -m 755 "$tmp_update_file" "$script_path"
                    echo "[INFO] Executing the updated script..."
                    exec "$script_path" "$@"
                fi
            fi
        else
            echo "[WARN] Unable to check for updates at $UPDATE_URL. Proceeding with current version."
        fi
        # tmp file auto-cleaned by trap
    else
        echo "[WARN] UPDATE_URL definido pero 'curl' no está disponible; omito auto-actualización."
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y

echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip

# Create installation directory
DEST_DIR="/opt/ai-agent"
echo "[INFO] Creating installation directory at $DEST_DIR"
${SUDO} mkdir -p "$DEST_DIR"

# Generate the Python agent script in the destination directory
echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
"""
ai_server_agent.py - Automate server tasks on Debian using OpenAI's API

Generalized hardening to handle complex tasks:
- Clear instruction that each command runs in a fresh shell; batching supported.
- Optional batching: if the model proposes a sequence that relies on shared state,
  the agent can execute them in a single bash session safely.
- Safer dangerous-command detection using tokenization, with a small whitelist for known-safe maintenance ops.
- Context-aware retries: allow reattempting key commands (e.g., 'apt-get update') once context (APT sources/keys) has changed.
- Higher defaults for steps and commands per step.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from glob import glob
from typing import Any, Dict, List, Tuple

try:
    from openai import OpenAI  # modern SDK
except ImportError:
    print("La librería 'openai' no está instalada. Instálala con: pip install 'openai>=1.40,<2'", file=sys.stderr)
    sys.exit(1)


# ===== Defaults configurable via environment =====
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "20"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "20"))


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
        return 124, (e.stdout or ""), (e.stderr or f"Timeout tras {timeout}s")
    except Exception as e:
        return 1, "", str(e)


# --- Dangerous command detection (tokenized) ---
SAFE_RM_PATTERNS = {
    "/var/lib/apt/lists/*",  # apt cache cleanup
}

def _tokenize(cmd: str) -> List[str]:
    try:
        return shlex.split(cmd, posix=True)
    except Exception:
        return []

def is_really_dangerous(cmd: str) -> bool:
    parts = _tokenize(cmd)
    if not parts:
        return False
    # rm -rf / or /* (but allow safe patterns)
    if parts[0] == "rm":
        flags = {p for p in parts[1:] if p.startswith("-")}
        targets = [p for p in parts[1:] if not p.startswith("-")]
        has_rf = any(("r" in f and "f" in f) for f in flags)
        if has_rf:
            # if any target is exactly "/" or "/*", it's dangerous
            for t in targets:
                if t in ("/", "/*"):
                    return True
                # allow common safe targets
                if t in SAFE_RM_PATTERNS:
                    continue
                # dangerous if target starts with "/" and looks like root wipe
                if t.startswith("/") and t.strip() in ("/root/*", "/etc/*") and "*" in t:
                    return True
    # mkfs / dd to block devices
    toks = set(parts)
    if any(tok.startswith("mkfs") for tok in toks):
        return True
    if parts[0] == "dd" and any(x.startswith("of=/dev/") for x in parts[1:]):
        return True
    # shutdown/reboot/halt
    if parts[0] in {"shutdown", "reboot", "halt"}:
        return True
    # fork bomb
    if ":(){ :|:& };:" in cmd:
        return True
    return False


def clip(text: str, limit: int = 9000) -> str:
    """Clip long text to avoid huge prompts."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head, tail = text[:4500], text[-2500:]
    return f"{head}\n...<clipped {len(text)-7000} chars>...\n{tail}"


def build_system_prompt() -> str:
    """System prompt defining JSON schema and behaviour."""
    return (
        "You are a server automation agent on Debian Linux.\n"
        "- IMPORTANT: Each command is executed in a fresh shell. If you need state (variables, set -euo pipefail, sourcing files), "
        "propose a single multi-line script executed with `bash -lc`.\n"
        "- Always avoid destructive operations.\n"
        "Respond ONLY with strict JSON using this schema:\n"
        "{\n"
        '  "commands": ["<cmd1>", "<cmd2>", ...],\n'
        '  "explanation": "<short explanation>",\n'
        '  "finished": true|false\n'
        "}\n"
        "Rules:\n"
        "- Prefer non-interactive flags.\n"
        "- If a command failed previously, propose an alternative or batch steps into one script if they rely on shared state.\n"
        "- When done, set finished=true and summarise in 'explanation'.\n"
        "- No markdown, no text outside the JSON object."
    )


def call_openai_chat(messages: List[Dict[str, str]], model: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call OpenAI chat completions in JSON mode; retry once to repair JSON if needed."""
    client = OpenAI()  # uses OPENAI_API_KEY from env
    def _call(msgs, with_temp=True):
        params = dict(model=model, messages=msgs, response_format={"type": "json_object"})
        if with_temp:
            params["temperature"] = 0.0
        return client.chat.completions.create(**params)

    # First attempt (temperature=0 for determinism)
    try:
        resp = _call(messages, with_temp=True)
    except Exception as exc:
        # Retry without temperature if model rejects it
        try:
            resp = _call(messages, with_temp=False)
        except Exception:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # JSON repair attempt
        repair_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY a valid JSON object that follows exactly the schema "
            '{"commands": ["..."], "explanation": "...", "finished": true|false} without any extra text.'
        )
        repaired_msgs = messages + [{"role": "user", "content": repair_prompt}]
        resp2 = _call(repaired_msgs, with_temp=False)
        content2 = resp2.choices[0].message.content
        try:
            data = json.loads(content2)
            resp = resp2
        except json.JSONDecodeError as e2:
            raise ValueError(f"Model output is not valid JSON after repair attempt:\n{content2}") from e2

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    return data, usage_dict


def log(msg: str, log_file: str = "") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# --- Context tracking for APT ---
def apt_context_signature() -> str:
    paths = ["/etc/apt/sources.list"]
    paths += sorted(glob("/etc/apt/sources.list.d/*.list"))
    paths += sorted(glob("/etc/apt/trusted.gpg.d/*.gpg"))
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


def should_batch(commands: List[str]) -> bool:
    """Heuristic: batch if sequence seems to rely on shared state or uses set -e/var assignments."""
    if len(commands) <= 1:
        return False
    text = "\n".join(commands)
    hints = [
        "set -e", "set -o pipefail", ". /etc/os-release", "source /etc/os-release",
        "export ", "VERSION_ID=", "CODENAME=", "&&", ";",
    ]
    return any(h in text for h in hints)


def run_batch(commands: List[str], timeout: int) -> Tuple[int, str, str]:
    """Run a list of commands in a single bash -lc session if safe."""
    # If any line is clearly dangerous, do not batch
    for line in commands:
        if is_really_dangerous(line):
            return 2, "", f"Refused to batch due to dangerous line: {line}"
    script = "\n".join(commands)
    return run_shell_command(f"bash -lc {shlex.quote(script)}", timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Autonomous server agent: turns natural language into shell commands on Debian.")
    )
    parser.add_argument("--task", type=str, help="Initial instruction for the agent.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"OpenAI model to use (default from env or '{DEFAULT_MODEL}').")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max reasoning cycles per task.")
    parser.add_argument("--max-commands-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP, help="Max commands to execute per step.")
    parser.add_argument("--timeout", type=int, default=900, help="Per-command timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute them.")
    parser.add_argument("--confirm", action="store_true", help="Ask for confirmation before executing each command.")
    parser.add_argument("--log-file", type=str, default="", help="Optional log file path.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY no está definida. Exporta tu clave antes de ejecutar.", file=sys.stderr)
        sys.exit(1)

    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": build_system_prompt()})

    total_prompt_tokens = 0
    total_completion_tokens = 0

    from typing import Optional
    pending_task: Optional[str] = args.task if args.task else None

    # Track APT context to allow retries of apt-get update when it actually changed
    last_apt_sig = apt_context_signature()
    failed_commands_count: Dict[str, int] = {}
    command_errors: Dict[str, str] = {}

    while True:
        # Get next task
        if pending_task is not None:
            user_task = pending_task.strip()
            pending_task = None
        else:
            try:
                user_task = input("Enter next task (or press Enter to exit): ").strip()
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            if not user_task or user_task.lower() in {"exit", "quit", "q"}:
                print("Exiting interactive session.")
                break
        if not user_task:
            print("No task provided. Skipping.")
            continue

        messages.append({"role": "user", "content": user_task})
        finished = False

        for step_num in range(1, args.max_steps + 1):
            apt_sig_before = apt_context_signature()
            try:
                data, usage = call_openai_chat(messages, args.model)
            except Exception as e:
                print(f"Error during API call: {e}", file=sys.stderr)
                finished = True
                break
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            if not isinstance(data, dict) or not all(k in data for k in ("commands", "explanation", "finished")):
                print("Model returned an unexpected JSON structure:", data, file=sys.stderr)
                finished = True
                break

            commands = data.get("commands", [])
            explanation = data.get("explanation", "")
            finished_flag = data.get("finished", False)

            if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
                print("'commands' must be a list of strings. Received:", commands, file=sys.stderr)
                finished = True
                break

            # Respect command limit
            if len(commands) > args.max_commands_per_step:
                log(f"Limiting commands from {len(commands)} to {args.max_commands_per_step}", args.log_file)
                commands = commands[: args.max_commands_per_step]

            print(f"\n=== Step {step_num} ===")
            print(f"AI explanation: {explanation}")
            if commands:
                print("Proposed commands:")
                for cmd in commands:
                    print(f"  $ {cmd}")
            else:
                print("No commands proposed.")

            # Decide batching
            used_batch = False
            if should_batch(commands):
                # Batch preview – check safety, then run
                if args.confirm:
                    ans = input("Se propone ejecutar los comandos en un único bash -lc. ¿Continuar? [y/N]: ").strip().lower()
                    if ans not in {"y", "yes", "s", "si", "sí"}:
                        print("Saltando ejecución por confirmación del usuario.")
                        # Still feed back the assistant reply to progress conversation
                        messages.append({"role": "assistant", "content": json.dumps(data)})
                        messages.append({"role": "user", "content": "User declined to run the batched script."})
                        continue
                if args.dry_run:
                    print("[dry-run] Ejecutaría en batch con bash -lc:")
                    print("\n".join(commands))
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": "Dry-run batch; no output."})
                    continue
                rc, stdout, stderr = run_batch(commands, timeout=args.timeout)
                used_batch = True
                if stdout:
                    print("-- STDOUT --")
                    print(stdout.rstrip())
                if stderr:
                    print("-- STDERR --")
                    print(stderr.rstrip(), file=sys.stderr)
                messages.append({"role": "assistant", "content": json.dumps(data)})
                output_summary = (
                    f"BATCH EXECUTION\n"
                    f"Return code: {rc}\n"
                    f"STDOUT:\n{clip(stdout)}\n"
                    f"STDERR:\n{clip(stderr)}"
                )
                messages.append({"role": "user", "content": output_summary})
                log(f"batch rc={rc}", args.log_file)
                # Update context tracking
                apt_sig_after = apt_context_signature()
                if rc != 0 and any("apt" in c for c in commands):
                    # Allow a follow-up retry of update if context changed
                    pass
            else:
                skip_remaining = False
                for idx, cmd in enumerate(commands, start=1):
                    # Context-aware retry logic: allow whitelisted commands to retry if APT context changed
                    allow_retry = False
                    for key in RETRY_WHITELIST:
                        if cmd.strip().startswith(key):
                            allow_retry = True
                            break

                    previously_failed = cmd in failed_commands_count
                    apt_sig_now = apt_context_signature()
                    apt_changed = (apt_sig_now != last_apt_sig) or (apt_sig_now != apt_sig_before)

                    if previously_failed and not (allow_retry and apt_changed and failed_commands_count[cmd] < 3):
                        prev_err = command_errors.get(cmd, "")
                        msg = (
                            f"The command '{cmd}' was already attempted and failed with the following error:\n"
                            f"{prev_err}\n"
                            "Please propose a different command or explain why the task cannot be completed."
                        )
                        print(f"Skipping repeated failing command: {cmd}")
                        messages.append({"role": "assistant", "content": json.dumps(data)})
                        messages.append({"role": "user", "content": msg})
                        skip_remaining = True
                        break

                    if is_really_dangerous(cmd):
                        msg = f"Blocked dangerous command: {cmd}"
                        print(msg)
                        messages.append({"role": "assistant", "content": json.dumps(data)})
                        messages.append({"role": "user", "content": msg})
                        skip_remaining = True
                        break

                    # Confirmation / dry-run
                    if args.confirm:
                        ans = input(f"¿Ejecutar el comando {idx}/{len(commands)}? [y/N]: ").strip().lower()
                        if ans not in {"y", "yes", "s", "si", "sí"}:
                            print("Saltando comando por confirmación del usuario.")
                            continue

                    if args.dry_run:
                        print(f"[dry-run] Simulando ejecución: {cmd}")
                        messages.append({"role": "assistant", "content": json.dumps(data)})
                        sim_output = f"Command (dry-run): {cmd}\nReturn code: 0\nSTDOUT:\n\nSTDERR:\n"
                        messages.append({"role": "user", "content": sim_output})
                        continue

                    print(f"\nExecuting command {idx}/{len(commands)}: {cmd}")
                    rc, stdout, stderr = run_shell_command(cmd, timeout=args.timeout)
                    if stdout:
                        print("-- STDOUT --")
                        print(stdout.rstrip())
                    if stderr:
                        print("-- STDERR --")
                        print(stderr.rstrip(), file=sys.stderr)

                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    output_summary = (
                        f"Command: {cmd}\n"
                        f"Return code: {rc}\n"
                        f"STDOUT:\n{clip(stdout)}\n"
                        f"STDERR:\n{clip(stderr)}"
                    )
                    messages.append({"role": "user", "content": output_summary})
                    log(f"cmd='{cmd}' rc={rc}", args.log_file)

                    if rc != 0:
                        failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                        command_errors[cmd] = (stderr or stdout).strip()

                    # Update last APT signature after each command
                    new_sig = apt_context_signature()
                    if new_sig != last_apt_sig:
                        last_apt_sig = new_sig

                if skip_remaining:
                    continue

            # If model signalled completion, break loop
            if isinstance(finished_flag, bool) and finished_flag:
                print("\nTask complete.\nSummary:")
                print(explanation)
                finished = True
                break

            time.sleep(1)

        if not finished and step_num >= args.max_steps:
            print("Maximum number of steps reached. The task may not be complete.")

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
            print(
                f"\nApproximate API usage: prompt_tokens={total_prompt_tokens}, "
                f"completion_tokens={total_completion_tokens}."
            )
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(
                f"\nToken usage: prompt={total_prompt_tokens}, completion={total_completion_tokens}. "
                f"Cost calculation not implemented for model '{args.model}'."
            )


if __name__ == "__main__":
    main()
PYCODE

${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

# Write the report file alongside the agent for reference
echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# Agente autónomo para administrar un servidor Debian con OpenAI (versión endurecida)

Mejoras generales para casos complejos:
- **Batching opcional** cuando los pasos dependen de estado compartido (variables, `set -e`, `source`).
- **Detección de comandos peligrosos tokenizada** con lista blanca para limpiezas seguras (p. ej. `rm -rf /var/lib/apt/lists/*`).
- **Reintentos con contexto**: si cambia la configuración/clave de APT, se permite reintentar comandos clave como `apt-get update` sin quedarnos bloqueados por “repeated failing command”.
- **Límites más altos por defecto** (`--max-steps`, `--max-commands-per-step`) para planes largos.
- **Modelo por defecto configurable** mediante variable de entorno `AI_AGENT_DEFAULT_MODEL` y también definido en wrapper.

Consulta el propio script para más detalles.
REPORTDOC
${SUDO} chmod 644 "$DEST_DIR/report.md"

# Create a Python virtual environment for the agent
VENV_DIR="$DEST_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    ${SUDO} python3 -m venv "$VENV_DIR"
fi

# Install Python dependencies inside the virtual environment (pin version range)
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir 'openai>=1.40,<2' >/dev/null

# Create a wrapper executable in /usr/local/bin
WRAPPER="/usr/local/bin/ai-agent"
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<EOF
#!/bin/bash
# Wrapper for launching the AI server agent

# Ensure the OpenAI API key is provided via environment variable
if [ -z "\${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set." >&2
    echo "Please export your OpenAI API key before running this command." >&2
    exit 1
fi

# Default model and limits (can be overridden by environment)
: "\${AI_AGENT_DEFAULT_MODEL:=${DEFAULT_MODEL}}"
: "\${AI_AGENT_DEFAULT_MAX_STEPS:=${DEFAULT_MAX_STEPS}}"
: "\${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=${DEFAULT_MAX_CMDS_PER_STEP}}"

# Fixed paths for the agent installation
AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
AGENT_SCRIPT="$AGENT_DIR/ai_server_agent.py"

exec env AI_AGENT_DEFAULT_MODEL="\$AI_AGENT_DEFAULT_MODEL" \
        AI_AGENT_DEFAULT_MAX_STEPS="\$AI_AGENT_DEFAULT_MAX_STEPS" \
        AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="\$AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP" \
    "\$PYTHON_BIN" "\$AGENT_SCRIPT" "\$@"
EOF
${SUDO} chmod +x "$WRAPPER"

echo "[SUCCESS] Installation complete."
echo ""
# Prompt the user for an API key and optionally run the agent immediately.
read -r -s -p "Introduce tu clave OpenAI API para ejecutar el agente ahora (se ocultará, deja en blanco para omitir): " API_KEY_INPUT
echo
if [ -n "$API_KEY_INPUT" ]; then
    export OPENAI_API_KEY="$API_KEY_INPUT"
    echo "[INFO] Ejecutando el agente interactivo..."
    # Forward any additional args to the Python program.
    AI_AGENT_DEFAULT_MODEL="$DEFAULT_MODEL" \
    AI_AGENT_DEFAULT_MAX_STEPS="$DEFAULT_MAX_STEPS" \
    AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="$DEFAULT_MAX_CMDS_PER_STEP" \
      "$VENV_DIR/bin/python" "$DEST_DIR/ai_server_agent.py" "$@"
else
    printf "No se ha introducido ninguna clave; la instalación ha finalizado sin ejecutar el agente.\n"
    printf "Para utilizar el agente más adelante:\n  export OPENAI_API_KEY=<tu-clave>\n  AI_AGENT_DEFAULT_MODEL=%s ai-agent --task \"tu instrucción\"\n" "$DEFAULT_MODEL"
fi
printf "\nPuedes volver a ejecutar este script para actualizar el agente, reescribir el informe o introducir una nueva clave de API.\n"

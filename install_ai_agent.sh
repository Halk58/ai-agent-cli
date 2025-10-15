#!/bin/bash
# install_ai_agent.sh - Deploy/Update the AI server agent on Debian 12/13
#
# - Instala dependencias (python/venv/pip)
# - Copia/actualiza el agente en /opt/ai-agent
# - Crea/actualiza un venv con 'openai'
# - Crea/actualiza wrapper /usr/local/bin/ai-agent (con gestión de API persistente)
# - Auto-actualización del propio instalador desde UPDATE_URL
#
# Uso:
#   sudo bash install_ai_agent.sh
#
# Después:
#   ai-agent --task "tu instrucción"
#   ai-agent --help-agent
#
set -euo pipefail

###############################################################################
# Versionado y auto-update del instalador
###############################################################################
SCRIPT_VERSION="1.4.0"
# Usa tu URL de actualización
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

# Detectar sudo (en contenedores minimal puede no existir)
if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

# Auto-update si UPDATE_URL está definido
if [ -n "${UPDATE_URL:-}" ]; then
  tmp_update_file="$(mktemp)"
  if curl -fsSL "$UPDATE_URL" -o "$tmp_update_file"; then
    remote_version="$(grep -Eo 'SCRIPT_VERSION="[0-9.]+"' "$tmp_update_file" | head -n1 | cut -d'"' -f2 || true)"
    if [ -n "$remote_version" ]; then
      newest_version="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
      if [ "$newest_version" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
        echo "[INFO] A newer version ($remote_version) is available. Updating self from $UPDATE_URL ..."
        script_path="$(readlink -f -- "$0")"
        if [ -w "$script_path" ]; then
          cp "$tmp_update_file" "$script_path"
        else
          ${SUDO} cp "$tmp_update_file" "$script_path"
        fi
        chmod +x "$script_path"
        echo "[INFO] Re-executing the updated installer..."
        exec "$script_path" "$@"
      fi
    fi
    rm -f "$tmp_update_file"
  else
    echo "[WARN] Unable to check for updates at $UPDATE_URL. Proceeding with current version."
  fi
fi

###############################################################################
# Rutas y constantes
###############################################################################
DEST_DIR="/opt/ai-agent"
VENV_DIR="$DEST_DIR/venv"
WRAPPER="/usr/local/bin/ai-agent"
CONF_DIR="/etc/ai-agent"
CONF_ENV="$CONF_DIR/agent.env"

# Variables por defecto para el agente (puedes cambiarlas aquí)
AI_AGENT_DEFAULT_MODEL="gpt-5-mini"
AI_AGENT_DEFAULT_MAX_STEPS="24"
AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="24"
AI_AGENT_WEB_MODE="auto"
AI_AGENT_CMD_TIMEOUT="900"           # 15 minutos
AI_AGENT_LOG_FILE=""                 # vacío = sin log persistente

###############################################################################
# Helper: imprime pasos
###############################################################################
ts() { date +"%Y-%m-%d %H:%M:%S"; }

###############################################################################
# Instalar dependencias
###############################################################################
echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y

echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip curl ca-certificates gnupg

###############################################################################
# Crear estructura destino
###############################################################################
echo "[INFO] Creating installation directory at $DEST_DIR"
${SUDO} mkdir -p "$DEST_DIR"
${SUDO} chmod 755 "$DEST_DIR"

${SUDO} mkdir -p "$CONF_DIR"
${SUDO} chmod 750 "$CONF_DIR"

###############################################################################
# Escribir el agente Python
###############################################################################
echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
"""
ai_server_agent.py - Agente autónomo para Debian 12/13 usando OpenAI

Características:
- Timestamps en cada paso/ejecución
- “Scriptificación” automática de bloques complejos (heredocs, multilínea, etc.)
- Sanitización fuerte: bloqueo/rewrite de comandos peligrosos (rm -rf /, curl|bash, apt-key, etc.)
- GPG/DPKG siempre no-interactivos (gpg --batch --yes, DEBIAN_FRONTEND=noninteractive)
- Compat. con modelos que NO soportan temperature=0.0 y/o max_completion_tokens
- Heurística de web-search (auto/on/off) — decisión interna, sin preguntar
- Hints de diagnóstico para errores comunes APT/GPG/TLS
"""

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple, Optional

# -------- Config por defecto (pueden ser sobrescritas por wrapper/env/CLI) --------
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
DEFAULT_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto|on|off
DEFAULT_CMD_TIMEOUT = int(os.getenv("AI_AGENT_CMD_TIMEOUT", "900"))
DEFAULT_LOG_FILE = os.getenv("AI_AGENT_LOG_FILE", "")

# -------- Utilidades de salida/log --------
def now_ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def pinfo(msg: str) -> None:
    print(f"[{now_ts()}] {msg}")

def plog(log_file: Optional[str], text: str) -> None:
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{now_ts()}] {text.rstrip()}\n")
        except Exception:
            pass

# -------- OpenAI API --------
try:
    import openai  # type: ignore
except ImportError:
    print("Falta el paquete 'openai'. Instálalo con: pip install openai", file=sys.stderr)
    sys.exit(1)

def _parse_response(resp) -> Tuple[Dict[str, Any], Dict[str, int]]:
    content = resp.choices[0].message.content
    data = json.loads(content)
    usage = getattr(resp, "usage", None)
    usage_dict = {"prompt_tokens": getattr(usage, "prompt_tokens", 0), "completion_tokens": getattr(usage, "completion_tokens", 0)} if usage else {"prompt_tokens":0,"completion_tokens":0}
    return data, usage_dict

def call_openai_chat(messages: List[Dict[str, str]], model: str, max_out_tokens: int = 2048) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Intenta llamadas en este orden, con backoff y compat.:
    1) temperature=0.0 + max_completion_tokens
    2) (si falla por temperature) sin temperature + max_completion_tokens
    3) (si falla por max_completion_tokens) sin temperature + max_tokens
    4) (fallback final) temperature=1 por defecto del modelo + max_tokens
    """
    last_exc = None
    for attempt in range(1, 5):
        try:
            # 1) temperature=0.0 + max_completion_tokens
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=max_out_tokens,
            )
            return _parse_response(resp)
        except Exception as e1:
            msg = str(e1)
            last_exc = e1
            # 2) si falla por temperature unsupported -> sin temperature
            if "Unsupported value" in msg and "temperature" in msg:
                try:
                    resp = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        max_completion_tokens=max_out_tokens,
                    )
                    return _parse_response(resp)
                except Exception as e2:
                    msg2 = str(e2)
                    last_exc = e2
                    # 3) si también falla por max_completion_tokens -> usar max_tokens
                    if ("Unsupported parameter" in msg2 and "max_completion_tokens" in msg2) or ("unsupported" in msg2.lower() and "max_completion_tokens" in msg2):
                        try:
                            resp = openai.chat.completions.create(
                                model=model,
                                messages=messages,
                                response_format={"type": "json_object"},
                                max_tokens=max_out_tokens,
                            )
                            return _parse_response(resp)
                        except Exception as e3:
                            last_exc = e3
            # 4) fallback total: temperature por defecto + max_tokens
            try:
                resp = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_out_tokens,
                )
                return _parse_response(resp)
            except Exception as e4:
                last_exc = e4
        time.sleep(0.6 * attempt)
    raise RuntimeError(f"OpenAI API call failed: {last_exc}")

# -------- Seguridad y sanitización --------
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\brm\s+-rf\s+/var/lib/apt/lists/\*",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bmkfs(\.| )", r"\bmkfs\b", r"\bdd\s+if=", r"\bdd\s+of=/dev/",
    r":\(\)\s*{\s*:\s*\|\s*:\s*;\s*}\s*;",
    r"\bcurl\b.*\|\s*(bash|sh)\b",
]

def is_dangerous_command(cmd: str) -> bool:
    low = cmd.strip()
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, low):
            return True
    return False

def rewrite_known_risky(cmd: str) -> Tuple[str, List[str]]:
    hints: List[str] = []
    out = cmd

    # rm -rf /var/lib/apt/lists/*  -> apt-get clean
    if re.search(r"\brm\s+-rf\s+/var/lib/apt/lists/\*", out):
        out = re.sub(r"\brm\s+-rf\s+/var/lib/apt/lists/\*", "apt-get clean", out)
        hints.append("Reescrito 'rm -rf /var/lib/apt/lists/*' a 'apt-get clean'.")

    # gpg --dearmor -> forzar --batch --yes
    out = out.replace("gpg --dearmor", "gpg --batch --yes --dearmor")
    out = out.replace("gpg  --dearmor", "gpg --batch --yes --dearmor")

    # apt-key -> keyrings + signed-by
    if "apt-key" in out:
        hints.append("Evita 'apt-key'; usa keyrings en /usr/share/keyrings y 'signed-by=' en la lista APT.")

    # curl ... | bash -> descarga + exec
    if re.search(r"\bcurl\b.*\|\s*(bash|sh)\b", out):
        hints.append("Se desaconseja 'curl | bash'. Descarga a /tmp, verifica y ejecuta.")
        m = re.search(r"curl\s+([^\|]+)\|\s*(bash|sh)\b", out)
        if m:
            repl = (
                "set -euo pipefail; TMP=$(mktemp); "
                + m.group(0).split("|")[0].strip()
                + " -o \"$TMP\"; bash \"$TMP\"; rm -f \"$TMP\""
            )
            out = out.replace(m.group(0), repl)

    # Si se pretende escribir a /etc/apt/sources.list* -> redirigir a /tmp (staging)
    # (El modelo verá la nota y podrá añadir validación/uso)
    out = re.sub(r"(/etc/apt/sources\.list\.d/[^ >]+)", r"/tmp/ai-agent-staging-\1", out)
    if "/tmp/ai-agent-staging-" in out:
        hints.append("Escritura de lista APT redirigida a /tmp (staging). Valida y mueve sólo si 'apt-get update' pasa.")

    return out, hints

def needs_scriptify(cmd: str) -> bool:
    return ("\n" in cmd) or ("<<" in cmd) or (len(cmd) > 600) or bool(re.search(r"\bcase\b|\bwhile\b|\bfor\b.*\bin\b", cmd))

def force_noninteractive_env(env: Dict[str, str]) -> Dict[str, str]:
    e = dict(env)
    e.setdefault("DEBIAN_FRONTEND", "noninteractive")
    e.setdefault("APT_LISTCHANGES_FRONTEND", "none")
    return e

def run_as_bash(cmd: str, timeout: int, env: Optional[Dict[str, str]] = None, log_file: Optional[str] = None) -> Tuple[int, str, str]:
    env2 = os.environ.copy()
    if env:
        env2.update(env)
    env2 = force_noninteractive_env(env2)
    pinfo(f"cmd: {cmd[:1600] + ('…' if len(cmd) > 1600 else '')}")
    plog(log_file, f"EXEC {cmd}")
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env2,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\n[ERROR] TimeoutExpired"

def scriptify_and_run(raw: str, timeout: int, env: Optional[Dict[str, str]] = None, log_file: Optional[str] = None) -> Tuple[int, str, str]:
    env2 = os.environ.copy()
    if env:
        env2.update(env)
    env2 = force_noninteractive_env(env2)

    header = "#!/usr/bin/env bash\nset -Eeuo pipefail\nIFS=$'\\n\\t'\n"
    script_content = header + raw + "\n"
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="ai-agent-", suffix=".sh") as tf:
        tf.write(script_content)
        path = tf.name
    os.chmod(path, 0o700)

    pinfo(f"script: {path}")
    plog(log_file, f"SCRIPT {path}\n{script_content[:4000]}{'…' if len(script_content)>4000 else ''}")

    try:
        proc = subprocess.run(
            ["/usr/bin/env", "bash", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env2,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\n[ERROR] TimeoutExpired"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def run_command_safely(cmd: str, timeout: int, log_file: Optional[str] = None) -> Tuple[int, str, str, List[str]]:
    hints: List[str] = []
    if is_dangerous_command(cmd):
        return 2, "", f"[BLOCKED] Comando peligroso detectado:\n{cmd}", ["Bloqueado por política de seguridad."]
    cmd2, h2 = rewrite_known_risky(cmd)
    hints.extend(h2)
    if needs_scriptify(cmd2):
        rc, out, err = scriptify_and_run(cmd2, timeout, log_file=log_file)
    else:
        rc, out, err = run_as_bash(cmd2, timeout, log_file=log_file)
    return rc, out, err, hints

# -------- Prompt del sistema conciso/estricto --------
def build_system_prompt() -> str:
    return (
        "Eres un agente de automatización en Debian 12/13. Convierte instrucciones en comandos shell.\n"
        "Reglas:\n"
        "- Responde SOLO JSON: {\"commands\": [...], \"explanation\": str, \"finished\": bool}.\n"
        "- Comandos Debian-friendly y NO interactivos.\n"
        "- Evita operaciones destructivas (rm -rf /, mkfs, dd a /dev, shutdown, etc.).\n"
        "- NO uses here-docs; si necesitas escribir ficheros usa 'tee' o 'printf'.\n"
        "- Si un comando falla, no repitas igual: analiza y propón alternativa.\n"
        "- Cuando termines, finished=true y un resumen breve.\n"
    )

# -------- Heurística web (transparente) --------
def decide_web_mode(task: str, cli_web_mode: str) -> str:
    if cli_web_mode in ("on", "off"):
        return cli_web_mode
    t = task.lower()
    hot = any(w in t for w in [
        "última versión", "latest", "precio", "cotización", "cve", "compatibilidad",
        "release notes", "mirror", "repo", "gpg key", "checksum", "bitcoin",
        "kernel", "proxmox", "mariadb", "postgres", "nginx mainline"
    ])
    return "on" if hot else "off"

# -------- Hints de diagnóstico APT/GPG --------
DIAG_PATTERNS = [
    (r"NO_PUBKEY\s+([0-9A-F]+)", "Falta clave GPG. Importa en /usr/share/keyrings y usa 'signed-by=' en la lista APT."),
    (r"Release.*does not have a Release file", "Repositorio sin Release. Revisa URL o usa mirror soportado."),
    (r"Clearsigned.*NOSPLIT", "Firma clearsigned inválida (NOSPLIT). Prueba mirror oficial."),
    (r"certificate.*verify.*failed", "Fallo TLS. Verifica hora del sistema/CA."),
]

def diag_hints_for(stderr: str) -> List[str]:
    hints = []
    for pat, hint in DIAG_PATTERNS:
        if re.search(pat, stderr, re.IGNORECASE):
            hints.append(hint)
    return hints

# -------- CLI principal --------
def main() -> None:
    parser = argparse.ArgumentParser(description="Agente autónomo para Debian que convierte lenguaje natural en comandos.")
    parser.add_argument("--task", type=str, help="Instrucción inicial (si no se indica, se pedirá).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Modelo a usar (por defecto: {DEFAULT_MODEL}).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Máximos ciclos por tarea (def: {DEFAULT_MAX_STEPS}).")
    parser.add_argument("--max-cmds-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP, help=f"Máx. comandos por ciclo (def: {DEFAULT_MAX_CMDS_PER_STEP}).")
    parser.add_argument("--web", type=str, choices=["auto", "on", "off"], default=DEFAULT_WEB_MODE, help=f"Web-search: auto/on/off (def: {DEFAULT_WEB_MODE}).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CMD_TIMEOUT, help=f"Timeout por comando (s) (def: {DEFAULT_CMD_TIMEOUT}).")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE, help="Ruta para log persistente (append).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: falta OPENAI_API_KEY en el entorno.", file=sys.stderr)
        sys.exit(1)
    openai.api_key = api_key

    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    total_prompt_tokens = 0
    total_completion_tokens = 0

    pinfo("Agent starting (interactive).")

    while True:
        if args.task is not None:
            user_task = args.task.strip()
            args.task = None
        else:
            try:
                user_task = input("Enter task (or press Enter to exit): ").strip()
            except KeyboardInterrupt:
                print()
                break
            if not user_task:
                print("Exiting interactive session.")
                break

        web_mode = decide_web_mode(user_task, args.web)
        pinfo(f"web-mode: {web_mode}")

        messages.append({"role": "user", "content": user_task})

        finished = False
        failed_commands_count: Dict[str, int] = {}
        command_errors: Dict[str, str] = {}

        for step in range(1, args.max_steps + 1):
            pinfo(f"=== Step {step} ===")
            try:
                data, usage = call_openai_chat(messages, args.model, max_out_tokens=2048)
            except Exception as e:
                print(f"Error durante llamada API: {e}", file=sys.stderr)
                finished = True
                break

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            if not isinstance(data, dict) or not all(k in data for k in ("commands", "explanation", "finished")):
                print("Estructura JSON inesperada:", data, file=sys.stderr)
                finished = True
                break

            commands = data.get("commands", [])
            explanation = data.get("explanation", "")
            finished_flag = data.get("finished", False)

            if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
                print("'commands' debe ser lista de strings:", commands, file=sys.stderr)
                finished = True
                break

            if len(commands) > args.max_cmds_per_step:
                commands = commands[: args.max_cmds_per_step]
                pinfo(f"Limiting commands to {args.max_cmds_per_step}")

            print(f"AI explanation: {explanation}")
            if commands:
                print("Proposed commands:")
                for c in commands:
                    print("  $", c)

            skip_rest = False
            for idx, cmd in enumerate(commands, start=1):
                if cmd in failed_commands_count:
                    prev_err = command_errors.get(cmd, "")
                    msg = (
                        f"El comando ya falló antes:\n{cmd}\nError previo:\n{prev_err}\n"
                        "Propón alternativa o explica por qué no se puede."
                    )
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": msg})
                    skip_rest = True
                    break

                rc, out, err, hints = run_command_safely(cmd, args.timeout, log_file=args.log_file)

                if out.strip():
                    print("-- STDOUT --")
                    print(out.rstrip())
                if err.strip():
                    print("-- STDERR --", file=sys.stderr)
                    print(err.rstrip(), file=sys.stderr)
                for h in hints + diag_hints_for(err):
                    print(f"[HINT] {h}")

                messages.append({"role": "assistant", "content": json.dumps(data)})
                output_summary = f"Command: {cmd}\nReturn code: {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                messages.append({"role": "user", "content": output_summary})

                if rc != 0:
                    failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                    command_errors[cmd] = (err or out).strip()

            if skip_rest:
                continue

            if isinstance(finished_flag, bool) and finished_flag:
                print("\nTask complete.\nSummary:")
                print(explanation)
                finished = True
                break

            time.sleep(0.6)

        if not finished and step >= args.max_steps:
            print("Maximum number of steps reached. The task may not be complete.")

    if total_prompt_tokens or total_completion_tokens:
        pricing = {
            "gpt-5-mini": {"input": 0.0001, "output": 0.0004},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }
        mk = args.model.lower()
        if mk in pricing:
            r = pricing[mk]
            cost = (total_prompt_tokens * r["input"] + total_completion_tokens * r["output"]) / 1000.0
            print(
                f"\nApproximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}."
            )
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(
                f"\nToken usage (chat): prompt={total_prompt_tokens}, completion={total_completion_tokens}."
                f"  (No pricing table for '{args.model}')"
            )

if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

###############################################################################
# (Opcional) Escribir un pequeño reporte
###############################################################################
echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# AI Server Agent (Debian 12/13)

- Timestamps por paso, ejecución robusta (scriptificación de bloques complejos), sanitización fuerte (bloqueo/rewrite de comandos peligrosos).
- GPG/DPKG no-interactivos, compatibilidad con modelos que no soportan `temperature=0.0` ni `max_completion_tokens`.
- Heurística de web-search (auto/on/off) sin preguntar.
- Hints de diagnóstico para errores comunes (APT/GPG/TLS).
REPORTDOC
${SUDO} chmod 644 "$DEST_DIR/report.md"

###############################################################################
# Virtualenv + dependencias Python
###############################################################################
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  ${SUDO} python3 -m venv "$VENV_DIR"
fi

echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir openai >/dev/null

###############################################################################
# Wrapper con gestión de API persistente y utilidades
###############################################################################
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<'WRAP'
#!/bin/bash
# ai-agent - ejecuta el agente y gestiona la API persistente
set -euo pipefail

CONF_DIR="/etc/ai-agent"
CONF_ENV="$CONF_DIR/agent.env"
AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
AGENT_SCRIPT="$AGENT_DIR/ai_server_agent.py"

# Defaults si no vienen del entorno
: "${AI_AGENT_DEFAULT_MODEL:=gpt-5-mini}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=24}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=24}"
: "${AI_AGENT_WEB_MODE:=auto}"
: "${AI_AGENT_CMD_TIMEOUT:=900}"
: "${AI_AGENT_LOG_FILE:=}"

mask_key() {
  local k="$1"
  local n=${#k}
  if [ "$n" -le 8 ]; then
    echo "********"
  else
    echo "${k:0:4}********${k: -4}"
  fi
}

usage() {
  cat <<USAGE
ai-agent - Ejecuta el agente de automatización para Debian

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en $CONF_ENV
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent --help-agent            # Ayuda del agente Python
  ai-agent -h | --help             # Esta ayuda

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  $CONF_ENV y, si hay TTY, la pedirá e incluso permitirá guardarla.
- Variables por defecto:
  AI_AGENT_DEFAULT_MODEL (por defecto: $AI_AGENT_DEFAULT_MODEL)
  AI_AGENT_DEFAULT_MAX_STEPS ($AI_AGENT_DEFAULT_MAX_STEPS)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP ($AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP)
  AI_AGENT_WEB_MODE ($AI_AGENT_WEB_MODE)
USAGE
}

ensure_confdir() {
  if [ ! -d "$CONF_DIR" ]; then
    sudo mkdir -p "$CONF_DIR" 2>/dev/null || mkdir -p "$CONF_DIR"
    sudo chmod 750 "$CONF_DIR" 2>/dev/null || chmod 750 "$CONF_DIR"
  fi
}

cmd="${1:-}"
if [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
  usage
  exit 0
fi

if [ "$cmd" = "--help-agent" ]; then
  exec "$PYTHON_BIN" "$AGENT_SCRIPT" --help
fi

if [ "$cmd" = "--set-key" ]; then
  ensure_confdir
  shift || true
  if [ -n "${1:-}" ]; then
    KEY="$1"
  else
    if [ -t 0 ]; then
      read -s -p "Introduce tu OPENAI_API_KEY: " KEY
      echo
    else
      echo "Error: no hay TTY para leer la clave y no se pasó como argumento." >&2
      exit 1
    fi
  fi
  printf 'OPENAI_API_KEY=%s\n' "$KEY" | sudo tee "$CONF_ENV" >/dev/null || printf 'OPENAI_API_KEY=%s\n' "$KEY" > "$CONF_ENV"
  sudo chmod 640 "$CONF_ENV" 2>/dev/null || chmod 640 "$CONF_ENV"
  echo "[OK] Clave guardada en $CONF_ENV"
  exit 0
fi

if [ "$cmd" = "--clear-key" ]; then
  if [ -f "$CONF_ENV" ]; then
    sudo rm -f "$CONF_ENV" 2>/dev/null || rm -f "$CONF_ENV"
    echo "[OK] Clave borrada de $CONF_ENV"
  else
    echo "[INFO] No existe $CONF_ENV"
  fi
  exit 0
fi

if [ "$cmd" = "--show-key" ]; then
  if [ -f "$CONF_ENV" ]; then
    # shellcheck disable=SC1090
    . "$CONF_ENV"
    masked="$(mask_key "${OPENAI_API_KEY:-}")"
    echo "OPENAI_API_KEY=$masked"
  else
    echo "[INFO] No existe $CONF_ENV"
  fi
  exit 0
fi

# Cargar clave si no viene del entorno
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$CONF_ENV" ]; then
  # shellcheck disable=SC1090
  . "$CONF_ENV"
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
  read -s -p "Introduce tu OPENAI_API_KEY (se ocultará): " OPENAI_API_KEY
  echo
  if [ -n "$OPENAI_API_KEY" ]; then
    echo -n "¿Quieres guardarla en $CONF_ENV para futuros usos? [y/N]: "
    read -r ans
    if [ "${ans,,}" = "y" ]; then
      ensure_confdir
      printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" | sudo tee "$CONF_ENV" >/dev/null || printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" > "$CONF_ENV"
      sudo chmod 640 "$CONF_ENV" 2>/dev/null || chmod 640 "$CONF_ENV"
      echo "[OK] Clave guardada en $CONF_ENV"
    fi
  fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY no está definida y no se pudo leer/guardar." >&2
  exit 1
fi

# Exportar defaults al entorno si no vienen definidos por el usuario
export AI_AGENT_DEFAULT_MODEL AI_AGENT_DEFAULT_MAX_STEPS AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP AI_AGENT_WEB_MODE AI_AGENT_CMD_TIMEOUT AI_AGENT_LOG_FILE

# Ejecutar el agente (pasa el resto de argumentos tal cual)
exec "$PYTHON_BIN" "$AGENT_SCRIPT" "$@"
WRAP
${SUDO} chmod +x "$WRAPPER"

###############################################################################
# Mensaje final + opción de guardar API key desde el instalador
###############################################################################
echo "[SUCCESS] Installation complete."
echo ""

if [ -t 0 ]; then
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " SAVEKEY
  if [ "${SAVEKEY,,}" = "y" ]; then
    read -s -p "Introduce tu OPENAI_API_KEY: " KEY_IN
    echo
    if [ -n "$KEY_IN" ]; then
      ${SUDO} mkdir -p "$CONF_DIR"
      printf 'OPENAI_API_KEY=%s\n' "$KEY_IN" | ${SUDO} tee "$CONF_ENV" >/dev/null
      ${SUDO} chmod 640 "$CONF_ENV"
      echo "[OK] Clave guardada en $CONF_ENV"
    fi
  fi
fi

echo ""
echo "Listo. Prueba ahora:"
echo "  ai-agent --help-agent"
echo "  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\""

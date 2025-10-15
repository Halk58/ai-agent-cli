#!/usr/bin/env bash
# install_ai_agent.sh - Deploy/update the AI server agent on Debian 12/13
# Uso:
#   sudo bash install_ai_agent.sh
#
# Tras instalar:
#   ai-agent --task "tu instrucción"
#
set -Eeuo pipefail

# =========================
#  Versioning / Auto-update
# =========================
SCRIPT_VERSION="1.5.0"
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

# Auto-update si hay URL y versión más nueva
if [[ -n "${UPDATE_URL}" ]]; then
  _tmp="$(mktemp)"
  if curl -fsSL "${UPDATE_URL}" -o "${_tmp}"; then
    remote_version="$(grep -Eo 'SCRIPT_VERSION="[0-9.]+"' "${_tmp}" | head -n1 | cut -d'"' -f2 || true)"
    if [[ -n "${remote_version}" ]]; then
      newest="$(printf '%s\n%s' "${SCRIPT_VERSION}" "${remote_version}" | sort -V | tail -n1)"
      if [[ "${newest}" = "${remote_version}" && "${SCRIPT_VERSION}" != "${remote_version}" ]]; then
        echo "[INFO] A newer version (${remote_version}) is available. Updating self from ${UPDATE_URL} ..."
        script_path="$(readlink -f -- "$0")"
        if [[ -w "${script_path}" ]]; then
          cp "${_tmp}" "${script_path}"
        else
          ${SUDO} cp "${_tmp}" "${script_path}"
        fi
        chmod +x "${script_path}"
        echo "[INFO] Re-executing the updated installer..."
        exec "${script_path}" "$@"
      fi
    fi
    rm -f "${_tmp}"
  else
    echo "[WARN] Unable to check updates at ${UPDATE_URL}."
  fi
fi

# =========================
#  Dependencias del sistema
# =========================
echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y
echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip ca-certificates curl gnupg

# =========================
#  Directorios / rutas
# =========================
DEST_DIR="/opt/ai-agent"
VENV_DIR="${DEST_DIR}/venv"
WRAPPER="/usr/local/bin/ai-agent"
CONF_DIR="/etc/ai-agent"
CONF_ENV="${CONF_DIR}/agent.env"
CONF_DEFAULTS="${CONF_DIR}/agent.conf"
LOG_DIR="/var/log/ai-agent"

${SUDO} mkdir -p "${DEST_DIR}" "${CONF_DIR}" "${LOG_DIR}"

# =========================
#  Agente Python
# =========================
echo "[INFO] Writing agent script to ${DEST_DIR}/ai_server_agent.py"
${SUDO} tee "${DEST_DIR}/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
"""
ai_server_agent.py - Agente autónomo para administrar Debian 12/13 usando OpenAI

• Timestamps en cada paso y ejecución
• “Scriptificación” automática de bloques complejos (heredocs, multilínea) -> ejecuta siempre con bash
• Sanitización fuerte: GPG/DPKG no-interactivo, bloqueo y reescritura de patrones peligrosos
• Hints de diagnóstico para APT/GPG/TLS
• Heurística web-mode (auto/on/off) transparente, decidida una sola vez por tarea
• Fallback para modelos que no soportan temperature override ni max_completion_tokens
• Límite de comandos por paso y pasos totales configurable
• Log opcional en fichero (--log-file o AI_AGENT_LOG_FILE)

Nota: Si el modelo sugiere modificar APT sources, se recomienda escribir primero en /tmp y validar antes de mover a /etc.
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

# -------- Config por defecto (vía entorno o flags) --------
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
DEFAULT_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto | on | off
DEFAULT_CMD_TIMEOUT = int(os.getenv("AI_AGENT_CMD_TIMEOUT", "900"))  # 15 min
DEFAULT_LOG_FILE = os.getenv("AI_AGENT_LOG_FILE", "")  # si se pasa, apendea logs

# -------- Utilidades de impresión/log --------
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

def _api_call(messages: List[Dict[str, str]], model: str, max_out_tokens: int, use_temp: Optional[float], use_mct: bool):
    """Helper para probar combinaciones de parámetros según soporte del modelo."""
    kwargs: Dict[str, Any] = dict(model=model, messages=messages, response_format={"type": "json_object"})
    if use_temp is not None:
        kwargs["temperature"] = use_temp
    if use_mct:
        kwargs["max_completion_tokens"] = max_out_tokens
    else:
        kwargs["max_tokens"] = max_out_tokens
    return openai.chat.completions.create(**kwargs)

def call_openai_chat(messages: List[Dict[str, str]], model: str, max_out_tokens: int = 2048) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Intenta en este orden:
      1) temperature=0.0 + max_completion_tokens
      2) sin temperature + max_completion_tokens
      3) temperature=0.0 + max_tokens
      4) sin temperature + max_tokens
    Con backoff suave entre intentos.
    """
    attempts = [
        (0.0, True),
        (None, True),
        (0.0, False),
        (None, False),
    ]
    last_exc: Optional[Exception] = None
    for i, (temp, use_mct) in enumerate(attempts, start=1):
        try:
            resp = _api_call(messages, model, max_out_tokens, temp, use_mct)
            content = resp.choices[0].message.content
            data = json.loads(content)
            usage = getattr(resp, "usage", None)
            usage_dict = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens} if usage else {"prompt_tokens":0, "completion_tokens":0}
            return data, usage_dict
        except Exception as e:
            emsg = str(e)
            # Si el fallo es por "unsupported parameter/value" avanzamos al siguiente intento
            if any(k in emsg for k in ["Unsupported parameter", "unsupported_parameter", "Unsupported value", "unsupported value", "does not support"]):
                last_exc = e
                time.sleep(0.5 * i)
                continue
            # Otros errores: reintenta leve, luego abandona
            last_exc = e
            time.sleep(0.4 * i)
    raise RuntimeError(f"OpenAI API call failed: {last_exc}")

# -------- Peligros y sanitización --------
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/(?:\s|$)",  # rm -rf /
    r"\brm\s+-rf\s+/var/lib/apt/lists/\*",  # será reescrito
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bmkfs(\.| )", r"\bmkfs\b", r"\bdd\s+if=", r"\bdd\s+of=/dev/",
    r":\(\)\s*{\s*:\s*\|\s*:\s*;\s*}\s*;",  # fork bomb
    r"\bcurl\b.*\|\s*(bash|sh)\b",  # pipe-to-shell (reescritura)
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
        hints.append("Se reescribió 'rm -rf /var/lib/apt/lists/*' a 'apt-get clean'.")

    # gpg --dearmor -> forzar --batch --yes
    out = out.replace("gpg --dearmor", "gpg --batch --yes --dearmor")
    out = out.replace("gpg  --dearmor", "gpg --batch --yes --dearmor")

    # apt-key (sólo aviso)
    if "apt-key" in out:
        hints.append("Evita 'apt-key'; usa keyrings en /usr/share/keyrings y 'signed-by=' en la lista APT.")

    # curl ... | bash  -> descarga + exec temporal
    if re.search(r"\bcurl\b.*\|\s*(bash|sh)\b", out):
        hints.append("Se desaconseja 'curl | bash'. Se sugiere descargar a /tmp, verificar y ejecutar.")
        m = re.search(r"curl\s+([^\|]+)\|\s*(bash|sh)\b", out)
        if m:
            repl = (
                "TMP=$(mktemp); set -euo pipefail; "
                + m.group(0).split("|")[0].strip()
                + " -o \"$TMP\"; bash \"$TMP\"; rm -f \"$TMP\""
            )
            out = out.replace(m.group(0), repl)

    return out, hints

def needs_scriptify(cmd: str) -> bool:
    return ("\n" in cmd) or ("<<" in cmd) or (len(cmd) > 600) or re.search(r"\bcase\b|\bwhile\b|\bfor\b.*in\b", cmd)

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
    body = raw
    script_content = header + body + "\n"

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

# -------- Prompt del sistema --------
def build_system_prompt() -> str:
    return (
        "Eres un agente de automatización en Debian 12/13. Convierte instrucciones en comandos shell.\n"
        "Reglas:\n"
        "- Responde SOLO JSON {commands:[...], explanation:str, finished:bool}.\n"
        "- Usa comandos Debian-friendly y NO interactivos.\n"
        "- Evita operaciones destructivas (rm -rf /, mkfs, dd a /dev, shutdown...).\n"
        "- No uses here-docs; si debes escribir ficheros usa tee/printf. Para APT sources, escribe primero en /tmp y valida antes de mover a /etc.\n"
        "- Si un comando falla, no repitas igual; analiza y propone alternativa.\n"
        "- Al terminar: finished=true y un resumen corto de lo hecho.\n"
    )

# -------- Heurística web (transparente) --------
def decide_web_mode(task: str, cli_web_mode: str) -> str:
    if cli_web_mode in ("on", "off"):
        return cli_web_mode
    t = task.lower()
    hot = any(w in t for w in [
        "última versión","latest","precio","cotización","cve",
        "compatibilidad","firmware","driver","release notes","mirror","repo","gpg key","checksum",
        "bitcoin","docker tag","image tag","kernel","proxmox","mariadb","postgres","nginx",
    ])
    return "on" if hot else "off"

# -------- DIAG Hints --------
DIAG_PATTERNS = [
    (r"NO_PUBKEY\s+([0-9A-F]+)", "Falta clave GPG: usa keyring en /usr/share/keyrings y 'signed-by=' en la fuente APT."),
    (r"Release.*does not have a Release file", "Repositorio sin Release; revisa URL o usa un mirror soportado."),
    (r"Clearsigned.*NOSPLIT", "Firma clearsigned inválida (NOSPLIT). Prueba mirror.mariadb.org."),
    (r"certificate.*verify.*failed", "Fallo TLS: revisa fecha/hora del sistema o CA."),
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Modelo a usar (def: {DEFAULT_MODEL}).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Máx. ciclos por tarea (def: {DEFAULT_MAX_STEPS}).")
    parser.add_argument("--max-cmds-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP, help=f"Máx. comandos por ciclo (def: {DEFAULT_MAX_CMDS_PER_STEP}).")
    parser.add_argument("--web", type=str, choices=["auto","on","off"], default=DEFAULT_WEB_MODE, help=f"Web-search: auto/on/off (def: {DEFAULT_WEB_MODE}).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CMD_TIMEOUT, help=f"Timeout por comando en segundos (def: {DEFAULT_CMD_TIMEOUT}).")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE, help="Ruta log (append).")
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

            if not isinstance(data, dict) or not all(k in data for k in ("commands","explanation","finished")):
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
            for cmd in commands:
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
            print(f"\nApproximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(f"\nToken usage (chat): prompt={total_prompt_tokens}, completion={total_completion_tokens}.")
            print(f"(No pricing table for '{args.model}')")

if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 0644 "${DEST_DIR}/ai_server_agent.py"

# =========================
#  Report (opcional, breve)
# =========================
echo "[INFO] Writing report to ${DEST_DIR}/report.md"
${SUDO} tee "${DEST_DIR}/report.md" >/dev/null <<'REPORT'
# AI Agent para Debian (resumen)
- Timestamps por paso, ejecución robusta (bash + scriptificación), sanitización fuerte y no-interactivo.
- Hints para errores de APT/GPG/TLS y bloqueo/rewrite de patrones peligrosos.
- Heurística web-mode transparente (auto/on/off) decidida una sola vez por tarea.
- Fallback de parámetros OpenAI (temperature y max_*tokens).
- Config vía variables: AI_AGENT_DEFAULT_MODEL, AI_AGENT_DEFAULT_MAX_STEPS, AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP, AI_AGENT_WEB_MODE, AI_AGENT_CMD_TIMEOUT, AI_AGENT_LOG_FILE.
REPORT
${SUDO} chmod 0644 "${DEST_DIR}/report.md"

# =========================
#  Entorno virtual Python
# =========================
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[INFO] Creating virtual environment..."
  ${SUDO} python3 -m venv "${VENV_DIR}"
fi
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
${SUDO} "${VENV_DIR}/bin/pip" install --no-cache-dir openai >/dev/null

# =========================
#  Defaults persistentes
# =========================
if [[ ! -f "${CONF_DEFAULTS}" ]]; then
  ${SUDO} tee "${CONF_DEFAULTS}" >/dev/null <<'CONF'
# Valores por defecto del agente (se pueden exportar en el entorno)
AI_AGENT_DEFAULT_MODEL="gpt-5-mini"
AI_AGENT_DEFAULT_MAX_STEPS="24"
AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="24"
AI_AGENT_WEB_MODE="auto"   # auto|on|off
AI_AGENT_CMD_TIMEOUT="900" # segundos
AI_AGENT_LOG_FILE=""
CONF
  ${SUDO} chmod 0644 "${CONF_DEFAULTS}"
fi

# =========================
#  Wrapper /usr/local/bin
# =========================
echo "[INFO] Creating wrapper executable ${WRAPPER}"
${SUDO} tee "${WRAPPER}" >/dev/null <<'WRAP'
#!/usr/bin/env bash
set -Eeuo pipefail

# Rutas fijas
AGENT_DIR="/opt/ai-agent"
VENV_DIR="${AGENT_DIR}/venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
AGENT_SCRIPT="${AGENT_DIR}/ai_server_agent.py"
CONF_DIR="/etc/ai-agent"
CONF_ENV="${CONF_DIR}/agent.env"
CONF_DEFAULTS="${CONF_DIR}/agent.conf"

print_help() {
  cat <<'HLP'
ai-agent - Ejecuta el agente de automatización para Debian

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en /etc/ai-agent/agent.env
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent --help-agent            # Ayuda del agente Python
  ai-agent -h | --help             # Esta ayuda

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  /etc/ai-agent/agent.env y, si hay TTY, te permitirá guardarla.
- Variables por defecto (puedes sobreescribir vía entorno o /etc/ai-agent/agent.conf):
  AI_AGENT_DEFAULT_MODEL (por defecto: gpt-5-mini)
  AI_AGENT_DEFAULT_MAX_STEPS (24)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP (24)
  AI_AGENT_WEB_MODE (auto|on|off; por defecto auto)
  AI_AGENT_CMD_TIMEOUT (900)
  AI_AGENT_LOG_FILE (vacío)
HLP
}

mask_key() {
  local k="${1:-}"
  if [[ -z "${k}" ]]; then echo "(no hay clave)"; return; fi
  local len=${#k}
  if (( len <= 8 )); then echo "********"; return; fi
  echo "${k:0:4}********${k: -4}"
}

ensure_defaults() {
  # shellcheck disable=SC1091
  if [[ -f "${CONF_DEFAULTS}" ]]; then source "${CONF_DEFAULTS}"; fi
  export AI_AGENT_DEFAULT_MODEL="${AI_AGENT_DEFAULT_MODEL:-gpt-5-mini}"
  export AI_AGENT_DEFAULT_MAX_STEPS="${AI_AGENT_DEFAULT_MAX_STEPS:-24}"
  export AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:-24}"
  export AI_AGENT_WEB_MODE="${AI_AGENT_WEB_MODE:-auto}"
  export AI_AGENT_CMD_TIMEOUT="${AI_AGENT_CMD_TIMEOUT:-900}"
  export AI_AGENT_LOG_FILE="${AI_AGENT_LOG_FILE:-}"
}

load_env_key() {
  # shellcheck disable=SC1091
  if [[ -f "${CONF_ENV}" ]]; then source "${CONF_ENV}"; fi
}

save_env_key() {
  local key="${1:-}"
  if [[ -z "${key}" ]]; then
    echo "Error: proporciona una clave o usa --set-key sin argumentos para introducirla de forma oculta." >&2
    exit 1
  fi
  mkdir -p "${CONF_DIR}"
  umask 077
  printf 'export OPENAI_API_KEY=%q\n' "${key}" > "${CONF_ENV}"
  chmod 0600 "${CONF_ENV}"
  echo "[OK] Clave guardada en ${CONF_ENV}"
}

interactive_save_key() {
  if [[ -t 0 && -t 1 ]]; then
    read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " ans || true
    if [[ "${ans,,}" == "y" ]]; then
      read -rs -p "Introduce tu OPENAI_API_KEY: " key_input; echo
      if [[ -n "${key_input}" ]]; then
        save_env_key "${key_input}"
      else
        echo "[INFO] No se guardó ninguna clave."
      fi
    fi
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

case "${1:-}" in
  --set-key)
    shift
    if [[ -n "${1:-}" ]]; then
      save_env_key "$1"
    else
      # interactivo oculto
      if [[ -t 0 && -t 1 ]]; then
        read -rs -p "Introduce tu OPENAI_API_KEY: " key_input; echo
        if [[ -n "${key_input}" ]]; then
          save_env_key "${key_input}"
        else
          echo "No se proporcionó clave." >&2
          exit 1
        fi
      else
        echo "Uso: ai-agent --set-key [API_KEY]" >&2
        exit 1
      fi
    fi
    exit 0
    ;;
  --clear-key)
    if [[ -f "${CONF_ENV}" ]]; then rm -f "${CONF_ENV}"; echo "[OK] Borrada ${CONF_ENV}"; else echo "No hay clave guardada."; fi
    exit 0
    ;;
  --show-key)
    load_env_key
    mask_key "${OPENAI_API_KEY:-}"
    exit 0
    ;;
  --help-agent)
    "${PYTHON_BIN}" "${AGENT_SCRIPT}" -h
    exit 0
    ;;
  *)
    # continúa
    ;;
esac

ensure_defaults
load_env_key

# Si no hay clave en entorno ni en env, preguntar si hay TTY
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    read -rs -p "Introduce tu OPENAI_API_KEY (se ocultará, deja en blanco para omitir): " key_input; echo
    if [[ -n "${key_input}" ]]; then
      export OPENAI_API_KEY="${key_input}"
      interactive_save_key
    fi
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY no está definida. Usa 'ai-agent --set-key' o exporta la variable." >&2
  exit 1
fi

# Ejecutar el agente Python, heredando flags del usuario
exec "${PYTHON_BIN}" "${AGENT_SCRIPT}" "$@"
WRAP
${SUDO} chmod +x "${WRAPPER}"

# =========================
#  Prompt para API (opcional)
# =========================
if [[ -t 0 && -t 1 ]]; then
  echo "[SUCCESS] Installation complete."
  echo
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " ans || true
  if [[ "${ans,,}" == "y" ]]; then
    read -rs -p "Introduce tu OPENAI_API_KEY: " API_KEY_INPUT; echo
    if [[ -n "${API_KEY_INPUT}" ]]; then
      umask 077
      ${SUDO} mkdir -p "${CONF_DIR}"
      ${SUDO} bash -c "printf 'export OPENAI_API_KEY=%q\n' '${API_KEY_INPUT}' > '${CONF_ENV}'"
      ${SUDO} chmod 0600 "${CONF_ENV}"
      echo "[OK] Clave guardada en ${CONF_ENV}"
    fi
  fi
  echo
  echo "Listo. Prueba ahora:"
  echo "  ai-agent --help-agent"
  echo "  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\""
else
  echo "[SUCCESS] Installation complete (no TTY; omitiendo prompt de API key)."
fi

#!/usr/bin/env bash
# install_ai_agent.sh - One-shot installer/updater for AI Agent (Debian/Ubuntu)
# Version metadata
VERSION="1.9.0"
PROJECT_NAME="ai-agent"
INSTALL_DIR="/opt/ai-agent"
PY_BIN="/usr/bin/python3"
VENV_DIR="$INSTALL_DIR/.venv"
WRAPPER="/usr/local/bin/ai-agent"
SQL_HELPER="/usr/local/bin/ai-sql"
STATE_DIR="/etc/ai-agent"
ENV_FILE="$STATE_DIR/agent.env"
REPORT_FILE="$INSTALL_DIR/report.md"
AGENT_PY="$INSTALL_DIR/ai_server_agent.py"
UPDATE_URL_DEFAULT="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

set -euo pipefail

log() { printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$*" ; }
info() { log "[INFO] $*"; }
warn() { log "[WARN] $*"; }
err()  { log "[ERROR] $*" >&2; }

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    err "This installer must run as root."
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

compare_versions() {
  # prints 0 if equal, 1 if $1 > $2, 2 if $1 < $2
  # simple dpkg-based compare if present, else fallback lexicographic
  if have_cmd dpkg; then
    if dpkg --compare-versions "$1" eq "$2"; then echo 0; return; fi
    if dpkg --compare-versions "$1" gt "$2"; then echo 1; return; fi
    echo 2; return
  else
    if [ "$1" = "$2" ]; then echo 0; return; fi
    if [ "$1" \> "$2" ]; then echo 1; return; fi
    echo 2; return
  fi
}

self_update() {
  local url="${UPDATE_URL:-$UPDATE_URL_DEFAULT}"
  # allow skipping update via env
  if [ "${NO_SELF_UPDATE:-}" = "1" ]; then
    info "NO_SELF_UPDATE=1, skipping self-update check."
    return 0
  fi
  local tmp="/tmp/${PROJECT_NAME}-installer.$$"
  if ! curl -fsSL "$url" -o "$tmp"; then
    warn "Could not fetch update url: $url (continuing with local script)"
    rm -f "$tmp" || true
    return 0
  fi
  local remote_ver
  remote_ver="$(grep -E '^VERSION=\"' \"$tmp\" | head -n1 | sed -E 's/^VERSION=\"([^\"]+)\".*/\1/')"
  if [ -z "$remote_ver" ]; then
    warn "Remote installer has no VERSION; skipping self-update."
    rm -f "$tmp" || true
    return 0
  fi
  local cmp
  cmp=$(compare_versions "$remote_ver" "$VERSION")
  if [ "$cmp" = "2" ] || [ "$cmp" = "0" ]; then
    # remote <= local
    rm -f "$tmp" || true
    return 0
  fi
  info "A newer version ($remote_ver) is available. Updating self from $url ..."
  chmod +x "$tmp" || true
  info "Re-executing the updated installer..."
  exec "$tmp" "$@"
}

ensure_packages() {
  info "Updating package lists..."
  apt-get update -y >/dev/null || apt-get update -y
  info "Installing Python and required packages..."
  apt-get install -y python3 python3-venv python3-pip curl ca-certificates >/dev/null || apt-get install -y python3 python3-venv python3-pip curl ca-certificates
}

write_report() {
  cat >"$REPORT_FILE" <<'EOF'
# AI Agent (Server Automation)

Esta máquina tiene instalado un agente de automatización por IA.

## Archivos
- Código del agente: `/opt/ai-agent/ai_server_agent.py`
- Entorno virtual: `/opt/ai-agent/.venv`
- Wrapper CLI: `/usr/local/bin/ai-agent`
- Helper SQL: `/usr/local/bin/ai-sql`
- Estado & API key: `/etc/ai-agent/agent.env` (si se guarda)
- Este informe: `/opt/ai-agent/report.md`

## Uso rápido
```bash
ai-agent --help
ai-agent --task "muestra 'uname -a'"
ai-agent --set-key sk-...    # guardar clave API
ai-agent --show-key           # ver clave enmascarada
ai-agent --clear-key          # borrar clave guardada
```

## Notas
- El agente decide automáticamente cuándo usar búsqueda web (si está activado) y limita los comandos por paso con guías de seguridad.
- `AI_AGENT_VERBOSE=1` añade trazas útiles, incluido el RAW devuelto por el modelo.
EOF
}

write_python_agent() {
  mkdir -p "$INSTALL_DIR"
  cat >"$AGENT_PY" <<'PYEOF'
#!/usr/bin/env python3
# ai_server_agent.py - Chat-driven automation agent
# - Verbose logging with timestamps
# - Empty-output recovery
# - No re-injection of assistant JSON per command
# - Auto web-mode decision (off/on/auto) - only heuristic flag (actual web calls via shell/curl when needed)
# - Generic DB access helper awareness via `ai-sql`
import os, sys, json, subprocess, shlex, time, argparse, textwrap, math
from datetime import datetime

# ---------- Config from env ----------
DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "30"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto").lower()  # auto|on|off
VERBOSE = os.getenv("AI_AGENT_VERBOSE", "").lower() not in ("", "0", "false", "no")
COST_PER_1K = float(os.getenv("AI_AGENT_COST_PER_1K_TOKENS", "0.0005"))  # rough default
TIME_FMT = "%Y-%m-%d %H:%M:%S"

# OpenAI
from openai import OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def now():
    return datetime.utcnow().strftime(TIME_FMT)

def log_info(msg):
    print(f"[{now()}] {msg}", flush=True)

def log_err(msg):
    print(f"[{now()}] {msg}", file=sys.stderr, flush=True)

def decide_web_mode(task_text: str) -> str:
    if WEB_MODE in ("on", "off"):
        return WEB_MODE
    # Heurística simple: palabras clave indicativas de web
    kw = ("noticias","news","precio","price","cotización","quote","último valor","latest","descarga","download","http","https","buscar","search")
    t = task_text.lower()
    for k in kw:
        if k in t:
            return "on"
    return "off"

def run_shell(cmd: str, timeout_sec: int = 600):
    """Execute shell command, capture stdout/stderr, rc"""
    try:
        p = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec, executable="/bin/bash")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + f"\nTIMEOUT after {timeout_sec}s"

def safe_command_line(c: str) -> bool:
    bad = ("rm -rf /", ":(){:|:&};:", "mkfs", "dd if=/dev/zero of=", ">: /dev/sda", "shutdown -h", "reboot -f", "init 0")
    lc = c.strip().lower()
    for b in bad:
        if b in lc:
            return False
    return True

def build_system_prompt(web_flag: str) -> str:
    # General, no casos específicos de servicios concretos; patrones reutilizables
    return textwrap.dedent(f"""
    Eres un agente de automatización en sistemas Debian/Ubuntu. Salida SIEMPRE en JSON válido:
    {{"explanation": "...", "commands": ["cmd1","cmd2",...], "finished": false}}
    Reglas:
    - Propón comandos seguros y no destructivos. Prefiere 'install' a borrar.
    - Agrupa pasos cuando ayude a coherencia, pero no sobrepases el límite indicado.
    - Si no hay nada que hacer, establece finished=true.
    - Limita comandos por paso (te diré el tope).
    - Usa 'bash -lc' para scripts multilínea cuando haga falta.
    - No uses 'apt-key'. Usa keyrings en /usr/share/keyrings + 'signed-by='.
    - Antes de modificar ficheros en /etc/, haz copia (tar a /root/*-backup-TS.tar.gz o .bak).
    - Cuando necesites inspección: muestra 'cat/grep' y 'systemctl status' sin interrumpir por errores (usa '|| true').
    - Al consultar servicios de base de datos o similares, intenta un enfoque GENÉRICO y reutilizable:
      * Para clientes SQL, intenta ejecutables genéricos disponibles y auto-descubrimiento de credenciales locales:
        - Usa primero 'ai-sql ...' si existe (es un wrapper que intenta sockets/credenciales locales de forma segura).
        - Si no existe 'ai-sql', intenta cliente nativo sin credenciales sensibles y con tolerancia a fallo:
          mariadb -e "..." || mysql -e "..." || true
      * Evita codificar contraseñas. Si se requiere, indica cómo se guardarían en un fichero de credenciales protegido.
    - Manejo web: web-mode={web_flag}. Si está 'on', puedes proponer 'curl' o clientes CLI adecuados con timeouts y parseo robusto.
    - Para outputs largos, resume. Evita verter ficheros enteros en salida.
    - Responde SOLO con el JSON indicado.
    """).strip()

def build_user_prompt(task: str, step: int, limits: dict) -> str:
    return textwrap.dedent(f"""
    Tarea del usuario: {task}

    Paso actual: {step}
    Límite de comandos por paso: {limits["max_cmds"]}
    Formato de respuesta requerido (JSON estricto):
    {{
      "explanation": "breve explicación del objetivo del paso",
      "commands": ["comando1", "comando2"],
      "finished": false
    }}
    """).strip()

def openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY)

def call_model(messages, model, max_tokens=None):
    client = openai_client()
    # Usamos Chat Completions; evitamos pasar temperature/top_p para compatibilidad
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # no temperature (algunos modelos no aceptan 0.0), no response_format
            # max_tokens opcional: mejor omitir para compatibilidad amplia
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    usage = getattr(resp, "usage", None)
    if usage:
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
    else:
        pt = ct = 0

    content = resp.choices[0].message.content if resp.choices else ""
    if VERBOSE:
        log_info("RAW >>> " + (content[:1600].replace("\n"," ") if content else "<empty>"))
    return content, pt, ct

def parse_model_json(text: str):
    try:
        # Busca el primer bloque JSON en el texto
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        raw = text[start:end+1]
        return json.loads(raw)
    except Exception:
        return {}

def estimate_cost(total_prompt, total_completion):
    tokens = total_prompt + total_completion
    return (tokens/1000.0) * COST_PER_1K

def main():
    parser = argparse.ArgumentParser(description="AI automation agent")
    parser.add_argument("--task", help="Initial single task (non-interactive if provided)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use (default env/installed)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-cmds-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP)
    parser.add_argument("--web-mode", default=WEB_MODE, choices=["auto","on","off"])
    parser.add_argument("--help-agent", action="store_true", help="Show agent help")
    args = parser.parse_args()

    if args.help_agent:
        print(textwrap.dedent(f"""
        Agent help
        ==========
        Environment:
          OPENAI_API_KEY             required
          AI_AGENT_DEFAULT_MODEL     default model (current: {DEFAULT_MODEL})
          AI_AGENT_DEFAULT_MAX_STEPS default max steps (current: {DEFAULT_MAX_STEPS})
          AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP default max cmds per step (current: {DEFAULT_MAX_CMDS_PER_STEP})
          AI_AGENT_WEB_MODE          auto|on|off (current: {WEB_MODE})
          AI_AGENT_VERBOSE           1 enables raw debug logs

        Usage examples:
          ai-agent --task "muestra 'uname -a'"
          ai-agent --model gpt-5-mini
        """).strip())
        return 0

    interactive = not bool(args.task)
    print(f"[{now()}] Agent starting ({'interactive' if interactive else 'one-shot'}).")

    while True:
        if interactive:
            try:
                task = input("Enter task (or press Enter to exit): ").strip()
            except EOFError:
                task = ""
            if not task:
                print(f"[{now()}] Exiting interactive session.")
                break
        else:
            task = args.task.strip()
            if not task:
                log_err("No task provided.")
                return 1

        web_flag = decide_web_mode(task)
        print(f"[{now()}] web-mode: {web_flag}")

        messages = []
        sys_prompt = build_system_prompt(web_flag)
        messages.append({"role": "system", "content": sys_prompt})
        user_prompt = build_user_prompt(task, step=1, limits={"max_cmds": args.max_cmds_per_step})
        messages.append({"role": "user", "content": user_prompt})

        total_prompt_tokens = 0
        total_completion_tokens = 0

        finished = False
        for step in range(1, args.max_steps+1):
            print(f"[{now()}] === Step {step} ===")
            text, pt, ct = call_model(messages, args.model)
            total_prompt_tokens += pt
            total_completion_tokens += ct

            data = parse_model_json(text)
            explanation = (data.get("explanation") or "").strip()
            commands = data.get("commands") or []
            finished_flag = bool(data.get("finished", False))

            # Robust empty/invalid handling
            if (not commands) and (explanation == "") and not finished_flag:
                # ask model to provide at least one command or finish
                messages.append({
                    "role": "user",
                    "content": "Tu respuesta anterior estaba vacía. Devuelve JSON con 'explanation', 'commands' (>=1) o 'finished': true. SOLO JSON."
                })
                text, pt, ct = call_model(messages, args.model)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                data = parse_model_json(text)
                explanation = (data.get("explanation") or "").strip()
                commands = data.get("commands") or []
                finished_flag = bool(data.get("finished", False))

            if explanation:
                print(f"[{now()}] AI explanation: {explanation}")
            if commands:
                print(f"[{now()}] Proposed commands:")
                for c in commands[:args.max_cmds_per_step]:
                    print(f"  $ {c}")

            if finished_flag:
                print(f"[{now()}] Task complete.")
                print("Summary:")
                print(explanation or "Finished.")
                finished = True
                break

            # Execute commands
            any_executed = False
            for idx, cmd in enumerate(commands[:args.max_cmds_per_step], start=1):
                if not safe_command_line(cmd):
                    log_err(f"Blocked dangerous command: {cmd}")
                    continue
                rc, out, err = run_shell(cmd)
                any_executed = True
                print(f"[{now()}] Executing command {idx}/{min(len(commands), args.max_cmds_per_step)}: {cmd}")
                if out.strip():
                    print("-- STDOUT --")
                    print(out.rstrip())
                if err.strip():
                    print("-- STDERR --")
                    print(err.rstrip())
                print(f"[{now()}] cmd='{cmd}' rc={rc}")
                # summarize back
                out_summary = f"Command: {cmd}\nReturn code: {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                messages.append({"role": "user", "content": out_summary})

            if any_executed:
                messages.append({"role": "user", "content": f"Proporciona el siguiente plan. Límite {args.max_cmds_per_step} comandos. JSON solo."})
            else:
                messages.append({"role": "user", "content": "No se ejecutó ningún comando. Ofrece un plan alternativo o finished=true. JSON solo."})

        print(f"\n[{now()}] Approximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
        cost = estimate_cost(total_prompt_tokens, total_completion_tokens)
        print(f"[{now()}] Estimated cost for model '{args.model}': ${cost:.4f} (USD)")

        if not interactive:
            break

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_err(f"Fatal error: {e}")
        sys.exit(1)
PYEOF
  chmod +x "$AGENT_PY"
}

write_sql_helper() {
  cat >"$SQL_HELPER" <<'EOF'
#!/usr/bin/env bash
# ai-sql: generic helper for local MariaDB/MySQL access without hardcoding secrets.
set -euo pipefail
# Try safest options first
if [ -r /etc/mysql/debian.cnf ]; then
  exec /usr/bin/mariadb --defaults-file=/etc/mysql/debian.cnf "$@"
fi
# Socket root without password (common on Debian/MariaDB)
if command -v mariadb >/dev/null 2>&1; then
  exec /usr/bin/mariadb -uroot "$@"
fi
if command -v mysql >/dev/null 2>&1; then
  exec /usr/bin/mysql -uroot "$@"
fi
echo "ai-sql: no suitable local credentials found. Provide --user/--password or create /etc/mysql/debian.cnf" >&2
exit 1
EOF
  chmod +x "$SQL_HELPER"
}

write_wrapper() {
  cat >"$WRAPPER" <<'BASHWRAP'
#!/usr/bin/env bash
# ai-agent - wrapper for the Python AI automation agent
set -euo pipefail

INSTALL_DIR="/opt/ai-agent"
VENV_DIR="$INSTALL_DIR/.venv"
AGENT="$INSTALL_DIR/ai_server_agent.py"
STATE_DIR="/etc/ai-agent"
ENV_FILE="$STATE_DIR/agent.env"

DEFAULT_MODEL="${AI_AGENT_DEFAULT_MODEL:-gpt-5-mini}"
DEFAULT_MAX_STEPS="${AI_AGENT_DEFAULT_MAX_STEPS:-30}"
DEFAULT_MAX_CMDS="${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:-24}"
DEFAULT_WEB_MODE="${AI_AGENT_WEB_MODE:-auto}"

usage() {
  cat <<'EOF'
ai-agent - Ejecuta el agente de automatización para Debian/Ubuntu

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en /etc/ai-agent/agent.env
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent -h | --help             # Ayuda del wrapper (no requiere API)

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  /etc/ai-agent/agent.env y, si hay TTY, la pedirá e incluso permitirá guardarla.
- Variables por defecto:
  AI_AGENT_DEFAULT_MODEL (por defecto: gpt-5-mini)
  AI_AGENT_DEFAULT_MAX_STEPS (30)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP (24)
  AI_AGENT_WEB_MODE (auto|on|off; por defecto auto)
- Para ver ayuda del agente Python:
  ai-agent --help-agent
EOF
}

mask() {
  local s="$1"
  if [ -z "$s" ]; then echo "(empty)"; return; fi
  local n=${#s}
  if [ $n -le 8 ]; then echo "********"; return; fi
  local head="${s:0:4}"
  local tail="${s:(-4)}"
  printf "%s********%s\n" "$head" "$tail"
}

ensure_envdir() { mkdir -p "$STATE_DIR"; chmod 0755 "$STATE_DIR"; }

set_key() {
  ensure_envdir
  local key="${1:-}"
  if [ -z "$key" ]; then
    if [ -t 0 ]; then
      read -r -s -p "Introduce tu OPENAI_API_KEY: " key
      echo
    else
      echo "No API key provided and no TTY to prompt." >&2
      exit 1
    fi
  fi
  echo "OPENAI_API_KEY=$key" > "$ENV_FILE"
  chmod 0600 "$ENV_FILE"
  echo "[OK] Clave guardada en $ENV_FILE"
}

clear_key() {
  rm -f "$ENV_FILE"
  echo "[OK] Clave eliminada (si existía)"
}

show_key() {
  if [ -r "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    echo "OPENAI_API_KEY=$(mask "${OPENAI_API_KEY:-}")"
  else
    echo "No hay clave persistida en $ENV_FILE"
  fi
}

load_key_if_missing() {
  if [ -n "${OPENAI_API_KEY:-}" ]; then return; fi
  if [ -r "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  fi
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
    read -r -s -p "Introduce tu clave OpenAI API (se ocultará, deja en blanco para omitir): " k
    echo
    if [ -n "$k" ]; then
      export OPENAI_API_KEY="$k"
      read -r -p "¿Quieres guardarla en $ENV_FILE para futuros usos? [y/N]: " ans
      ans="${ans:-N}"
      case "$ans" in
        y|Y) set_key "$k" ;;
        *) ;;
      esac
    fi
  fi
}

if [ $# -gt 0 ]; then
  case "$1" in
    --set-key) shift; set_key "${1:-}"; exit 0 ;;
    --clear-key) clear_key; exit 0 ;;
    --show-key) show_key; exit 0 ;;
    -h|--help) usage; exit 0 ;;
  esac
fi

load_key_if_missing

if [ ! -x "$AGENT" ]; then
  echo "[ERROR] No se encuentra $AGENT. Reinstala el agente." >&2
  exit 1
fi
if [ ! -x "$VENV_DIR/bin/python3" ]; then
  echo "[ERROR] Entorno virtual no encontrado en $VENV_DIR. Reinstala el agente." >&2
  exit 1
fi

exec "$VENV_DIR/bin/python3" "$AGENT" "$@"
BASHWRAP
  chmod +x "$WRAPPER"
}

create_venv_and_deps() {
  info "Creating virtual environment..."
  mkdir -p "$INSTALL_DIR"
  if [ ! -x "$VENV_DIR/bin/python3" ]; then
    "$PY_BIN" -m venv "$VENV_DIR"
  fi
  info "Installing Python dependencies into the virtual environment..."
  "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
  # Keep deps minimal for reliability
  "$VENV_DIR/bin/pip" install --upgrade openai >/dev/null
}

persist_key_prompt_post_install() {
  if [ -t 0 ]; then
    echo
    read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " ans
    ans="${ans:-N}"
    if [[ "$ans" =~ ^[yY]$ ]]; then
      ensure_state_dir
      read -r -s -p "Introduce tu OPENAI_API_KEY: " key
      echo
      echo "OPENAI_API_KEY=$key" > "$ENV_FILE"
      chmod 0600 "$ENV_FILE"
      echo "[OK] Clave guardada en $ENV_FILE"
      echo
      echo "Listo. Prueba ahora:"
      echo "  ai-agent --help-agent"
      echo "  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\""
    fi
  fi
}

ensure_state_dir() { mkdir -p "$STATE_DIR"; chmod 0755 "$STATE_DIR"; }

main() {
  need_root
  self_update "$@"
  ensure_packages
  mkdir -p "$INSTALL_DIR"
  ensure_state_dir
  info "Creating installation directory at $INSTALL_DIR"
  write_python_agent
  info "Writing agent script to $AGENT_PY"
  write_report
  info "Writing report to $REPORT_FILE"
  create_venv_and_deps
  write_wrapper
  write_sql_helper
  info "Creating wrapper executable $WRAPPER"
  echo "[SUCCESS] Installation complete."
  persist_key_prompt_post_install || true
}

main "$@"

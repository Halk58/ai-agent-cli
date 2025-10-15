#!/usr/bin/env bash
# ai-agent installer
# Version
SCRIPT_VERSION="1.9.3"
SELF_URL_DEFAULT="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Color helpers ----------
c_info="\033[1;34m[INFO]\033[0m"
c_warn="\033[1;33m[WARN]\033[0m"
c_err="\033[1;31m[ERROR]\033[0m"
c_ok="\033[1;32m[SUCCESS]\033[0m"

log()   { printf "%b %s\n" "$c_info" "$*"; }
warn()  { printf "%b %s\n" "$c_warn" "$*"; }
error() { printf "%b %s\n" "$c_err"  "$*" >&2; }
ok()    { printf "%b %s\n" "$c_ok"   "$*"; }

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    error "Este instalador necesita privilegios de root."
    exit 1
  fi
}

# ---------- Compare versions ----------
ver_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }
ver_ge() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" = "$2"; }

# ---------- Self-update ----------
self_update() {
  local url="${1:-$SELF_URL_DEFAULT}"
  local tmp
  tmp=$(mktemp)
  if curl -fsSL "$url" -o "$tmp"; then
    local remote_ver
    remote_ver=$(grep -E '^SCRIPT_VERSION=' "$tmp" | head -n1 | cut -d'"' -f2 || true)
    if [ -n "$remote_ver" ] && ver_gt "$SCRIPT_VERSION" "$remote_ver"; then
      log "Una versión más nueva ($remote_ver) está disponible. Actualizando desde $url ..."
      install -m 0755 "$tmp" "$0"
      rm -f "$tmp"
      log "Re-ejecutando el instalador actualizado..."
      exec "$0" "$@"
    fi
  fi
  rm -f "$tmp" || true
}

# ---------- Globals ----------
AGENT_DIR="/opt/ai-agent"
WRAPPER="/usr/local/bin/ai-agent"
CONF_DIR="/etc/ai-agent"
ENV_FILE="$CONF_DIR/agent.env"
VENV_DIR="$AGENT_DIR/.venv"
PY="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"
DEFAULT_MODEL="gpt-5-mini"         # Cambiable en wrapper/env
DEFAULT_MAX_STEPS="24"
DEFAULT_MAX_CMDS_PER_STEP="24"
DEFAULT_WEB_MODE="auto"            # auto|on|off

need_root
self_update

log "Actualizando índices de paquetes..."
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
else
  warn "No se detecta apt-get. Este instalador está pensado para Debian/Ubuntu. Continuaré, pero puede fallar."
fi

log "Instalando dependencias del sistema..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends     python3 python3-venv python3-pip     curl ca-certificates gpg jq coreutils     procps lsb-release
fi

log "Creando directorio de instalación en $AGENT_DIR"
mkdir -p "$AGENT_DIR"

# ---------- Escribir agente Python ----------
log "Escribiendo agente a $AGENT_DIR/ai_server_agent.py"
cat >"$AGENT_DIR/ai_server_agent.py" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Server Agent for Debian-like systems
- Ejecuta tareas del sistema propuestas por un modelo OpenAI
- Con pasos explicables y ejecución de comandos bash segura
- Verbosidad con timestamps y control de errores robusto
"""

import os, sys, json, time, shlex, subprocess, textwrap, datetime, re, platform
from typing import List, Dict, Any, Optional

# ====== Config por defecto (sobrescribible por env o args) ======
DEFAULT_MODEL = os.environ.get("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
MAX_STEPS     = int(os.environ.get("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
MAX_CMDS      = int(os.environ.get("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
WEB_MODE      = os.environ.get("AI_AGENT_WEB_MODE", "auto")  # auto|on|off

# Tokens/costos aproximados (informativo)
MODEL_PRICES = {
    # valores orientativos, ajustar a tu tabla real si lo deseas
    "gpt-5-mini": {"prompt": 0.0000005, "completion": 0.0000010},  # $/token aprox
}

def ts() -> str:
    return datetime.datetime.utcnow().strftime("[%Y-%m-%d %H:%M:%S]")

def log(msg: str) -> None:
    print(f"{ts()} {msg}", flush=True)

def run(cmd: List[str], timeout: Optional[int]=None, check: bool=False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=check)

def run_shell(bash_script: str, timeout: Optional[int]=None) -> Dict[str,str]:
    # Ejecuta bajo bash -lc con flags seguros cuando el script los añada.
    cp = subprocess.run(["bash", "-lc", bash_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return {"stdout": cp.stdout, "stderr": cp.stderr, "returncode": str(cp.returncode)}

def which(cmd: str) -> bool:
    return subprocess.call(["bash", "-lc", f"command -v {shlex.quote(cmd)} >/dev/null 2>&1"]) == 0

# ====== OpenAI Client (robusto a variaciones) ======
def new_openai_client():
    """
    Construye cliente OpenAI (openai>=1.x). Evita parámetros no soportados.
    Preferimos Chat Completions por compatibilidad; Responses queda como fallback manual.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        log(f"ERROR: no se puede importar openai: {e}")
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        log("ERROR: OPENAI_API_KEY no está definido.")
        sys.exit(2)
    return OpenAI(api_key=api_key)

def support_responses_api(client) -> bool:
    # Heurístico: comprobamos que exista atributo .responses
    return hasattr(client, "responses")

def call_chat_completions(client, model: str, messages: List[Dict[str,Any]], max_tokens: int=1200) -> Dict[str,Any]:
    """
    Llamada segura a Chat Completions API.
    - No pasamos temperature ni response_format (evita errores de modelos con restricciones).
    - max_tokens solo para chat.completions (Responses no lo soporta en varios clientes).
    """
    from openai import OpenAI
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return {"ok": True, "raw": resp}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def call_responses(client, model: str, sys_prompt: str, user_prompt: str) -> Dict[str,Any]:
    """
    Fallback a Responses API SIN parámetros problemáticos.
    - No usamos 'max_completion_tokens', ni 'response_format', ni 'temperature'.
    - Algunos clientes exigen input en formato {'role','content'} o 'input' simple.
    """
    try:
        if support_responses_api(client):
            # Intento mínimo posible
            resp = client.responses.create(
                model=model,
                input=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_prompt},
                ]
            )
            return {"ok": True, "raw": resp}
        else:
            return {"ok": False, "error": "Responses API no disponible en el cliente"}
    except TypeError as te:
        return {"ok": False, "error": f"TypeError en Responses: {te}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ====== Prompting ======
def build_system_prompt(db_probe: Dict[str,str]) -> str:
    # Instrucciones generales para tareas de administración en Debian, con énfasis en seguridad.
    # Incluye guía genérica de DB accesos sin password -> debian.cnf fallback.
    guidance = f"""
Eres un asistente experto DevOps/SRE que genera planes de acción en Bash **seguros y no destructivos** para sistemas Debian/Ubuntu.
REGLAS:
- Genera una explicación breve y una lista de comandos shell a ejecutar.
- Los comandos deben usar: bash -lc con 'set -euo pipefail' cuando modifiquen estado.
- Para APT usa siempre: DEBIAN_FRONTEND=noninteractive y opciones Dpkg para conservar config:
  apt-get -y -o Dpkg::Options::=\"--force-confdef\" -o Dpkg::Options::=\"--force-confold\"
- Para gpg/llaves usa: gpg --batch --yes, y evita prompts interactivos.
- Escribe ficheros de configuración con here-docs con comillas simples en EOF (<<'EOF') o con printf/tee para evitar expansión.
- Al modificar MariaDB/MySQL:
  * Antes de 'SHOW VARIABLES' intenta ejecutar cliente sin password usando socket:
    primero '{db_probe.get('try_1')}', si falla usa '{db_probe.get('try_2')}' y si no '{db_probe.get('try_3')}'.
  * Si cambias innodb_log_file_size, detén servicio, mueve ib_logfile* a .bak con timestamp y arranca de nuevo.
  * Evita asignar más del 60% de la RAM al buffer pool en VPS pequeños.
- Nunca pidas confirmaciones interactivas; todo debe ser automatizado y reversible (backup antes de tocar).
- Si necesitas web/HTTP, usa curl con timeouts y retries y explica la fuente.
- Devuelve JSON válido con claves: explanation (string), commands (lista de strings).
"""
    return textwrap.dedent(guidance).strip()

def build_user_prompt(task: str, facts: str) -> str:
    prompt = f"""
TAREA: {task}

ENTORNO/SO HECHOS:
{facts}

DEVUELVE:
{{
  "explanation": "breve explicación de lo que harás",
  "commands": ["bash -lc '...'", "comando 2", "... (máximo {MAX_CMDS} comandos)"]
}}
"""
    return textwrap.dedent(prompt).strip()

# ====== Fact gathering ======
def detect_db_probe() -> Dict[str,str]:
    # Construye intentos genéricos para consultar DB sin password y con fallback a debian.cnf
    tries = {}
    if which("mariadb"):
        tries["try_1"] = "mariadb -e \"SELECT 1;\""
        tries["try_2"] = "mariadb --defaults-file=/etc/mysql/debian.cnf -e \"SELECT 1;\""
        tries["try_3"] = "mysql --defaults-file=/etc/mysql/debian.cnf -e \"SELECT 1;\""
    elif which("mysql"):
        tries["try_1"] = "mysql -e \"SELECT 1;\""
        tries["try_2"] = "mysql --defaults-file=/etc/mysql/debian.cnf -e \"SELECT 1;\""
        tries["try_3"] = "mariadb --defaults-file=/etc/mysql/debian.cnf -e \"SELECT 1;\""
    else:
        # ni mariadb ni mysql: sugerir instalación
        tries["try_1"] = "mariadb -e \"SELECT 1;\""
        tries["try_2"] = "mysql -e \"SELECT 1;\""
        tries["try_3"] = "mariadb --defaults-file=/etc/mysql/debian.cnf -e \"SELECT 1;\""
    return tries

def collect_facts() -> str:
    cmds = [
        "uname -a",
        "cat /etc/os-release || true; echo DEBIAN_VERSION_FULL=$(cat /etc/debian_version 2>/dev/null || true)",
        "free -m || true",
        "nproc 2>/dev/null || echo 1",
        "df -h / || true",
        "systemctl is-active mariadb 2>/dev/null || systemctl is-active mysql 2>/dev/null || echo inactive",
        "ps -o pid,pmem,rss,cmd -C mariadbd --no-headers 2>/dev/null | head -n1 || true",
        "mariadb --version 2>/dev/null || mysql --version 2>/dev/null || echo no-mysql-client",
        "grep -RHiE \"(innodb_buffer_pool_size|max_connections|innodb_log_file_size|tmp_table_size|max_heap_table_size)\" /etc/mysql 2>/dev/null || true",
        "grep -RHiE \"^\\s*(!includedir|!include)\" /etc/mysql 2>/dev/null || true",
        "du -sh /var/lib/mysql 2>/dev/null || true",
        "ls -lah /var/lib/mysql 2>/dev/null | sed -n '1,40p' || true",
    ]
    out = []
    for c in cmds:
        proc = subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        snippet = proc.stdout.strip()
        if not snippet:
            snippet = proc.stderr.strip()
        out.append(snippet)
    return "\n".join(out)

# ====== JSON helpers ======
def parse_json_block(s: str) -> Optional[Dict[str,Any]]:
    # Extrae primer bloque JSON válido de una cadena.
    match = re.search(r'\{(?:[^{}]|(?R))*\}', s, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

# ====== Main agent ======
def main():
    import argparse
    ap = argparse.ArgumentParser(description="AI Agent (Debian)")
    ap.add_argument("--task", help="Tarea a ejecutar", default="")
    ap.add_argument("--model", help="Modelo OpenAI", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--max-cmds", type=int, default=MAX_CMDS)
    ap.add_argument("--web", choices=["auto","on","off"], default=WEB_MODE)
    args = ap.parse_args()

    task = args.task
    if not task:
        print(f"{ts()} Agent starting (interactive).")
        try:
            task = input("Enter task (or press Enter to exit): ").strip()
        except EOFError:
            task = ""
        if not task:
            print(f"{ts()} Exiting interactive session.")
            return

    client = new_openai_client()
    model = args.model

    # Pre-facts
    db_probe = detect_db_probe()
    sys_prompt = build_system_prompt(db_probe)
    facts = collect_facts()
    user_prompt = build_user_prompt(task, facts)

    step = 1
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def print_step_header(n: int):
        print(f"{ts()} === Step {n} ===", flush=True)

    max_steps = args.max_steps
    # Mensajes para Chat Completions
    chat_messages = [
        {"role":"system","content":sys_prompt},
        {"role":"user","content":user_prompt},
    ]

    while step <= max_steps:
        print_step_header(step)
        # 1) Intentar Chat Completions primero
        cc = call_chat_completions(client, model, chat_messages, max_tokens=int(os.environ.get("AI_AGENT_MAX_TOKENS", "1200")))
        plan = None
        used_cc = False
        if cc["ok"]:
            used_cc = True
            raw = cc["raw"]
            try:
                content = raw.choices[0].message.content or ""
            except Exception:
                content = ""
            if content:
                plan = parse_json_block(content)
        else:
            log(f"Chat Completions falló: {cc.get('error')}")

        # 2) Fallback a Responses si no hay plan
        if not plan:
            rr = call_responses(client, model, sys_prompt, user_prompt)
            if rr["ok"]:
                raw = rr["raw"]
                # responses payload varía según cliente; intentar campos comunes
                content = ""
                try:
                    if hasattr(raw, "output") and hasattr(raw.output, "text"):
                        content = raw.output.text or ""
                    elif hasattr(raw, "output_text"):
                        content = raw.output_text or ""
                    else:
                        content = str(raw)
                except Exception:
                    content = str(raw)
                plan = parse_json_block(content)
            else:
                log(f"Responses API falló: {rr.get('error')}")

        # 3) Si seguimos sin plan, reintentar con instrucción adicional
        if not plan:
            retry_note = {
                "role":"user",
                "content":"El JSON no fue válido o estaba vacío. Devuelve SOLO un JSON con 'explanation' y 'commands' (lista)."
            }
            chat_messages.append(retry_note)
            step += 1
            continue

        explanation = plan.get("explanation", "").strip()
        commands = plan.get("commands", [])
        if explanation:
            print(f"{ts()} AI explanation: {explanation}")
        else:
            print(f"{ts()} AI explanation: ")

        if not isinstance(commands, list):
            commands = []

        # Ejecutar comandos con límites
        cmds = commands[:args.max_cmds]
        if not cmds:
            # pedir refinamiento
            chat_messages.append({"role":"user","content":"No has propuesto comandos. Propón comandos shell seguros (máximo {MAX_CMDS})."})
            step += 1
            continue

        print(f"{ts()} Proposed commands:")
        for c in cmds:
            print(f"  $ {c}")

        # Ejecutar uno por uno
        for idx, c in enumerate(cmds, start=1):
            print()
            print(f"{ts()} Executing command {idx}/{len(cmds)}: {c}")
            res = run_shell(c)
            out = res.get("stdout","")
            err = res.get("stderr","")
            rc  = res.get("returncode","")
            if out.strip():
                print("-- STDOUT --")
                print(out.rstrip())
            if err.strip():
                print("-- STDERR --")
                print(err.rstrip())
            if rc != "0":
                print(f"{ts()} Command failed with return code {rc}. Aborting step.")
                break
        else:
            print(f"{ts()} Task complete.")
            print("Summary:")
            print(explanation or "(sin explicación)")
            break

        # Si falló un comando, pedir al modelo corrección incremental
        chat_messages.append({"role":"assistant","content":json.dumps(plan, ensure_ascii=False)})
        chat_messages.append({"role":"user","content":f"El comando falló con rc={rc}. STDERR:\n{err}\nAjusta el plan y devuelve JSON válido."})
        step += 1

    # Costos aproximados si están disponibles
    try:
        if used_cc and 'raw' in locals():
            usage = getattr(raw, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
        # No todas las rutas acumulan correctamente; mostramos últimos si están.
        if total_prompt_tokens or total_completion_tokens:
            rate = MODEL_PRICES.get(model, {"prompt":0.0,"completion":0.0})
            cost = total_prompt_tokens*rate["prompt"] + total_completion_tokens*rate["completion"]
            print()
            print(f"{ts()} Approximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
            print(f"{ts()} Estimated cost for model '{model}': ${cost:.4f} (USD)")
    except Exception:
        pass

if __name__ == "__main__":
    main()
PY
chmod 0755 "$AGENT_DIR/ai_server_agent.py"

# ---------- Escribir report ----------
log "Escribiendo reporte a $AGENT_DIR/report.md"
cat >"$AGENT_DIR/report.md" <<'MD'
# AI Agent for Debian/Ubuntu

**Versión:** 1.9.3

Cambios clave:
- Sonda MySQL/MariaDB genérica: primero sin credenciales, luego `/etc/mysql/debian.cnf` (si existe).
- Verbosidad con timestamps por paso y en ejecución de comandos.
- Manejo robusto de OpenAI API: Chat Completions por defecto; fallback a Responses sin parámetros conflictivos.
- Buenas prácticas de shell (no prompts interactivos, here-docs seguros, gpg --batch --yes, APT no interactivo).
- Control de errores y reintentos: si un comando falla, se replanifica con el STDERR.

## Rutas
- Código: `/opt/ai-agent/ai_server_agent.py`
- Wrapper: `/usr/local/bin/ai-agent`
- Config persistente: `/etc/ai-agent/agent.env`

MD

# ---------- Crear venv e instalar deps ----------
log "Creando entorno virtual..."
python3 -m venv "$VENV_DIR"
log "Instalando dependencias en el venv..."
"$PIP" install --upgrade pip >/dev/null
# OpenAI moderno; evita extras innecesarios
"$PIP" install --no-cache-dir "openai>=1.51.0" >/dev/null

# ---------- Wrapper cli ----------
log "Creando wrapper ejecutable $WRAPPER"
cat >"$WRAPPER" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'
	'

PROG="ai-agent"
AGENT_DIR="/opt/ai-agent"
CONF_DIR="/etc/ai-agent"
ENV_FILE="$CONF_DIR/agent.env"
VENV="$AGENT_DIR/.venv"
PY="$VENV/bin/python3"
AGENT="$AGENT_DIR/ai_server_agent.py"

# Defaults (overridable via env or key-values)
: "${AI_AGENT_DEFAULT_MODEL:=gpt-5-mini}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=24}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=24}"
: "${AI_AGENT_WEB_MODE:=auto}"
: "${AI_AGENT_MAX_TOKENS:=1200}"

usage() {
cat <<EOF
$PROG - Ejecuta el agente de automatización para Debian/Ubuntu

Uso:
  $PROG [opciones] --task "instrucción"
  $PROG                        # modo interactivo
  $PROG --set-key [API_KEY]    # guarda la API en $ENV_FILE
  $PROG --clear-key            # borra la API persistida
  $PROG --show-key             # muestra la API (enmascarada)
  $PROG -h | --help            # ayuda del wrapper

Opciones:
  --model NOMBRE               Modelo OpenAI (por defecto: $AI_AGENT_DEFAULT_MODEL)
  --max-steps N                Límite de pasos (por defecto: $AI_AGENT_DEFAULT_MAX_STEPS)
  --max-cmds N                 Máx comandos por paso (por defecto: $AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP)
  --web auto|on|off            Modo web (por defecto: $AI_AGENT_WEB_MODE)

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  $ENV_FILE y, si hay TTY, podrá pedírtela y guardarla.

Para ver ayuda del agente Python:
  $PROG --help-agent
EOF
}

help_agent() {
  echo "Lanzando ayuda integrada del agente..."
  "$PY" "$AGENT" -h || true
}

mask() {
  local s="${1:-}"
  if [ -z "$s" ]; then echo "(vacío)"; return; fi
  local len=${#s}
  if [ $len -le 8 ]; then echo "********"; return; fi
  printf "%s%s
" "${s:0:4}" "$(printf '*%.0s' $(seq 1 $((len-8))))${s: -4}"
}

ensure_api_key() {
  # 1) If env has the key, done
  if [ -n "${OPENAI_API_KEY:-}" ]; then return 0; fi
  # 2) Try to load from file
  if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    if [ -n "${OPENAI_API_KEY:-}" ]; then return 0; fi
  fi
  # 3) If interactive TTY -> ask
  if [ -t 0 ]; then
    read -r -p "Introduce tu OPENAI_API_KEY (Enter para cancelar): " key || true
    if [ -n "$key" ]; then
      export OPENAI_API_KEY="$key"
      read -r -p "¿Quieres guardarla para futuros usos? [y/N]: " yn || true
      if [[ "${yn,,}" == "y" ]]; then
        mkdir -p "$CONF_DIR"
        umask 077
        printf "OPENAI_API_KEY=%s
" "$OPENAI_API_KEY" > "$ENV_FILE"
        echo "[OK] Clave guardada en $ENV_FILE"
      fi
      return 0
    fi
  fi
  echo "[ERROR] No se encontró OPENAI_API_KEY. Usa '$PROG --set-key' o exporta la variable." >&2
  return 1
}

set_key() {
  local val="${1:-}"
  if [ -z "$val" ]; then
    if [ -t 0 ]; then
      read -r -p "Introduce tu OPENAI_API_KEY: " val || true
    fi
  fi
  if [ -z "$val" ]; then
    echo "No se proporcionó clave." >&2
    exit 2
  fi
  mkdir -p "$CONF_DIR"
  umask 077
  printf "OPENAI_API_KEY=%s
" "$val" > "$ENV_FILE"
  echo "[OK] Clave guardada en $ENV_FILE"
}

clear_key() {
  if [ -f "$ENV_FILE" ]; then
    rm -f "$ENV_FILE"
    echo "[OK] Clave eliminada de $ENV_FILE"
  else
    echo "No hay clave almacenada."
  fi
}

show_key() {
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY (env): $(mask "$OPENAI_API_KEY")"
  elif [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    if [ -n "${OPENAI_API_KEY:-}" ]; then
      echo "OPENAI_API_KEY (archivo): $(mask "$OPENAI_API_KEY")"
    else
      echo "No hay clave en $ENV_FILE"
    fi
  else
    echo "No hay clave."
  fi
}

# Parse args
TASK=""
MODEL="$AI_AGENT_DEFAULT_MODEL"
MAX_STEPS="$AI_AGENT_DEFAULT_MAX_STEPS"
MAX_CMDS="$AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP"
WEB="$AI_AGENT_WEB_MODE"

if [ $# -eq 0 ]; then
  # interactive; continue
  :
fi

while [ $# -gt 0 ]; do
  case "${1:-}" in
    --task) TASK="${2:-}"; shift 2;;
    --model) MODEL="${2:-}"; shift 2;;
    --max-steps) MAX_STEPS="${2:-}"; shift 2;;
    --max-cmds) MAX_CMDS="${2:-}"; shift 2;;
    --web) WEB="${2:-}"; shift 2;;
    --set-key) set_key "${2:-}"; exit 0;;
    --clear-key) clear_key; exit 0;;
    --show-key) show_key; exit 0;;
    --help-agent) help_agent; exit 0;;
    -h|--help) usage; exit 0;;
    *) echo "Opción desconocida: $1" >&2; usage; exit 1;;
  esac
done

# Ensure API key before launching agent (unless just asking help/keys)
ensure_api_key

# Launch agent
export AI_AGENT_DEFAULT_MODEL="$MODEL"
export AI_AGENT_DEFAULT_MAX_STEPS="$MAX_STEPS"
export AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="$MAX_CMDS"
export AI_AGENT_WEB_MODE="$WEB"

if [ -z "${TASK}" ]; then
  "$PY" "$AGENT"
else
  "$PY" "$AGENT" --task "$TASK"
fi
SH
chmod 0755 "$WRAPPER"

# ---------- Post install: prompt to store key if not set ----------
if [ -t 0 ] && [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "$ENV_FILE" ]; then
  echo
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " yn || true
  if [[ "${yn,,}" == "y" ]]; then
    read -r -p "Introduce tu OPENAI_API_KEY: " key || true
    if [ -n "$key" ]; then
      mkdir -p "$CONF_DIR"
      umask 077
      printf "OPENAI_API_KEY=%s
" "$key" > "$ENV_FILE"
      ok "Clave guardada en $ENV_FILE"
    else
      warn "No se guardó clave."
    fi
  fi
fi

ok "Instalación completada."
echo
echo "Prueba ahora:"
echo "  ai-agent --help-agent"
echo "  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\""

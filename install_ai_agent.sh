#!/bin/bash
# install_ai_agent.sh - Deploy / Update the AI server agent on Debian 12/13+
# Version: 1.4.0
# Usage:
#   sudo bash install_ai_agent.sh
#
# After install:
#   ai-agent --task "muestra 'uname -a' y 'lsb_release -a'"
#
# Notas:
# - Incluye auto-actualización desde UPDATE_URL si la versión remota es mayor.
# - Crea /opt/ai-agent, virtualenv, y /usr/local/bin/ai-agent (wrapper).
# - Gestión de API persistente en /etc/ai-agent/agent.env.

set -euo pipefail

SCRIPT_VERSION="1.8.0"
# Auto-update URL (pedida por ti):
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

# ---- sudo helper ------------------------------------------------------------
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

# ---- self-update ------------------------------------------------------------
if [ -n "${UPDATE_URL:-}" ]; then
  tmp_update="$(mktemp)"
  if curl -fsSL "$UPDATE_URL" -o "$tmp_update"; then
    remote_ver="$(grep -Eo 'SCRIPT_VERSION="[0-9.]+"' "$tmp_update" | head -n1 | cut -d'"' -f2 || true)"
    if [ -n "$remote_ver" ]; then
      newest="$(printf '%s\n%s\n' "$SCRIPT_VERSION" "$remote_ver" | sort -V | tail -n1)"
      if [ "$newest" = "$remote_ver" ] && [ "$SCRIPT_VERSION" != "$remote_ver" ]; then
        echo "[INFO] A newer version ($remote_ver) is available. Updating self from $UPDATE_URL ..."
        script_path="$(readlink -f -- "$0")"
        if [ -w "$script_path" ]; then
          cp "$tmp_update" "$script_path"
        else
          ${SUDO} cp "$tmp_update" "$script_path"
        fi
        chmod +x "$script_path"
        echo "[INFO] Re-executing the updated installer..."
        exec "$script_path" "$@"
      fi
    fi
  fi
  rm -f "$tmp_update" || true
fi

# ---- packages ---------------------------------------------------------------
echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y
echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip ca-certificates curl git

# ---- layout -----------------------------------------------------------------
DEST_DIR="/opt/ai-agent"
VENV_DIR="$DEST_DIR/venv"
WRAPPER="/usr/local/bin/ai-agent"
CONF_DIR="/etc/ai-agent"
ENV_FILE="$CONF_DIR/agent.env"

echo "[INFO] Creating installation directory at $DEST_DIR"
${SUDO} mkdir -p "$DEST_DIR" "$CONF_DIR"

# ---- write agent python -----------------------------------------------------
echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
# ai_server_agent.py - generic server automation agent
# - Robust JSON parsing (sin JSON mode obligatorio)
# - Fallback Responses API / Chat Completions (sin temperature)
# - Max tokens seguro (max_completion_tokens)
# - Verbose RAW opcional (AI_AGENT_VERBOSE=1)
# - Evita bucles por respuestas vacías; reintenta y corta a la 3ª
# - No reinyecta el JSON del asistente en cada paso (menos ruido)
# - Normaliza comandos (no interactivo, gpg --batch, mysql defaults-file, etc.)
# - Combina comandos por paso en script temporal (preserva variables)
# - Detección mejorada de peligros; allowlist para limpiezas de APT
# - Web-mode auto/on/off (transparente); solo se loguea si verbose
import os, sys, re, json, time, shlex, tempfile, subprocess, argparse
from typing import List, Dict, Tuple, Any, Optional

# ------------------- logging -------------------
def now_ts() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

def log_info(msg: str) -> None:
    print(f"{now_ts()} {msg}")

def log_err(msg: str) -> None:
    print(f"{now_ts()} {msg}", file=sys.stderr)

VERBOSE = os.getenv("AI_AGENT_VERBOSE", "").lower() not in ("", "0", "false", "no")

# ------------------- openai client -------------------
# Compatible con openai>=1.0.0
try:
    from openai import OpenAI
    _client = OpenAI()
except Exception as e:
    log_err("OpenAI client not available. Did you install 'openai' and set OPENAI_API_KEY?")
    raise

# ------------------- CLI -------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI server agent for Debian automation")
    p.add_argument("--task", type=str, help="Initial natural language instruction")
    p.add_argument("--model", type=str, default=os.getenv("AI_AGENT_DEFAULT_MODEL","gpt-5-mini"))
    p.add_argument("--max-steps", type=int, default=int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS","24")))
    p.add_argument("--max-cmds", type=int, default=int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP","24")))
    p.add_argument("--web-mode", choices=["auto","on","off"], default=os.getenv("AI_AGENT_WEB_MODE","auto"))
    p.add_argument("--help-agent", action="store_true", help="Show agent help and exit")
    return p.parse_args()

# ------------------- helpers -------------------
def is_tty() -> bool:
    return sys.stdin.isatty()

def run(cmd: List[str], env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None) -> Tuple[int,str,str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)

def run_shell(command: str, timeout: Optional[int]=None) -> Tuple[int,str,str]:
    return run(["/bin/bash","-lc", command], timeout=timeout)

def dangerous_command(cmd: str) -> bool:
    s = cmd.strip()
    low = s.lower()

    # allowlist rm limpiezas legitimas
    ALLOW_RM = [
        "/var/lib/apt/lists/*",
        "/var/cache/apt/archives/*",
        "/tmp/*",
        "/var/tmp/*",
    ]
    if re.search(r'^\s*rm\s+-rf\s+', s):
        # si va contra /, bloquea; si es allowlist, permite
        if re.search(r'^\s*rm\s+-rf\s+/\s*($|[#;&|\s])', s):
            return True
        if any(p in s for p in ALLOW_RM):
            return False
        # eliminar cosas arbitrarias fuera de allowlist: pedir al modelo que explique primero
        return True

    blocked_patterns = [
        r'\bmkfs(\.| )', r'\bwipefs\b', r'\bsfdisk\b', r'\bparted\b.+/dev/',
        r'\bdd\s+if=/dev/', r'\bchattr\s+-i\s+/', r'\buserdel\s+-r\s+root\b',
        r'\b(shutdown|reboot|halt)\b', r':\(\)\s*{\s*:\|\:&\s*};:',
        r'\blvremove\b', r'\bvgremove\b', r'\bpvremove\b',
    ]
    for pat in blocked_patterns:
        if re.search(pat, low):
            return True
    return False

def normalize_command(cmd: str) -> str:
    s = cmd.strip()

    # apt-get: forzar no interactivo + -y en installs/upgrades
    if re.search(r'^\s*apt(-get)?\s+(install|upgrade|dist-upgrade|full-upgrade)\b', s):
        if "DEBIAN_FRONTEND" not in s:
            s = f"DEBIAN_FRONTEND=noninteractive {s}"
        if re.search(r'\sinstall\b', s) and " -y" not in s:
            s = s.replace("install", "install -y", 1)
        # opciones de dpkg seguras
        if "Dpkg::Options::=" not in s:
            s = s + " -o Dpkg::Options::='--force-confdef' -o Dpkg::Options::='--force-confold'"

    # curl: endurecer por defecto
    if re.search(r'^\s*curl\b', s) and "-f" not in s and "--fail" not in s:
        if "--silent" not in s and "-s" not in s:
            s = s.replace("curl", "curl -fsS", 1)
        else:
            s = s.replace("curl", "curl -fS", 1)
        if "--retry" not in s:
            s = s + " --retry 2"
        if "--max-time" not in s:
            s = s + " --max-time 20"

    # gpg --dearmor: evitar prompts
    if "gpg" in s and "--dearmor" in s and "--batch" not in s:
        s = s.replace("gpg ", "gpg --batch --yes ", 1)

    # systemctl status: sin paginador
    if re.search(r'^\s*systemctl\s+status\b', s) and "--no-pager" not in s:
        s = s + " --no-pager"

    # mysql/mariadb tools: usar defaults-file si existe y no hay credenciales
    defaults = None
    for cand in ("/etc/mysql/debian.cnf", "/root/.my.cnf"):
        if os.path.isfile(cand):
            defaults = cand
            break
    if defaults and re.search(r'^\s*(mysql|mariadb|mysqldump)\b', s):
        if "--defaults-file" not in s and " -p" not in s and "--user" not in s:
            # insertar justo después del binario
            s = re.sub(r'^\s*(mysql|mariadb|mysqldump)\b', rf"\g<0> --defaults-file={defaults}", s, count=1)

    return s

def commands_to_script(commands: List[str]) -> str:
    # normaliza cada línea y compone un script bash temporal
    lines = []
    for c in commands:
        c = c.strip()
        if not c:
            continue
        lines.append(normalize_command(c))
    if not lines:
        return ""
    script = "#!/usr/bin/env bash\nset -euo pipefail\n"
    script += "\n".join(lines) + "\n"
    return script

def exec_commands_as_script(commands: List[str]) -> Tuple[int,str,str]:
    script = commands_to_script(commands)
    if not script:
        return 0, "", ""
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="ai-agent-step-", suffix=".sh") as f:
        path = f.name
        f.write(script)
    os.chmod(path, 0o755)
    rc, out, err = run(["/bin/bash", path])
    try:
        os.unlink(path)
    except Exception:
        pass
    return rc, out, err

def extract_json_from_text(text: str) -> Optional[Dict[str,Any]]:
    if not text:
        return None
    # intenta parsear todo
    try:
        return json.loads(text)
    except Exception:
        pass
    # busca el primer objeto {...} balanceado de forma simple
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return None
    return None

def model_supports_responses(model: str) -> bool:
    # Heurística amplia; si falla, caemos a chat completions
    return any(model.lower().startswith(p) for p in ("gpt-5", "o4", "o3", "gpt-4o"))

def call_openai(messages: List[Dict[str,str]], model: str,
                web_mode: str,
                max_completion_tokens: int = 2048) -> Tuple[Dict[str,Any], Dict[str,int]]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    content_text = ""

    if model_supports_responses(model):
        # Responses API
        try:
            # No usamos response_format para evitar errores de compatibilidad.
            # Pedimos al modelo JSON via prompt; luego validamos aquí.
            r = _client.responses.create(
                model=model,
                input=messages,
                max_completion_tokens=max_completion_tokens
            )
            # extrae texto
            text = ""
            for out in r.output_text or []:
                text += out
            if not text and hasattr(r, "output_text"):
                text = r.output_text
            if VERBOSE:
                log_info("RAW (responses) >>> " + (text[:1200].replace("\n"," ") if text else "<empty>"))
            content_text = text or ""
            # usage (si viene)
            try:
                usage["prompt_tokens"] = getattr(r.usage, "input_tokens", 0) or 0
                usage["completion_tokens"] = getattr(r.usage, "output_tokens", 0) or 0
            except Exception:
                pass
        except Exception as e:
            log_err(f"Responses API failed ({type(e).__name__}): {e}. Falling back to Chat Completions.")
            # fallback a chat completions
            pass

    if not content_text:
        # Chat Completions fallback
        try:
            # No temperature, no response_format para compatibilidad amplia
            comp = _client.chat.completions.create(
                model=model,
                messages=messages,
                # algunos modelos no aceptan 'max_tokens'; otros sí. Mejor omitir.
            )
            content_text = comp.choices[0].message.content or ""
            if VERBOSE:
                log_info("RAW (chat) >>> " + (content_text[:1200].replace("\n"," ") if content_text else "<empty>"))
            try:
                usage["prompt_tokens"] = getattr(comp.usage, "prompt_tokens", 0) or 0
                usage["completion_tokens"] = getattr(comp.usage, "completion_tokens", 0) or 0
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}")

    data = extract_json_from_text(content_text)
    if not data:
        # fuerza al modelo a responder con JSON válido en el siguiente turno
        data = {}

    return data, usage

# ------------------- prompt -------------------
def build_system_msg(web_mode: str) -> str:
    # Instrucciones generales; SIN reglas específicas de MySQL.
    return (
        "You are a Linux server automation agent on Debian 12/13.\n"
        "Translate user goals into safe shell commands. After each step, you see the outputs and decide the next step.\n"
        "ALWAYS reply ONLY with a JSON object having EXACT keys: "
        "{\"commands\": [\"...\"], \"explanation\": \"...\", \"finished\": true|false}.\n"
        "Guidelines:\n"
        "- Prefer non-interactive flags; avoid prompts (e.g. use --yes/--batch and noninteractive envs).\n"
        "- Use Debian-appropriate commands; avoid interactive editors.\n"
        "- If a previous command failed, do NOT repeat it unchanged; propose an alternative.\n"
        "- Never suggest destructive operations.\n"
        "- If information is missing, state the assumption or propose a safe probe command to gather it.\n"
        "- If the task is already complete, set finished=true.\n"
        f"- Web-mode is '{web_mode}'. If 'on' and you need current web info, plan accordingly; if 'off', avoid web. If 'auto', decide pragmatically.\n"
    )

# ------------------- main loop -------------------
def main() -> None:
    args = parse_args()
    if args.help_agent:
        print("Agent help:\n"
              "  --task \"...\"                 Initial instruction\n"
              "  --model NAME                  Model (default env AI_AGENT_DEFAULT_MODEL)\n"
              "  --max-steps N                 Max reasoning steps per task\n"
              "  --max-cmds N                  Max commands per step\n"
              "  --web-mode auto|on|off        Web mode policy (metadata only)\n"
              "ENV:\n"
              "  AI_AGENT_VERBOSE=1            Show RAW model outputs and decisions\n"
              "  AI_AGENT_DEFAULT_MODEL        Default model if --model omitted\n"
              "  AI_AGENT_DEFAULT_MAX_STEPS    Default max steps\n"
              "  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP  Default max cmds per step\n"
        )
        return

    log_info("Agent starting (interactive).")
    # Decide web-mode de forma transparente (solo lo mostramos si VERBOSE)
    web_mode = args.web_mode
    if web_mode not in ("auto","on","off"): web_mode = "auto"
    if VERBOSE:
        log_info(f"web-mode: {web_mode}")

    messages: List[Dict[str,str]] = []
    messages.append({"role":"system","content": build_system_msg(web_mode)})

    total_prompt = 0
    total_completion = 0

    while True:
        if args.task:
            task = args.task.strip()
            args.task = None
        else:
            try:
                task = input("Enter task (or press Enter to exit): ").strip()
            except EOFError:
                break
        if not task:
            print("Exiting interactive session.")
            break

        messages.append({"role":"user","content": task})
        idle_empty = 0  # respuestas vacías seguidas
        finished = False

        for step in range(1, args.max_steps+1):
            log_info(f"=== Step {step} ===")
            try:
                data, usage = call_openai(messages, args.model, web_mode)
            except Exception as e:
                log_err(f"Error durante llamada API: {e}")
                break

            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

            commands = data.get("commands", [])
            explanation = (data.get("explanation", "") or "").strip()
            finished_flag = bool(data.get("finished", False))

            if (not commands) and (explanation == "") and not finished_flag:
                idle_empty += 1
                log_err("Modelo devolvió JSON vacío. Reintentando con recordatorio estricto.")
                messages.append({"role":"user","content":
                    "La respuesta anterior estaba vacía. Responde SOLO con JSON válido con las claves EXACTAS: "
                    "{\"commands\": [\"...\"], \"explanation\": \"...\", \"finished\": true|false}. "
                    "Incluye al menos un comando seguro o marca finished=true si ya has terminado."
                })
                if idle_empty >= 3:
                    log_err("Tres respuestas vacías seguidas. Abortando esta tarea para evitar bucle.")
                    break
                continue

            if explanation:
                log_info(f"AI explanation: {explanation}")

            # limitar comandos
            if isinstance(commands, list):
                if len(commands) > args.max_cmds:
                    log_info(f"Limiting commands from {len(commands)} to {args.max_cmds}")
                    commands = commands[:args.max_cmds]
            else:
                commands = []

            # filtrar peligrosos
            safe_cmds: List[str] = []
            blocked: List[str] = []
            for c in commands:
                if dangerous_command(c):
                    blocked.append(c)
                else:
                    safe_cmds.append(c)

            if blocked:
                messages.append({"role":"user","content":
                    "Se bloquearon comandos peligrosos por seguridad:\n" + "\n".join(blocked) +
                    "\nProporciona alternativas seguras o explica cómo proceder sin acciones destructivas."
                })
                # seguimos con los seguros (si hay)
            if not safe_cmds and not finished_flag:
                # sin comandos que ejecutar
                messages.append({"role":"user","content":
                    "No hay comandos seguros que ejecutar. Propón otra estrategia segura o marca finished=true si no hay nada más que hacer."
                })
                continue

            if finished_flag and not safe_cmds:
                finished = True
                break

            # Ejecutar como script para preservar entorno entre líneas
            rc, out, err = exec_commands_as_script(safe_cmds)
            if out:
                print("-- STDOUT --")
                print(out.rstrip())
            if err:
                print("-- STDERR --", file=sys.stderr)
                print(err.rstrip(), file=sys.stderr)

            # Añadir SOLO el resumen de ejecución (no reinyectar el JSON del asistente)
            out_summary = f"Commands:\n" + "\n".join(f"$ {c}" for c in safe_cmds) + \
                          f"\nReturn code: {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
            messages.append({"role":"user","content": out_summary})

            if rc != 0:
                # sugerir reconsideración
                messages.append({"role":"user","content":
                    "El retorno fue distinto de 0. No repitas el mismo comando; analiza el error y propone otra vía."
                })

            if finished_flag:
                finished = True
                break

        if finished:
            log_info("Task complete.")
        else:
            log_info("Task finished (loop ended).")

    if (total_prompt or total_completion):
        # estimación muy aproximada
        pricing = {
            "gpt-5-mini": {"input": 0.00013, "output": 0.00052},  # EJEMPLO; ajústalo según tu cuenta
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0003},
        }
        mk = args.model.lower()
        if mk in pricing:
            rates = pricing[mk]
            cost = (total_prompt * rates["input"] + total_completion * rates["output"]) / 1000.0
            print(f"\nApproximate API usage (chat): prompt_tokens={total_prompt}, completion_tokens={total_completion}.")
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(f"\nApproximate API usage (chat): prompt_tokens={total_prompt}, completion_tokens={total_completion}.")
            print(f"No pricing table for model '{args.model}'.")
if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

# ---- (opcional) breve informe ----------------------------------------------
echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORT'
# AI Agent (Debian)
- Respuestas vacías: detectadas, reintenta y corta a la 3ª (evita bucles).
- Historial limpio: no reinyecta el JSON del asistente por cada comando.
- Verbose RAW: `AI_AGENT_VERBOSE=1` imprime el bruto que devuelve el modelo.
- Normalización de comandos:
  - `apt-get`: no interactivo + banderas Dpkg seguras.
  - `curl`: `-fS --retry 2 --max-time 20` si falta.
  - `gpg --dearmor`: `--batch --yes` para evitar prompts.
  - `systemctl status`: `--no-pager`.
  - `mysql/mariadb/mysqldump`: añade `--defaults-file` si hay credenciales estándar.
- Ejecución por paso como script temporal (preserva variables).
- Detector de peligros mejorado; allowlist para limpiezas de APT.
- Web-mode `auto|on|off`; decisión transparente.
REPORT
${SUDO} chmod 644 "$DEST_DIR/report.md"

# ---- virtualenv & deps ------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  ${SUDO} python3 -m venv "$VENV_DIR"
fi
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
# openai 1.x (Responses API) + typing-extensions (por compat)
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir "openai>=1.50.0" >/dev/null

# ---- wrapper ---------------------------------------------------------------
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<'WRAP'
#!/usr/bin/env bash
# ai-agent wrapper

set -euo pipefail

CONF_DIR="/etc/ai-agent"
ENV_FILE="$CONF_DIR/agent.env"
AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PY="$VENV_DIR/bin/python"
AGENT="$AGENT_DIR/ai_server_agent.py"

default_model="${AI_AGENT_DEFAULT_MODEL:-gpt-5-mini}"
default_max_steps="${AI_AGENT_DEFAULT_MAX_STEPS:-24}"
default_max_cmds="${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:-24}"
default_web_mode="${AI_AGENT_WEB_MODE:-auto}"

usage() {
  cat <<EOF
ai-agent - Ejecuta el agente de automatización para Debian

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en $ENV_FILE
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent --help-agent            # Ayuda del agente Python
  ai-agent -h | --help             # Ayuda del wrapper

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  $ENV_FILE y, si hay TTY, permitirá introducirla y guardarla.
- Variables por defecto:
  AI_AGENT_DEFAULT_MODEL (por defecto: $default_model)
  AI_AGENT_DEFAULT_MAX_STEPS ($default_max_steps)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP ($default_max_cmds)
  AI_AGENT_WEB_MODE ($default_web_mode)
- Verbosidad: export AI_AGENT_VERBOSE=1 para debug detallado.
EOF
}

mask() {
  local x="${1:-}"
  if [ -z "$x" ]; then echo "(no-set)"; return; fi
  local len=${#x}
  if [ $len -le 6 ]; then echo "******"; else echo "${x:0:3}******${x: -3}"; fi
}

ensure_key() {
  if [ -n "${OPENAI_API_KEY:-}" ]; then return 0; fi
  if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  fi
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
    read -r -p "Introduce tu OPENAI_API_KEY (deja en blanco para cancelar): " key || true
    if [ -n "${key:-}" ]; then
      OPENAI_API_KEY="$key"
      read -r -p "¿Quieres guardarla en $ENV_FILE para futuros usos? [y/N]: " ans || true
      if [[ "${ans,,}" == "y" ]]; then
        mkdir -p "$CONF_DIR"
        umask 077
        printf 'OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY" > "$ENV_FILE"
        echo "[OK] Clave guardada en $ENV_FILE"
      fi
      export OPENAI_API_KEY
    fi
  fi
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERR] OPENAI_API_KEY no disponible." >&2
    exit 1
  fi
}

set_key() {
  local val="${1:-}"
  if [ -z "$val" ] && [ -t 0 ]; then
    read -r -p "Introduce tu OPENAI_API_KEY: " val || true
  fi
  if [ -z "$val" ]; then
    echo "[ERR] No se proporcionó clave." >&2; exit 1
  fi
  mkdir -p "$CONF_DIR"; umask 077
  printf 'OPENAI_API_KEY=%q\n' "$val" > "$ENV_FILE"
  echo "[OK] Clave guardada en $ENV_FILE"
}

show_key() {
  if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    echo "OPENAI_API_KEY=$(mask "${OPENAI_API_KEY:-}")"
  else
    echo "OPENAI_API_KEY=(no-set)"
  fi
}

clear_key() {
  if [ -f "$ENV_FILE" ]; then rm -f "$ENV_FILE"; echo "[OK] Borrada $ENV_FILE"; else echo "Nada que borrar."; fi
}

if [ $# -gt 0 ]; then
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --help-agent) ensure_key; exec "$PY" "$AGENT" --help-agent ;;
    --set-key) shift; set_key "${1:-}"; exit 0 ;;
    --show-key) show_key; exit 0 ;;
    --clear-key) clear_key; exit 0 ;;
  esac
fi

ensure_key
exec "$PY" "$AGENT" "$@"
WRAP
${SUDO} chmod +x "$WRAPPER"

echo "[SUCCESS] Installation complete."
echo

# ---- post-install key prompt (solo si hay TTY) ------------------------------
if [ -t 0 ]; then
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " ans || true
  if [[ "${ans,,}" == "y" ]]; then
    read -r -p "Introduce tu OPENAI_API_KEY: " KEY || true
    if [ -n "${KEY:-}" ]; then
      ${SUDO} mkdir -p "$CONF_DIR"
      umask 077
      echo "OPENAI_API_KEY=$(printf '%q' "$KEY")" | ${SUDO} tee "$ENV_FILE" >/dev/null
      ${SUDO} chmod 600 "$ENV_FILE"
      echo "[OK] Clave guardada en $ENV_FILE"
    fi
  fi
  echo
fi

echo "Listo. Prueba ahora:"
echo "  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\""
echo "  AI_AGENT_VERBOSE=1 ai-agent --task \"actualiza el sistema\""

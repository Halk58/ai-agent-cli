#!/bin/bash
# install_ai_agent.sh - Instalador/actualizador del agente de automatización para Debian (12/13)
#
# - Instala dependencias de Python, copia el agente en /opt/ai-agent,
#   crea un venv, instala la librería openai y genera el wrapper /usr/local/bin/ai-agent.
# - Auto-actualiza este propio instalador si hay una versión más nueva.
# - Opcionalmente guarda OPENAI_API_KEY de forma persistente en /etc/ai-agent/agent.env.
#
# Uso:
#   sudo bash install_ai_agent.sh
#
# Después:
#   ai-agent --help
#
# Notas:
#   * Ejecuta SIEMPRE los comandos del agente con /bin/bash -lc para evitar incompatibilidades con /bin/sh.
#   * El agente usa por defecto el modelo configurable por variable (ver sección VARIABLES POR DEFECTO).
#   * Maneja automáticamente la diferencia entre max_completion_tokens / max_tokens y conmutación a Responses API.
#
set -euo pipefail

###############################################################################
#                               VERSIONADO                                    #
###############################################################################
SCRIPT_VERSION="1.7.0"
# URL pública para auto-actualización de este mismo script
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

###############################################################################
#                            DETECCIÓN DE SUDO                                #
###############################################################################
if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

###############################################################################
#                         AUTO-ACTUALIZACIÓN DEL SCRIPT                       #
###############################################################################
_do_self_update() {
  local tmp="$(mktemp)"
  if curl -fsSL "$UPDATE_URL" -o "$tmp"; then
    local remote_version
    remote_version="$(grep -Eo 'SCRIPT_VERSION="[0-9.]+"' "$tmp" | head -n1 | cut -d'"' -f2)"
    if [ -n "${remote_version:-}" ]; then
      local newest
      newest="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
      if [ "$newest" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
        echo "[INFO] A newer version ($remote_version) is available. Updating self from $UPDATE_URL ..."
        local self
        self="$(readlink -f -- "$0")"
        if [ -w "$self" ]; then
          cp "$tmp" "$self"
        else
          ${SUDO} cp "$tmp" "$self"
        fi
        chmod +x "$self"
        echo "[INFO] Re-executing the updated installer..."
        exec "$self" "$@"
      fi
    fi
    rm -f "$tmp"
  else
    echo "[WARN] Unable to check for updates at $UPDATE_URL. Continuing with $SCRIPT_VERSION."
  fi
}
_do_self_update "$@"

###############################################################################
#                        VARIABLES Y RUTAS DE INSTALACIÓN                     #
###############################################################################
DEST_DIR="/opt/ai-agent"
VENV_DIR="$DEST_DIR/venv"
WRAPPER="/usr/local/bin/ai-agent"
CONF_DIR="/etc/ai-agent"
CONF_ENV="$CONF_DIR/agent.env"

# Valores por defecto (pueden modificarse luego en /etc/ai-agent/agent.env)
DEFAULT_MODEL="gpt-5-mini"
DEFAULT_MAX_STEPS=24
DEFAULT_MAX_CMDS_PER_STEP=24
DEFAULT_WEB_MODE="auto"    # auto|on|off

###############################################################################
#                         INSTALACIÓN DE DEPENDENCIAS                         #
###############################################################################
echo "[INFO] Updating package lists..."
${SUDO} apt-get update -y

echo "[INFO] Installing Python and required packages..."
${SUDO} apt-get install -y python3 python3-venv python3-pip curl ca-certificates gnupg

echo "[INFO] Creating installation directory at $DEST_DIR"
${SUDO} mkdir -p "$DEST_DIR"
${SUDO} mkdir -p "$CONF_DIR"

###############################################################################
#                         ESCRITURA DEL AGENTE (Python)                       #
###############################################################################
echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_server_agent.py — Agente autónomo para tareas en Debian con OpenAI

Características destacadas:
- Ejecuta SIEMPRE con /bin/bash -lc para evitar problemas de /bin/sh.
- Protocolo JSON estricto: {commands: [..], explanation: str, finished: bool}.
- Salvaguardas contra comandos peligrosos con lista de permitidos para limpiezas seguras.
- Endurecimiento de comandos (apt, dpkg, gpg) para evitar prompts (no-interactivo).
- Reintentos inteligentes de API:
  * Chat Completions → usa max_completion_tokens; si el modelo no lo soporta, cae a max_tokens.
  * Si el modelo exige Responses API, conmuta automáticamente.
  * Si el SDK no soporta "response_format" en Responses, reintenta sin él.
  * Evita forzar temperature; sólo se envía si el usuario la define.
- Decisión de web (auto|on|off) sin preguntar al usuario; se sugiere usar curl/HTTP sólo si es necesario.
- Timestamps legibles en cada paso y ejecución de comando.

ADVERTENCIA: Ejecutar comandos generados por un modelo conlleva riesgo. Úsese en entornos controlados.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

# ----------------------------- Configuración ------------------------------- #
ENV_DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
ENV_DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
ENV_DEFAULT_MAX_CMDS = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
ENV_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto|on|off
ENV_TEMPERATURE = os.getenv("AI_AGENT_TEMPERATURE", "")  # vacío = no enviar

# Presupuesto por paso (no todos los modelos aceptan estos parámetros)
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("AI_AGENT_MAX_COMPLETION_TOKENS", "2048"))

# Modelos que NO deben recibir temperature
MODELS_NO_TEMPERATURE = {
    "gpt-5-mini", "o4-mini", "o4-mini-high", "o3-mini", "o3"
}

# -------------------------- Utilidades de impresión ------------------------ #
def ts() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

def log_info(msg: str) -> None:
    print(f"{ts()} {msg}")

def log_err(msg: str) -> None:
    print(f"{ts()} {msg}", file=sys.stderr)

# ----------------------- Ejecución segura de comandos ---------------------- #
def run_shell_command(command: str) -> Tuple[int, str, str]:
    """
    Ejecuta SIEMPRE el comando con /bin/bash -lc para evitar parseo por /bin/sh.
    """
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)

_ALLOWLIST_SAFE = [
    # Limpieza segura de APT
    r"^rm\s+-rf\s+/var/lib/apt/lists/\*\s*$",
    r"^apt-get\s+clean\s*$",
    # Limpieza de /tmp
    r"^rm\s+-rf\s+/tmp/.*",
]

_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/$",
    r"\bmkfs(\.| )",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r":\(\)\s*{\s*:\|\s*:\s*;\s*}\s*;:\s*",  # fork bomb
]

def _matches(patterns: List[str], cmd: str) -> bool:
    for p in patterns:
        if re.search(p, cmd, re.IGNORECASE):
            return True
    return False

def is_dangerous_command(cmd: str) -> bool:
    """Bloquea comandos obviamente destructivos, con allowlist para limpiezas comunes."""
    cmd_s = cmd.strip()
    # allowlist primero
    if _matches(_ALLOWLIST_SAFE, cmd_s):
        return False
    # peligrosos
    if _matches(_DANGEROUS_PATTERNS, cmd_s):
        return True
    return False

def harden_command(cmd: str) -> str:
    """
    Endurece comandos comunes para evitar prompts:
      - apt-get install/upgrade/dist-upgrade → añade DEBIAN_FRONTEND, -y y dpkg options.
      - gpg --dearmor → fuerza --batch --yes
      - tee a /etc/* → si falta sudo y no somos root, deja igual (el wrapper suele ejecutarse como root)
    """
    s = cmd

    # apt-get install/upgrade
    if re.search(r"\bapt-get\s+(install|upgrade|dist-upgrade)\b", s):
        if "DEBIAN_FRONTEND=" not in s:
            s = "DEBIAN_FRONTEND=noninteractive " + s
        if " -y " not in s and not s.rstrip().endswith(" -y"):
            s = s.replace("apt-get ", "apt-get -y ")
        if "Dpkg::Options" not in s and "install" in s:
            s = s.replace(
                "apt-get -y ",
                "apt-get -y -o Dpkg::Options::='--force-confdef' -o Dpkg::Options::='--force-confold' "
            )

    # gpg --dearmor
    if re.search(r"\bgpg\s+.*\s--dearmor\b", s) and "--batch" not in s:
        s = s.replace("gpg ", "gpg --batch --yes ")

    return s

# ------------------------------ OpenAI Calls ------------------------------- #
# Compatibilidad con diferentes versiones del SDK.
try:
    import openai  # type: ignore
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

def _extract_responses_text(resp: Any) -> str:
    """
    Extrae texto de Responses API para diferentes variantes del SDK.
    """
    # SDK moderno: output_text
    if hasattr(resp, "output_text"):
        return resp.output_text
    # Variante choices-like
    try:
        return resp.choices[0].message["content"][0]["text"]
    except Exception:
        pass
    # Variante outputs list
    try:
        parts = []
        for out in getattr(resp, "output", []):
            for c in out.get("content", []):
                t = c.get("text", {}).get("value")
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    # Último recurso: repr
    return json.dumps(getattr(resp, "dict", lambda: {} )(), ensure_ascii=False)

def _json_loads_safe(raw: str) -> Dict[str, Any]:
    """
    Parsea JSON; si falla, intenta reparar extrayendo el primer bloque {...} balanceado.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass
    # intentar extraer bloque
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    # como último recurso, devolver mínima estructura
    return {"commands": [], "explanation": raw.strip(), "finished": False}

def _chat_call(messages, model, max_out_tokens, use_mct=True, send_temp=False):
    kwargs = {
        "model": model,
        "messages": messages,
    }
    # JSON mode (muchos modelos lo aceptan en Chat)
    kwargs["response_format"] = {"type": "json_object"}
    if send_temp:
        kwargs["temperature"] = float(ENV_TEMPERATURE)
    if use_mct:
        kwargs["max_completion_tokens"] = max_out_tokens
    else:
        kwargs["max_tokens"] = max_out_tokens
    return openai.chat.completions.create(**kwargs)

def _responses_call(messages, model, max_out_tokens, try_mct=True, send_temp=False):
    # Responses API
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError("Responses API client no disponible: {}".format(e))
    base = {
        "model": model,
        "input": messages,  # soporta mensajes estilo chat en algunos SDK
    }
    # algunos SDK no aceptan response_format → lo intentamos y si da TypeError, reintentamos sin él
    if send_temp:
        base["temperature"] = float(ENV_TEMPERATURE)
    # Intentos de límite de tokens
    def try_create(payload):
        try:
            return client.responses.create(**payload)
        except TypeError as e:
            em = str(e)
            if "response_format" in em:
                payload.pop("response_format", None)
                return client.responses.create(**payload)
            raise
    # Primero con response_format si es posible (para forzar JSON)
    payload = dict(base)
    payload["response_format"] = {"type": "json_object"}
    if try_mct:
        payload["max_completion_tokens"] = max_out_tokens
        try:
            return try_create(payload)
        except Exception as e1:
            em = str(e1)
            if "max_completion_tokens" in em:
                payload = dict(base)
                payload["response_format"] = {"type": "json_object"}
                payload["max_output_tokens"] = max_out_tokens
                return try_create(payload)
            # si el fallo no es por ese parámetro, probamos sin response_format
            payload = dict(base)
            payload["max_output_tokens"] = max_out_tokens
            return try_create(payload)
    else:
        payload["max_output_tokens"] = max_out_tokens
        return try_create(payload)

def call_openai(messages: List[Dict[str, str]], model: str, max_out_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Estrategia robusta:
      1) Chat + max_completion_tokens (sin temperature por defecto).
      2) Si error sugiere Responses API → conmutar a Responses.
      3) Si 'max_completion_tokens' no soportado → usar 'max_tokens' en Chat.
      4) Si el SDK de Responses no acepta 'response_format' → reintenta sin él.
    """
    if not HAVE_OPENAI:
        raise RuntimeError("La librería 'openai' no está instalada en el venv.")

    send_temp = False
    if ENV_TEMPERATURE and model not in MODELS_NO_TEMPERATURE:
        try:
            float(ENV_TEMPERATURE)
            send_temp = True
        except Exception:
            send_temp = False

    # 1) Chat
    try:
        resp = _chat_call(messages, model, max_out_tokens, use_mct=True, send_temp=send_temp)
        content = resp.choices[0].message.content
        data = _json_loads_safe(content)
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
        return data, usage_dict
    except Exception as e:
        emsg = str(e)
        # peticiones típicas de conmutación
        if "Responses API" in emsg or "use the Responses API" in emsg or "not supported with the Chat Completions API" in emsg:
            try:
                r = _responses_call(messages, model, max_out_tokens, try_mct=True, send_temp=send_temp)
                text = _extract_responses_text(r)
                data = _json_loads_safe(text)
                usage = getattr(r, "usage", None)
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
                return data, usage_dict
            except Exception as e2:
                raise RuntimeError(f"OpenAI API (Responses) error: {e2}") from e2

        # Si no soporta max_completion_tokens → usar max_tokens en Chat
        if "max_completion_tokens" in emsg and ("Unsupported" in emsg or "unsupported" in emsg or "parameter" in emsg.lower()):
            try:
                resp = _chat_call(messages, model, max_out_tokens, use_mct=False, send_temp=send_temp)
                content = resp.choices[0].message.content
                data = _json_loads_safe(content)
                usage = getattr(resp, "usage", None)
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
                return data, usage_dict
            except Exception as e4:
                # Último recurso → Responses sin response_format
                try:
                    r = _responses_call(messages, model, max_out_tokens, try_mct=False, send_temp=send_temp)
                    text = _extract_responses_text(r)
                    data = _json_loads_safe(text)
                    usage = getattr(r, "usage", None)
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                    } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
                    return data, usage_dict
                except Exception as e5:
                    raise RuntimeError(f"OpenAI API call failed: {e5}") from e5

        # Último recurso directo → Responses
        try:
            r = _responses_call(messages, model, max_out_tokens, try_mct=True, send_temp=send_temp)
            text = _extract_responses_text(r)
            data = _json_loads_safe(text)
            usage = getattr(r, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
            return data, usage_dict
        except Exception as e6:
            raise RuntimeError(f"OpenAI API call failed: {e6}") from e6

# ----------------------------- Prompt del sistema -------------------------- #
def build_system_prompt(web_mode: str, max_cmds: int) -> str:
    web_note = (
        "Web mode: OFF. Evita comandos de red salvo que sean imprescindibles.\n"
        if web_mode == "off" else
        "Web mode: ON/AUTO. Si necesitas información externa, decide UNA VEZ al principio y limita las peticiones HTTP (curl/wget) a un máximo de 3, explicando tu plan.\n"
    )
    return (
        "Eres un agente de automatización en Debian 12/13.\n"
        "Convierte instrucciones del usuario en comandos de shell. Tras cada paso verás la salida y decidirás el siguiente.\n"
        f"{web_note}"
        "Responde EXCLUSIVAMENTE con JSON (no texto suelto) con la siguiente forma:\n"
        "{\n"
        '  "commands": ["<cmd1>", "<cmd2>", ...],   // máximo ' + str(max_cmds) + " por paso\n"
        '  "explanation": "resumen breve de lo que harás",\n'
        '  "finished": false // o true cuando termines\n'
        "}\n"
        "Reglas IMPORTANTES:\n"
        "- Usa SIEMPRE comandos compatibles con Debian. Evita banderas interactivas; para APT usa DEBIAN_FRONTEND=noninteractive, -y y opciones de dpkg para preservar config.\n"
        "- Cuando crees/reescribas ficheros de claves GPG usa: gpg --batch --yes --dearmor -o <destino> <origen>.\n"
        "- Si concatenas muchas acciones, empaquétalas en un ÚNICO comando: bash -lc '...'; usa comillas simples externas y dobles internas si necesitas variables.\n"
        "- Evita comandos potencialmente destructivos (rm -rf /, mkfs, dd, shutdown/reboot, etc.).\n"
        "- Si un comando falla, no repitas lo mismo; analiza el error y adapta el plan.\n"
        "- Si detectas que un valor de variable no persistirá entre comandos separados, agrupa todo en un único bash -lc '...' para que funcione.\n"
        "- Justifica brevemente en 'explanation' y marca finished=true cuando completes la tarea.\n"
    )

# ------------------------------- Bucle principal --------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agente autónomo para convertir instrucciones en comandos de shell (Debian 12/13)."
    )
    parser.add_argument("--task", type=str, help="Instrucción inicial (opcional).")
    parser.add_argument("--model", type=str, default=ENV_DEFAULT_MODEL, help=f"Modelo a usar (por defecto: {ENV_DEFAULT_MODEL}).")
    parser.add_argument("--max-steps", type=int, default=ENV_DEFAULT_MAX_STEPS, help="Pasos máximos por tarea.")
    parser.add_argument("--max-cmds-per-step", type=int, default=ENV_DEFAULT_MAX_CMDS, help="Número máximo de comandos por paso.")
    parser.add_argument("--web-mode", type=str, choices=["auto","on","off"], default=ENV_WEB_MODE, help=f"Decisión de red (por defecto: {ENV_WEB_MODE}).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        log_err("OPENAI_API_KEY no está definido; configúralo antes de ejecutar.")
        sys.exit(1)

    messages: List[Dict[str, str]] = []
    system_prompt = build_system_prompt(args.web_mode, args.max_cmds_per_step)
    messages.append({"role": "system", "content": system_prompt})

    total_prompt_tokens = 0
    total_completion_tokens = 0

    pending_task = args.task.strip() if args.task else None

    log_info("Agent starting (interactive).")

    while True:
        if pending_task is not None:
            user_task = pending_task
            pending_task = None
        else:
            try:
                user_task = input("Enter task (or press Enter to exit): ").strip()
            except KeyboardInterrupt:
                print()
                break
        if not user_task:
            log_info("Exiting interactive session.")
            break

        messages.append({"role": "user", "content": user_task})

        failed_commands_count: Dict[str, int] = {}
        command_errors: Dict[str, str] = {}
        finished = False

        for step in range(1, args.max_steps + 1):
            log_info(f"=== Step {step} ===")
            try:
                data, usage = call_openai(messages, args.model, DEFAULT_MAX_COMPLETION_TOKENS)
            except Exception as e:
                log_err(f"Error durante llamada API: {e}")
                break

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            if not isinstance(data, dict):
                log_err(f"Estructura JSON inesperada: {data!r}")
                break

            commands = data.get("commands", [])
            explanation = data.get("explanation", "")
            finished_flag = bool(data.get("finished", False))

            if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
                log_err(f"'commands' debe ser lista de strings. Recibido: {commands!r}")
                break

            # Limitar nº de comandos
            if len(commands) > args.max_cmds_per_step:
                log_info(f"Limiting commands from {len(commands)} to {args.max_cmds_per_step}")
                commands = commands[:args.max_cmds_per_step]

            print(f"{ts()} AI explanation: {explanation}")
            if commands:
                print(f"{ts()} Proposed commands:")
                for c in commands:
                    print(f"  $ {c}")

            # Ejecutar comandos
            skip_next_round = False
            for idx, raw_cmd in enumerate(commands, start=1):
                # endurecer y filtrar peligros
                cmd = harden_command(raw_cmd)
                if cmd in failed_commands_count:
                    prev_err = command_errors.get(cmd, "")
                    msg = (
                        f"La orden ya falló antes:\n{cmd}\n"
                        f"Error previo:\n{prev_err}\n"
                        "Propón una alternativa o explica la imposibilidad."
                    )
                    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
                    messages.append({"role": "user", "content": msg})
                    skip_next_round = True
                    break
                if is_dangerous_command(cmd):
                    msg = f"Bloqueado comando peligroso: {cmd}"
                    log_err(msg)
                    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
                    messages.append({"role": "user", "content": msg})
                    skip_next_round = True
                    break

                log_info(f"Executing command {idx}/{len(commands)}: {cmd}")
                rc, out, err = run_shell_command(cmd)
                if out:
                    print("-- STDOUT --")
                    print(out.rstrip())
                if err:
                    print("-- STDERR --", file=sys.stderr)
                    print(err.rstrip(), file=sys.stderr)
                # feedback
                messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
                out_summary = f"Command: {cmd}\nReturn code: {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                messages.append({"role": "user", "content": out_summary})

                if rc != 0:
                    failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                    command_errors[cmd] = (err or out or "").strip()

            if skip_next_round:
                continue

            if finished_flag:
                print(f"\n{ts()} Task complete.\nSummary:\n{explanation}")
                finished = True
                break

            time.sleep(0.5)

        if not finished and step >= args.max_steps:
            log_err("Se alcanzó el máximo de pasos; la tarea puede no estar completa.")

    # Informe de uso (si el SDK lo proporcionó)
    if total_prompt_tokens or total_completion_tokens:
        print(f"\n{ts()} Approximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
        # Tabla simple (orientativa)
        PRICING = {
            "gpt-5-mini": {"input": 0.00015, "output": 0.0006},  # ejemplo orientativo
            "o4-mini": {"input": 0.001, "output": 0.003},
        }
        key = args.model.lower()
        if key in PRICING:
            rates = PRICING[key]
            cost = (total_prompt_tokens * rates["input"] + total_completion_tokens * rates["output"]) / 1000.0
            print(f"{ts()} Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(f"{ts()} No pricing table for model '{args.model}'.")
    return

if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 0644 "$DEST_DIR/ai_server_agent.py"

###############################################################################
#                         DOCUMENTO DE REFERENCIA                             #
###############################################################################
echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORT'
# AI Agent (Debian 12/13)

- Ejecuta comandos generados por un modelo OpenAI con bucle de razonamiento.
- JSON estricto (commands, explanation, finished).
- Ejecución con **/bin/bash -lc** siempre (evita errores de /bin/sh).
- Endurecimiento apt/gpg para no interactivo.
- Conmutación automática entre Chat Completions y Responses API.
- Impresiones con timestamp en cada paso/ejecución.

### Variables por defecto (puedes cambiarlas en `/etc/ai-agent/agent.env`)
- `AI_AGENT_DEFAULT_MODEL` (por defecto: `gpt-5-mini`)
- `AI_AGENT_DEFAULT_MAX_STEPS` (24)
- `AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP` (24)
- `AI_AGENT_WEB_MODE` (`auto|on|off`, por defecto `auto`)
- `AI_AGENT_TEMPERATURE` (vacío = no enviar)

### Wrapper
- `ai-agent --set-key [API_KEY]` guarda la clave en `/etc/ai-agent/agent.env`.
- `--clear-key`, `--show-key`, `--help`, `--help-agent`.
REPORT
${SUDO} chmod 0644 "$DEST_DIR/report.md"

###############################################################################
#                         CREACIÓN DEL VENV + OPENAI                          #
###############################################################################
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  ${SUDO} python3 -m venv "$VENV_DIR"
fi

echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
# openai >=1.0 (SDK moderno). Evitamos fijar una versión exacta para compatibilidad amplia.
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir openai >/dev/null

###############################################################################
#                                WRAPPER CLI                                  #
###############################################################################
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<'WRAP'
#!/bin/bash
# ai-agent — Wrapper de ejecución y utilidades de clave para el agente

set -euo pipefail

AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PY_BIN="$VENV_DIR/bin/python"
AGENT_PY="$AGENT_DIR/ai_server_agent.py"
CONF_DIR="/etc/ai-agent"
CONF_ENV="$CONF_DIR/agent.env"

# Valores por defecto (si no están en el entorno ni en agent.env)
: "${AI_AGENT_DEFAULT_MODEL:=gpt-5-mini}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=24}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=24}"
: "${AI_AGENT_WEB_MODE:=auto}"  # auto|on|off

usage() {
  cat <<EOF
ai-agent - Ejecuta el agente de automatización para Debian

Uso:
  ai-agent [opciones del agente] --task "instrucción"
  ai-agent --set-key [API_KEY]     # Guarda la API en /etc/ai-agent/agent.env
  ai-agent --clear-key             # Borra la API persistida
  ai-agent --show-key              # Muestra la API (enmascarada)
  ai-agent -h | --help             # Ayuda del wrapper (no requiere API)

Notas:
- Si OPENAI_API_KEY no está en el entorno, el wrapper intentará cargarla desde
  /etc/ai-agent/agent.env y, si hay TTY, permitirá guardarla.
- Variables por defecto:
  AI_AGENT_DEFAULT_MODEL (por defecto: ${AI_AGENT_DEFAULT_MODEL})
  AI_AGENT_DEFAULT_MAX_STEPS (${AI_AGENT_DEFAULT_MAX_STEPS})
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP (${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP})
  AI_AGENT_WEB_MODE (${AI_AGENT_WEB_MODE})
- Para ver ayuda del agente Python:
  ai-agent --help-agent
EOF
}

mask() {
  local s="$1"
  if [ -z "${s:-}" ]; then echo "(vacío)"; return; fi
  local n=${#s}
  if [ $n -le 6 ]; then echo "***"; else echo "${s:0:3}***${s: -3}"; fi
}

ensure_env_file() {
  mkdir -p "$CONF_DIR"
  touch "$CONF_ENV"
  chmod 0600 "$CONF_ENV"
}

save_key() {
  ensure_env_file
  local key="$1"
  if [ -z "$key" ]; then
    # pedir por TTY
    if [ -t 0 ]; then
      read -r -s -p "Introduce tu OPENAI_API_KEY: " key
      echo
    else
      echo "[ERR] No se ha proporcionado API key y no hay TTY." >&2
      exit 1
    fi
  fi
  if ! grep -q '^OPENAI_API_KEY=' "$CONF_ENV" 2>/dev/null; then
    echo "OPENAI_API_KEY=$key" >> "$CONF_ENV"
  else
    # reemplazo idempotente
    sed -i "s|^OPENAI_API_KEY=.*$|OPENAI_API_KEY=$key|" "$CONF_ENV"
  fi
  echo "[OK] Clave guardada en $CONF_ENV"
}

clear_key() {
  if [ -f "$CONF_ENV" ]; then
    sed -i '/^OPENAI_API_KEY=/d' "$CONF_ENV"
    echo "[OK] Clave eliminada de $CONF_ENV"
  else
    echo "[INFO] No existe $CONF_ENV"
  fi
}

show_key() {
  if [ -f "$CONF_ENV" ] && grep -q '^OPENAI_API_KEY=' "$CONF_ENV"; then
    local line
    line="$(grep '^OPENAI_API_KEY=' "$CONF_ENV" | head -n1 | cut -d'=' -f2-)"
    echo "OPENAI_API_KEY=$(mask "$line")"
  else
    echo "OPENAI_API_KEY no está guardada."
  fi
}

if [ "${1:-}" = "--set-key" ]; then
  save_key "${2:-}"
  exit 0
elif [ "${1:-}" = "--clear-key" ]; then
  clear_key
  exit 0
elif [ "${1:-}" = "--show-key" ]; then
  show_key
  exit 0
elif [ "${1:-}" = "--help-agent" ]; then
  exec "$PY_BIN" "$AGENT_PY" --help
elif [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

# Cargar variables persistidas si existen
if [ -f "$CONF_ENV" ]; then
  # shellcheck disable=SC1090
  . "$CONF_ENV"
fi

# Asegurar OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ]; then
  if [ -f "$CONF_ENV" ] && grep -q '^OPENAI_API_KEY=' "$CONF_ENV"; then
    # shellcheck disable=SC1090
    . "$CONF_ENV"
  fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  if [ -t 0 ]; then
    echo "No se encontró OPENAI_API_KEY."
    read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " yn
    case "${yn:-N}" in
      y|Y)
        save_key ""
        # recargar
        # shellcheck disable=SC1090
        . "$CONF_ENV"
        ;;
      *)
        echo "[ERR] Falta OPENAI_API_KEY." >&2
        exit 1
        ;;
    esac
  else
    echo "[ERR] Falta OPENAI_API_KEY (sin TTY para pedirla)." >&2
    exit 1
  fi
fi

# Exportar defaults al entorno si no están definidos (para el agente Python)
export AI_AGENT_DEFAULT_MODEL="${AI_AGENT_DEFAULT_MODEL}"
export AI_AGENT_DEFAULT_MAX_STEPS="${AI_AGENT_DEFAULT_MAX_STEPS}"
export AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP}"
export AI_AGENT_WEB_MODE="${AI_AGENT_WEB_MODE}"

# Ejecutar el agente
exec "$PY_BIN" "$AGENT_PY" "$@"
WRAP
${SUDO} chmod +x "$WRAPPER"

###############################################################################
#                        OFRECER GUARDAR OPENAI_API_KEY                       #
###############################################################################
if [ -t 0 ]; then
  echo
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " yn
  if [[ "${yn:-N}" =~ ^[yY]$ ]]; then
    read -r -s -p "Introduce tu OPENAI_API_KEY: " key
    echo
    if [ -n "$key" ]; then
      ${SUDO} mkdir -p "$CONF_DIR"
      ${SUDO} touch "$CONF_ENV"
      ${SUDO} chmod 0600 "$CONF_ENV"
      if grep -q '^OPENAI_API_KEY=' "$CONF_ENV" 2>/dev/null; then
        ${SUDO} sed -i "s|^OPENAI_API_KEY=.*$|OPENAI_API_KEY=$key|" "$CONF_ENV"
      else
        echo "OPENAI_API_KEY=$key" | ${SUDO} tee -a "$CONF_ENV" >/dev/null
      fi
      echo "[OK] Clave guardada en $CONF_ENV"
      echo
      echo "Listo. Prueba ahora:"
      echo "  ai-agent --help"
    else
      echo "[INFO] No se ha introducido ninguna clave."
    fi
  fi
fi

echo "[SUCCESS] Installation complete."

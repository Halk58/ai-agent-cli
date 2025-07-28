#!/bin/bash
# install_ai_agent.sh - Deploy the AI server agent on a Debian 12 system
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

# Versioning and update mechanism
#
# The following variables control the auto‑update behaviour of this script.  You can
# modify `SCRIPT_VERSION` to reflect the current version and set `UPDATE_URL` to
# point to a remote location where the latest version of this script is hosted.
# When this script starts it will attempt to download the remote script and
# compare its version to the local `SCRIPT_VERSION`.  If the remote version is
# newer, the script will replace itself and re‑execute using the updated copy.
SCRIPT_VERSION="1.0.0"
# Set UPDATE_URL to the address of the latest install script.  Leave empty to
# disable auto‑update.  Example:
# UPDATE_URL="https://example.com/install_ai_agent.sh"
UPDATE_URL=""

# Determine whether sudo is available.  Some minimal containers do not have sudo
# installed.  If sudo is missing, fall back to running commands directly.
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# Auto‑update: fetch the script at UPDATE_URL and compare versions
if [ -n "$UPDATE_URL" ]; then
    tmp_update_file="$(mktemp)"
    if curl -fsSL "$UPDATE_URL" -o "$tmp_update_file"; then
        # Extract remote version from the downloaded file
        remote_version="$(grep -Eo 'SCRIPT_VERSION="[0-9.]+"' "$tmp_update_file" | head -n1 | cut -d'"' -f2)"
        if [ -n "$remote_version" ]; then
            # Determine which version is newer using sort -V
            newest_version="$(printf '%s\n%s' "$SCRIPT_VERSION" "$remote_version" | sort -V | tail -n1)"
            if [ "$newest_version" = "$remote_version" ] && [ "$SCRIPT_VERSION" != "$remote_version" ]; then
                echo "[INFO] A newer version ($remote_version) is available at $UPDATE_URL. Updating..."
                script_path="$(readlink -f -- "$0")"
                # Replace the current script with the downloaded one. Use sudo if needed.
                if [ -w "$script_path" ]; then
                    cp "$tmp_update_file" "$script_path"
                else
                    ${SUDO} cp "$tmp_update_file" "$script_path"
                fi
                chmod +x "$script_path"
                echo "[INFO] Executing the updated script..."
                exec "$script_path" "$@"
            fi
        fi
        rm -f "$tmp_update_file"
    else
        echo "[WARN] Unable to check for updates at $UPDATE_URL. Proceeding with current version."
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

This script implements a simple autonomous agent that runs on a Debian‑based server
and uses an OpenAI model to translate natural‑language requests into shell
commands.  The agent follows a loop: it asks the language model for the next
command(s) to run, executes them, captures their output, and feeds the output
back into the conversation.  The cycle repeats until the model signals that
the task is complete.

Features:
  • Supports multi‑step tasks – the model can refine its plan after seeing
    command output and issue additional commands until the goal is reached.
  • Executes commands safely using the system shell and returns stdout/stderr
    to the model.
  • Uses the OpenAI Chat Completions API with JSON mode to ensure the model
    returns structured data.  The agent enforces strict JSON and handles
    parsing errors gracefully.
  • Contains simple safeguards to prevent obviously destructive commands.

Usage:
  python3 ai_server_agent.py --task "install nginx and start it"

You must set an environment variable called OPENAI_API_KEY with your OpenAI
API key before running this script.  On first run you may also specify
`--model` to choose a particular model (default: gpt‑4o).  See the README
or comments in this file for further details.

WARNING: Running shell commands generated by an AI model can be dangerous.
Always review commands and run this software on non‑critical systems or
containers.  The `ai‑shell‑agent` package warns that AI can "generate wrong
and possibly destructive commands" and advises users to be mindful of
dangerous operations【114342969493772†L211-L213】.  While this agent includes
basic safety checks, it does not guarantee complete protection against
harmful commands.  Use at your own risk.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import openai  # type: ignore
except ImportError:
    print("The 'openai' package is not installed. Please install it with 'pip install openai'", file=sys.stderr)
    sys.exit(1)


def run_shell_command(command: str) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr).

    The command is executed with `shell=True` so that compound commands work.
    A timeout is not enforced here; long‑running commands will block until
    completion.  You may wish to extend this to include timeouts or resource
    limits.
    """
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        # In case the subprocess module raises unexpected exceptions
        return 1, "", str(e)


def is_dangerous_command(cmd: str) -> bool:
    """Simple heuristic to block obviously destructive commands.

    Returns True if the command appears to be inherently dangerous.  This
    check is intentionally conservative – it blocks patterns like 'rm -rf /',
    shutdown/reboot, and formatting commands.  You should extend this list to
    reflect your threat model.
    """
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf\"",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "halt",
        ":(){ :|:& };:",  # fork bomb
    ]
    lower = cmd.lower().strip()
    for pattern in dangerous_patterns:
        if pattern in lower:
            return True
    return False


def call_openai_chat(messages: List[Dict[str, str]], model: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call the OpenAI Chat Completion API and return parsed JSON content and token usage.

    The API is called in JSON mode.  Some experimental models (for example
    certain mini variants) do not allow overriding the temperature.  In such
    cases the call is retried without specifying a temperature, which causes
    the model to use its default.  The second return value is a dictionary
    containing token usage metadata (prompt and completion tokens) when
    available.
    """
    try:
        # First attempt with temperature=0 for deterministic behaviour
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        msg = str(exc)
        # If the error indicates temperature is unsupported, retry without it
        if "temperature" in msg.lower() and "unsupported" in msg.lower():
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception:
                raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
        else:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}\nRaw content:\n{content}") from e
    usage = getattr(response, "usage", None)
    if usage:
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
    else:
        usage_dict = {"prompt_tokens": 0, "completion_tokens": 0}
    return data, usage_dict


def build_system_prompt() -> str:
    """Return the system prompt used to instruct the language model.

    The prompt explains the agent's role and defines the JSON structure for
    responses.  It uses a Reason‑and‑Act (ReACT) style where the model must
    reason step‑by‑step, choose commands to run, and set the 'finished'
    flag when the task is complete.
    """
    return (
        "You are a server automation agent running on a Debian 12 Linux system.\n"
        "Your job is to take high‑level instructions from the user and translate them\n"
        "into a sequence of shell commands that achieve the user's goals.  After each\n"
        "command is executed you will receive its output.  Use that output to decide\n"
        "what to do next.  Continue until the overall task is complete.\n"
        "\n"
        "Follow this protocol:\n"
        "1. Think step by step about what needs to happen next.\n"
        "2. Respond **only** with a JSON object.  The JSON must have these keys:\n"
        "   - commands: an array of one or more shell commands to run next.\n"
        "   - explanation: a short explanation of what these commands do.  Keep it concise.\n"
        "   - finished: a boolean that is true when the overall task has been completed;\n"
        "     otherwise false.\n"
        "3. Do not wrap the JSON in markdown or text.  The client will parse it\n"
        "   programmatically.\n"
        "4. Use Debian‑friendly commands and avoid interactive flags when possible.\n"
        "5. Never propose dangerous or destructive operations (e.g. do not use\n"
        "   'rm -rf /', 'shutdown', 'reboot', 'mkfs', 'dd', etc.).\n"
        "6. If a command fails, do not repeat exactly the same command.\n"
        "   Instead, analyse the error and suggest a different approach or explain\n"
        "   why the task cannot be completed.\n"
        "7. When finished, set finished=true and summarise in the explanation what was done."
    )


def main() -> None:
    """Entry point for the CLI.  Supports interactive sessions.

    On launch this function parses command‑line arguments and prepares the
    conversation with the system prompt.  It then enters a loop: for each
    task provided by the user (either via --task or interactively) it
    delegates to the language model to propose commands, executes them,
    collects the outputs, and feeds them back to the model.  After each
    task completes (or the maximum number of reasoning steps is reached)
    the user is invited to enter another instruction.  The conversation
    context (messages) persists across tasks so that follow‑up requests
    can build on previous results.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous server agent that uses OpenAI's API to convert natural\n"
            "language tasks into shell commands and executes them on a Debian system."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Initial natural language instruction describing what you want the agent to do.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of reasoning cycles per task before stopping (to prevent infinite loops).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.  Please set your API key.",
            file=sys.stderr,
        )
        sys.exit(1)
    openai.api_key = api_key

    # Initialize the conversation once with the system prompt
    messages: List[Dict[str, str]] = []
    system_prompt = build_system_prompt()
    messages.append({"role": "system", "content": system_prompt})

    # Track token usage across the entire session to estimate cost
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Use Optional type for compatibility with Python <3.10
    from typing import Optional
    pending_task: Optional[str] = args.task if args.task else None

    while True:
        # Acquire the next task from the user
        if pending_task is not None:
            user_task = pending_task.strip()
            pending_task = None
        else:
            try:
                user_task = input("Enter next task (or press Enter to exit): ").strip()
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            # Exit if the user provides an empty string or explicit exit commands
            if not user_task or user_task.lower() in {"exit", "quit", "q"}:
                print("Exiting interactive session.")
                break
        if not user_task:
            print("No task provided. Skipping.")
            continue
        # Append the user's instruction to the conversation history
        messages.append({"role": "user", "content": user_task})

        # Reset per-task tracking of failed commands
        failed_commands_count: Dict[str, int] = {}
        command_errors: Dict[str, str] = {}
        finished = False

        # Reasoning loop per task
        for step_num in range(1, args.max_steps + 1):
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
            # Validate commands list
            if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
                print("'commands' must be a list of strings. Received:", commands, file=sys.stderr)
                finished = True
                break
            print(f"\n=== Step {step_num} ===")
            print(f"AI explanation: {explanation}")
            if commands:
                print("Proposed commands:")
                for cmd in commands:
                    print(f"  $ {cmd}")
            else:
                print("No commands proposed.")

            skip_remaining = False
            for idx, cmd in enumerate(commands, start=1):
                if cmd in failed_commands_count:
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
                if is_dangerous_command(cmd):
                    msg = f"Blocked dangerous command: {cmd}"
                    print(msg)
                    messages.append({"role": "assistant", "content": json.dumps(data)})
                    messages.append({"role": "user", "content": msg})
                    skip_remaining = True
                    break
                print(f"\nExecuting command {idx}/{len(commands)}: {cmd}")
                returncode, stdout, stderr = run_shell_command(cmd)
                if stdout:
                    print("-- STDOUT --")
                    print(stdout.rstrip())
                if stderr:
                    print("-- STDERR --")
                    print(stderr.rstrip(), file=sys.stderr)
                messages.append({"role": "assistant", "content": json.dumps(data)})
                output_summary = (
                    f"Command: {cmd}\n"
                    f"Return code: {returncode}\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}"
                )
                messages.append({"role": "user", "content": output_summary})
                if returncode != 0:
                    failed_commands_count[cmd] = failed_commands_count.get(cmd, 0) + 1
                    command_errors[cmd] = stderr.strip() or stdout.strip()
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
        # After completing the current task we loop back to ask for the next task

    # Upon exiting the interactive loop, report token usage and estimated cost
    if total_prompt_tokens or total_completion_tokens:
        pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0003},
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
                f"\nToken usage: prompt={total_prompt_tokens}, completion={total_completion_tokens}."
                f"  Cost calculation not implemented for model '{args.model}'."
            )


if __name__ == "__main__":
    main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

# Write the report file alongside the agent for reference
echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# Agente autónomo para administrar un servidor Debian con OpenAI

## Antecedentes y consideraciones

Para convertir órdenes en lenguaje natural en acciones sobre un servidor es posible usar un agente que llame a la API de OpenAI y que ejecute comandos de *shell* en ciclos.  Existen herramientas de línea de comandos como **shellgpt** que generan órdenes y piden confirmación del usuario antes de ejecutarlas.  Su `README` explica que, al usar la opción `--shell`, genera un comando adecuado para el sistema operativo y solicita `[E]xecute`, `[D]escribe` o `[A]bort`【206575835293472†L338-L363】.  Este tipo de herramientas requieren que el usuario revise las órdenes por motivos de seguridad.  Igualmente, proyectos como **ai‑shell‑agent** advierten que las herramientas basadas en LLM pueden generar comandos erróneos o destructivos y recomiendan usarlas bajo su propio riesgo【114342969493772†L209-L213】.  En su documentación se indica que el agente puede ejecutar y depurar comandos en varios pasos, utilizando el patrón ReACT para iterar hasta completar la tarea【114342969493772†L184-L189】.

La implantación que se propone se inspira en estos enfoques, pero automatiza la ejecución de comandos sin intervención humana.  Se han incorporado salvaguardas básicas para bloquear órdenes claramente peligrosas (como `rm -rf /` o `shutdown`), aunque se recomienda ejecutar el agente en un entorno de pruebas y **nunca** en sistemas críticos.  El script necesita disponer de una clave de API válida; el usuario debe exportarla en la variable `OPENAI_API_KEY`.

## Instalación en Debian 12

En este script se incluye un instalador que prepara el entorno en un servidor Debian 12.  El script:

1. Actualiza los índices de paquetes e instala Python 3 con `pip` y el módulo `venv`.
2. Crea el directorio `/opt/ai‑agent`, escribe el archivo `ai_server_agent.py` y el presente informe, y prepara un entorno virtual en `venv`.
3. Instala la biblioteca `openai` en el entorno virtual.
4. Genera un ejecutable `/usr/local/bin/ai‑agent` que encapsula el entorno virtual y lanza el agente.
5. Solicita la clave de API al usuario al final de la instalación y, si se introduce, ejecuta el agente.

El instalador detecta si `sudo` está disponible; en contenedores sin `sudo` ejecutará las órdenes directamente.  Se debe ejecutar con privilegios suficientes para instalar paquetes y escribir en `/opt` y `/usr/local/bin`.

## Funcionamiento del agente

El fichero `ai_server_agent.py` implementa la lógica de la conversación con OpenAI y la ejecución de comandos.  Su funcionamiento se resume en estos pasos:

1. **Contexto inicial**.  Establece un mensaje de sistema que explica al modelo su rol: debe recibir una instrucción del usuario, razonar paso a paso qué hacer y devolver un objeto JSON con tres campos: `commands` (lista de comandos), `explanation` (explicación breve) y `finished` (booleano que indica si ha terminado).  Se insiste en que debe evitar operaciones destructivas y en que la respuesta siempre debe ser JSON.
2. **Ciclo de razonamiento**.  En cada iteración el script envía todas las interacciones previas al modelo y recibe un objeto JSON.  Si el campo `finished` es `true`, muestra el resumen y termina.  En caso contrario, ejecuta cada comando propuesto utilizando `subprocess.run`, captura su salida y la añade a la conversación.  La salida (código de retorno, STDOUT y STDERR) sirve de contexto para la siguiente consulta al modelo.
3. **Salvaguardas**.  Antes de ejecutar cada orden se aplica una heurística simple para bloquear comandos obviamente peligrosos (`rm -rf /`, `mkfs`, `shutdown`, etc.).  Si se detecta uno, se informa al modelo de que la orden fue bloqueada y se solicita un comando alternativo.
4. **Límites**.  Para prevenir bucles infinitos se puede establecer un número máximo de iteraciones con la opción `--max-steps` (por defecto, 10).  Si se alcanza este límite sin que el modelo marque la tarea como terminada, el script finaliza.
5. **Modelo configurable**.  Se puede elegir el modelo de OpenAI mediante `--model`; por defecto usa `gpt-4o`.  La temperatura se fija en `0` para obtener salidas deterministas.

### Sesión interactiva

La versión actual del agente admite un modo **interactivo**.  En lugar de finalizar tras completar una tarea, el programa pregunta al usuario si desea realizar otra acción; si se introduce un nuevo comando (o petición en lenguaje natural), este se añade a la misma conversación y el modelo lo tendrá en cuenta como contexto.  El ciclo de razonamiento y ejecución se repite para cada nueva orden hasta que el usuario presiona *Enter* sin texto o escribe `exit`/`quit`/`q`.  Los tokens consumidos se acumulan durante toda la sesión y, al salir, se muestra la estimación de coste.  Esta modalidad facilita encadenar peticiones relacionadas sin reiniciar el programa.

Además, el sistema resetea la lista de comandos fallidos al iniciar cada nueva orden, de modo que los errores previos no bloquean comandos legítimos en tareas posteriores.  Esta sesión interactiva permite realizar varias operaciones encadenadas sin perder el contexto ni tener que volver a invocar el script.  Si no se desea este comportamiento, se puede seguir proporcionando la instrucción inicial mediante `--task` y salir después del primer ciclo.

## Uso básico

Una vez instalado el agente se puede ejecutar así:

```bash
export OPENAI_API_KEY=sk-***************  # clave de API de OpenAI
ai-agent --task "instala nginx y levanta el servicio"
```

El agente generará un plan, ejecutará los comandos necesarios (por ejemplo, `sudo apt update && sudo apt install -y nginx` seguido de `sudo systemctl enable --now nginx`) y evaluará la salida de cada orden.  En cada paso mostrará qué pretende hacer y el resultado de cada ejecución.  Cuando determine que la tarea ha finalizado, imprimirá un breve resumen y terminará.  Para otras tareas se puede omitir `--task` y escribir la petición cuando el script la solicite.

## Archivo generado

Se han creado dos archivos que forman el corazón del despliegue:

| Archivo | Propósito |
|---|---|
| `ai_server_agent.py` | Implementa el agente autónomo en Python. Gestiona la conversación con la API de OpenAI, ejecuta los comandos propuestos y controla el flujo hasta completar la tarea. Incluye salvaguardas contra comandos destructivos. |
| `install_ai_agent.sh` | Script de instalación para Debian 12. Prepara el entorno, instala dependencias, copia el agente y genera el comando `ai‑agent`. |

Ambos ficheros están disponibles en esta respuesta y pueden desplegarse directamente en el servidor.
REPORTDOC
${SUDO} chmod 644 "$DEST_DIR/report.md"

# Create a Python virtual environment for the agent
VENV_DIR="$DEST_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    ${SUDO} python3 -m venv "$VENV_DIR"
fi

# Install Python dependencies inside the virtual environment
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir openai >/dev/null

# Create a wrapper executable in /usr/local/bin
WRAPPER="/usr/local/bin/ai-agent"
echo "[INFO] Creating wrapper executable $WRAPPER"
# Use a single‑quoted here‑doc to prevent variable expansion during script creation.
${SUDO} tee "$WRAPPER" >/dev/null <<'EOF'
#!/bin/bash
# Wrapper for launching the AI server agent

# Ensure the OpenAI API key is provided via environment variable
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set." >&2
    echo "Please export your OpenAI API key before running this command." >&2
    exit 1
fi

# Fixed paths for the agent installation
AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
AGENT_SCRIPT="$AGENT_DIR/ai_server_agent.py"

exec "$PYTHON_BIN" "$AGENT_SCRIPT" "$@"
EOF
${SUDO} chmod +x "$WRAPPER"

echo "[SUCCESS] Installation complete."
echo ""
# Prompt the user for an API key and optionally run the agent immediately.
read -r -p "Introduce tu clave OpenAI API para ejecutar el agente ahora (deja en blanco para omitir): " API_KEY_INPUT
if [ -n "$API_KEY_INPUT" ]; then
    export OPENAI_API_KEY="$API_KEY_INPUT"
    echo "[INFO] Ejecutando el agente interactivo..."
    # Use the Python binary in the virtual environment to run the agent. Forward any additional
    # arguments passed to this installer script to the Python program. This allows calls like
    # ./install_ai_agent.sh --task "mi primera orden".
    "$VENV_DIR/bin/python" "$DEST_DIR/ai_server_agent.py" "$@"
else
    echo "No se ha introducido ninguna clave; la instalación ha finalizado sin ejecutar el agente."
    echo "Para utilizar el agente más adelante, exporta tu clave y ejecuta el wrapper:\n"
    echo "  export OPENAI_API_KEY=<tu-clave>"
    echo "  ai-agent --task \"tu instrucción\""
fi
echo ""
echo "Puedes volver a ejecutar este script en cualquier momento para actualizar el agente,"
echo "reescribir el informe o introducir una nueva clave de API."

#!/bin/bash
# install_ai_agent.sh - Deploy the AI server agent on a Debian 12/13 system
# v1.4.2
set -euo pipefail

# ===== User defaults (Wrapper fallbacks) =====
DEFAULT_MODEL="gpt-5-mini"
DEFAULT_MAX_STEPS=24
DEFAULT_MAX_CMDS_PER_STEP=24
DEFAULT_WEB_MODE="auto"   # auto|on|off

# ===== Must be root or have sudo =====
if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] Necesitas ser root o tener sudo." >&2
  exit 1
fi
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

# ===== Lock =====
LOCK_FILE="/tmp/ai-agent.install.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE" || true
  flock -n 9 || { echo "[ERROR] Otra instalación en curso."; exit 1; }
fi

# ===== Version & Self-update =====
SCRIPT_VERSION="1.4.2"
UPDATE_URL="https://raw.githubusercontent.com/Halk58/ai-agent-cli/main/install_ai_agent.sh"

# Distro warn
if [ -r /etc/os-release ]; then . /etc/os-release; fi
[ "${ID:-}" = "debian" ] || echo "[WARN] Pensado para Debian; detectado ${ID:-?} ${VERSION_ID:-?}"

# Self-update
if [ -n "$UPDATE_URL" ] && command -v curl >/dev/null 2>&1; then
  tmp_update_file="$(mktemp)"; trap 'rm -f "$tmp_update_file"' EXIT
  if curl -fsSL -H 'Cache-Control: no-cache' --connect-timeout 5 -m 15 "$UPDATE_URL" -o "$tmp_update_file"; then
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
      echo "[WARN] No pude extraer SCRIPT_VERSION remoto."
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
${SUDO} mkdir -p "$DEST_DIR"
${SUDO} mkdir -p /etc/ai-agent && ${SUDO} chmod 755 /etc/ai-agent

echo "[INFO] Writing agent script to $DEST_DIR/ai_server_agent.py"
${SUDO} tee "$DEST_DIR/ai_server_agent.py" >/dev/null <<'PYCODE'
#!/usr/bin/env python3
# ai_server_agent.py - v1.4.2
import argparse, json, os, shlex, subprocess, sys, time
from glob import glob
from typing import Any, Dict, List, Tuple
try:
    from openai import OpenAI
except ImportError:
    print("Instala 'openai' (pip install 'openai>=1.60,<2')", file=sys.stderr); sys.exit(1)

DEFAULT_MODEL = os.getenv("AI_AGENT_DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_MAX_STEPS = int(os.getenv("AI_AGENT_DEFAULT_MAX_STEPS", "24"))
DEFAULT_MAX_CMDS_PER_STEP = int(os.getenv("AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP", "24"))
DEFAULT_WEB_MODE = os.getenv("AI_AGENT_WEB_MODE", "auto")  # auto|on|off

def run_shell_command(cmd: str, timeout: int = 900) -> Tuple[int,str,str]:
    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND","noninteractive")
    env.setdefault("LC_ALL","C.UTF-8"); env.setdefault("LANG","C.UTF-8")
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or f"Timeout after {timeout}s")
    except Exception as e:
        return 1, "", str(e)

def _tokenize(s: str) -> List[str]:
    try: return shlex.split(s, posix=True)
    except Exception: return []

SAFE_RM_PATTERNS = {"/var/lib/apt/lists/*"}

def is_really_dangerous(cmd: str) -> bool:
    parts = _tokenize(cmd)
    if not parts: return False
    head = parts[0]
    if head == "rm":
        flags = {p for p in parts[1:] if p.startswith("-")}
        targets = [p for p in parts[1:] if not p.startswith("-")]
        if any(("r" in f and "f" in f) for f in flags):
            for t in targets:
                if t in ("/", "/*"): return True
                if t in SAFE_RM_PATTERNS: continue
    if any(p.startswith("mkfs") for p in parts): return True
    if head == "dd" and any(x.startswith("of=/dev/") for x in parts[1:]): return True
    if head in {"shutdown","reboot","halt","poweroff"}: return True
    if ":(){ :|:& };:" in cmd: return True
    if cmd.strip().startswith("chmod -R") and " /" in cmd: return True
    return False

def sanitize_command(cmd: str) -> str:
    """Hacer comandos no-interactivos sin cambiar su semántica."""
    # gpg --dearmor → añade --batch --yes (evita Overwrite? y/o prompts)
    if "gpg" in cmd and "--dearmor" in cmd and "--batch" not in cmd:
        cmd = cmd.replace("gpg ", "gpg --batch --yes ", 1)
    # evita apt prompts si el usuario olvidó el entorno
    if cmd.strip().startswith(("apt-get ","apt ")):
        if "install " in cmd or "dist-upgrade" in cmd or "full-upgrade" in cmd or "upgrade " in cmd:
            if "DEBIAN_FRONTEND=" not in cmd:
                cmd = f"DEBIAN_FRONTEND=noninteractive {cmd}"
    return cmd

def clip(txt: str, limit: int=2500) -> str:
    if txt is None: return ""
    if len(txt)<=limit: return txt
    return txt[:1200] + f"\n...<clipped {len(txt)-1900} chars>...\n" + txt[-700:]

def read_os_release() -> Dict[str,str]:
    d={}
    try:
        for line in open("/etc/os-release","r",encoding="utf-8"):
            line=line.strip()
            if "=" in line and not line.startswith("#"):
                k,v=line.split("=",1); d[k]=v.strip().strip('"')
    except Exception: pass
    return d

def net_status()->str:
    rc,_,_=run_shell_command("getent hosts deb.debian.org >/dev/null 2>&1 || getent hosts 1.1.1.1 >/dev/null 2>&1", timeout=5)
    if rc!=0: return "degraded/no-dns"
    rc,_,_=run_shell_command("ping -n -c1 -W1 1.1.1.1 >/dev/null 2>&1", timeout=3)
    return "online" if rc==0 else "dns-only"

def env_snapshot()->str:
    osr=read_os_release(); root=os.geteuid()==0
    rc_df,out_df,_=run_shell_command("df -h / | awk 'NR==2{print $4\" free\"}'", timeout=3)
    rc_mem,out_mem,_=run_shell_command("awk '/MemAvailable/{printf \"%.0f MB\", $2/1024}' /proc/meminfo", timeout=3)
    return f"OS={osr.get('PRETTY_NAME','?')} CODENAME={osr.get('VERSION_CODENAME','?')} ROOT={'yes' if root else 'no'} NET={net_status()} DISK_FREE={(out_df.strip() if rc_df==0 else 'n/a')} MEM_FREE={(out_mem.strip() if rc_mem==0 else 'n/a')}"

def build_system_prompt()->str:
    return (
      "You are a Debian server automation agent.\n"
      "- Respond ONLY strict JSON: {\"commands\":[\"...\"],\"explanation\":\"...\",\"finished\":true|false}.\n"
      "- Keep explanation <=280 chars; propose <=5 commands per step (prefer one bash -lc block when state is shared).\n"
      "- NEVER use apt-key. When importing keys use: gpg --batch --yes --dearmor -o /usr/share/keyrings/<name>.gpg\n"
      "- When adding APT repos: WRITE to /tmp/ai-agent-repo-<name>.list (DO NOT write /etc/* yet). The client will validate and move if safe.\n"
      "- Use signed-by=/usr/share/keyrings/<name>.gpg in sources entries. Prefer non-interactive flags; avoid destructive ops.\n"
      "- stdin is non-interactive; commands must not prompt."
    )

# ---- Token-cap & retries ----
def chat_request_with_backoff(callable_factory, max_attempts=4, first_wait=0.6):
    wait=first_wait; last=None
    for _ in range(max_attempts):
        try: return callable_factory()
        except Exception as e:
            s=str(e).lower(); last=e
            if any(x in s for x in (" 429"," 500"," 502"," 503","timeout")):
                time.sleep(wait); wait=min(wait*2,5.0); continue
            break
    raise last

def call_chat_json(messages: List[Dict[str,str]], model: str)->Tuple[Dict[str,Any],Dict[str,int]]:
    client=OpenAI()
    def _call(with_temp:bool, cap_key:str|None):
        params=dict(model=model, messages=messages, response_format={"type":"json_object"})
        if with_temp: params["temperature"]=0.0
        if cap_key: params[cap_key]=600
        return chat_request_with_backoff(lambda: client.chat.completions.create(**params))
    try:
        resp=_call(True,"max_completion_tokens")
    except Exception as e1:
        try: resp=_call(True,"max_tokens")
        except Exception as e2:
            try: resp=_call(True,None)
            except Exception as e3:
                try: resp=_call(False,None)
                except Exception: raise RuntimeError(f"OpenAI API call failed: {e1}\nThen: {e2}\nFinally: {e3}") from e3
    content=resp.choices[0].message.content
    try: data=json.loads(content)
    except json.JSONDecodeError as e: raise ValueError(f"Model output is not valid JSON: {e}\nRaw content:\n{content}") from e
    usage=getattr(resp,"usage",None)
    return data, {"prompt_tokens":getattr(usage,"prompt_tokens",0) if usage else 0, "completion_tokens":getattr(usage,"completion_tokens",0) if usage else 0}

# ---- Responses API (web one-shot) ----
def responses_available()->bool:
    try: _=OpenAI().responses; return True
    except Exception: return False

def should_force_web(task:str)->bool:
    t=task.lower()
    kws=["última","latest","repo","repositorio","gpg","clave","mariadb","proxmox","docker","kubernetes","nodejs","nginx","postgres","hashicorp","grafana","prometheus","helm","tailscale","install from official"]
    return any(k in t for k in kws)

def responses_web_autoplan(task:str, model:str, mode:str)->str:
    if mode=="off" or not responses_available(): return ""
    prompt=("Decide if web search materially helps the following server task. If yes, PERFORM it via web_search and return:\n"
            "WEB_CONTEXT:\n<concise <=2000 chars summary with key facts, versions, and safe repo instructions for Debian>\n"
            "If not needed, return NO_WEB: <reason>.\nTASK:\n"+task)
    client=OpenAI()
    try: resp=client.responses.create(model=model, input=prompt, tools=[{"type":"web_search"}])
    except Exception: return ""
    text=getattr(resp,"output_text","") or str(resp)
    return text.split("WEB_CONTEXT:",1)[1].strip() if "WEB_CONTEXT:" in text else ""

# ---- APT repo validation workflow ----
def validate_tmp_repos_and_apply()->str:
    """
    Detect /tmp/ai-agent-repo-*.list files created by the model.
    For each, run apt-get update against ONLY that list (isolated). If OK, move into /etc/apt/sources.list.d/.
    """
    msgs=[]
    tmp_lists=sorted(glob("/tmp/ai-agent-repo-*.list"))
    for path in tmp_lists:
        # isolated update
        rc,out,err=run_shell_command(
            f"apt-get update -o Dir::Etc::sourcelist={shlex.quote(path)} -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0", timeout=120
        )
        if rc==0:
            # apply
            base=os.path.basename(path)
            dest=f"/etc/apt/sources.list.d/{base}"
            # move (overwrite silently)
            run_shell_command(f"install -m 644 {shlex.quote(path)} {shlex.quote(dest)}", timeout=10)
            run_shell_command(f"rm -f {shlex.quote(path)}", timeout=5)
            msgs.append(f"APPLIED_REPO: {dest}")
        else:
            # keep for inspection, but report failure
            msgs.append(f"REPO_VALIDATION_FAILED: {path}\nSTDERR:\n{clip(err)}\nSTDOUT:\n{clip(out)}")
    return "\n".join(msgs)

# ---- Diagnostics ----
def diag_hints_for(cmd:str, rc:int, stderr:str, stdout:str)->str:
    s=(stderr or "")+"\n"+(stdout or ""); sl=s.lower()
    hints=[]
    if "dpkg was interrupted" in s or "run 'dpkg --configure -a'" in sl: hints.append("Run: dpkg --configure -a (non-interactive).")
    if "could not get lock" in sl and ("dpkg" in sl or "apt" in sl): hints.append("Another APT/dpkg is running. Wait/kill process; then retry apt-get update.")
    if "hash sum mismatch" in sl: hints.append("Try: apt-get clean && apt-get update.")
    if "temporary failure resolving" in sl or "name resolution" in sl: hints.append("DNS issue. Check resolv.conf or wait for connectivity.")
    if "is not signed" in sl and "inrelease" in sl: hints.append("Repo not signed; ensure correct keyring and signed-by= in the .list file.")
    if "missing key" in sl or "no pubkey" in sl: hints.append("Missing repo key; import key to /usr/share/keyrings and reference via signed-by=.")
    if "does not have a release file" in sl or "404  not found" in sl: hints.append("Repo URL or distribution not available. Use a published release (e.g., latest supported non-LTS) instead of a non-existent branch.")
    if "permission denied" in sl: hints.append("Permission denied: use sudo/root or adjust perms/ownership.")
    if "command not found" in sl: hints.append("Command not found: install the package providing it (apt-get install <pkg>).")
    if "no space left on device" in sl: hints.append("Disk full; free space on /.")
    if rc==124: hints.append("Command timeout; split steps or increase timeout.")
    return "DIAGNOSTIC_HINTS:\n- "+"\n- ".join(hints) if hints else ""

def maybe_summarize_history(messages: List[Dict[str,str]], max_keep:int=18)->None:
    if len(messages)<=max_keep: return
    head=[messages[0]]; tail=messages[-12:]; middle=messages[1:-12]
    summaries=[]
    for m in middle:
        if m.get("role") in ("user","assistant"):
            summaries.append(f"{m['role']}: {clip(m.get('content',''), 400)}")
    messages[:]=head+[{"role":"system","content":"SUMMARY_OF_PRIOR_CONTEXT:\n"+"\n".join(summaries[:40])}]+tail

def apt_context_signature()->str:
    paths=["/etc/apt/sources.list"]; paths+=sorted(glob("/etc/apt/sources.list.d/*.list")); paths+=sorted(glob("/usr/share/keyrings/*.gpg")); paths+=sorted(glob("/etc/apt/trusted.gpg.d/*.gpg"))
    sig=[]
    for p in paths:
        try:
            st=os.stat(p); sig.append(f"{p}:{int(st.st_mtime)}:{st.st_size}")
        except FileNotFoundError: sig.append(f"{p}:absent")
        except Exception: pass
    return "|".join(sig)

def main()->None:
    parser=argparse.ArgumentParser(description="Autonomous server agent for Debian.")
    parser.add_argument("--task", type=str, help="Initial instruction.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-commands-per-step", type=int, default=DEFAULT_MAX_CMDS_PER_STEP)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--web", type=str, default=DEFAULT_WEB_MODE, choices=["auto","on","off"])
    args=parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY no está definida. Usa 'ai-agent --set-key' o exporta la variable.", file=sys.stderr); sys.exit(1)

    print("[INFO] Agent starting (interactive).", flush=True)
    messages=[{"role":"system","content":build_system_prompt()},
              {"role":"system","content":"ENV_SNAPSHOT: "+env_snapshot()}]

    # Single prompt
    if args.task: first_task=args.task.strip()
    else:
        first_task=input("Enter task (or press Enter to exit): ").strip() if sys.stdin.isatty() else ""
        if not first_task: print("Exiting interactive session.", flush=True); return

    # One-shot web context (auto heuristic)
    web_mode = args.web
    if web_mode=="auto" and should_force_web(first_task): web_mode="on"
    if web_mode in ("on","auto"):
        web_ctx=responses_web_autoplan(first_task, args.model, web_mode)
        if web_ctx: messages.append({"role":"system","content":"WEB_CONTEXT:\n"+clip(web_ctx,2000)})

    total_prompt_tokens=0; total_completion_tokens=0
    failed: Dict[str,int]={}; errors: Dict[str,str]={}
    last_apt_sig=apt_context_signature()

    pending=first_task
    while True:
        user_task=pending if pending is not None else input("Enter next task (or press Enter to exit): ").strip()
        pending=None
        if not user_task: print("Exiting interactive session.", flush=True); break
        messages.append({"role":"user","content":user_task})
        finished=False

        for step in range(1, args.max_steps+1):
            apt_sig_before=apt_context_signature()
            maybe_summarize_history(messages)

            # validate any staged repos before asking model again (apply from previous run)
            repo_report=validate_tmp_repos_and_apply()
            if repo_report:
                messages.append({"role":"system","content":repo_report})

            try:
                data, usage=call_chat_json(messages, args.model)
            except Exception as e:
                print(f"Error during API call: {e}", file=sys.stderr, flush=True); finished=True; break
            total_prompt_tokens+=usage.get("prompt_tokens",0); total_completion_tokens+=usage.get("completion_tokens",0)

            if not isinstance(data,dict) or not all(k in data for k in ("commands","explanation","finished")):
                print("Unexpected JSON:", data, file=sys.stderr, flush=True); finished=True; break

            cmds: List[str]=(data.get("commands") or [])
            expl=str(data.get("explanation",""))[:500]; done=bool(data.get("finished",False))
            if not isinstance(cmds,list) or not all(isinstance(c,str) for c in cmds):
                print("'commands' must be list[str].", file=sys.stderr, flush=True); finished=True; break

            if len(cmds)>args.max_commands_per_step:
                print(f"[{time.strftime('%F %T')}] Limiting commands from {len(cmds)} to {args.max_commands_per_step}", flush=True)
                cmds=cmds[:args.max_commands_per_step]

            print(f"\n=== Step {step} ===", flush=True)
            print(f"AI explanation: {expl}", flush=True)
            if cmds: print("Proposed commands:", flush=True); [print(f"  $ {c}", flush=True) for c in cmds]
            else: print("No commands proposed.", flush=True)

            # auto-batch heuristic
            combine=False
            if len(cmds)>1:
                joined="\n".join(cmds); combine = any(k in joined for k in [" set -e","set -o pipefail","export ","=", "&&",";",". /etc/os-release","source /etc/os-release","cd "])

            if combine:
                if any(is_really_dangerous(c) for c in cmds):
                    print("Blocked batch due to dangerous line.", flush=True)
                else:
                    block="\n".join(sanitize_command(c) for c in cmds)
                    if args.confirm and sys.stdin.isatty():
                        ans=input("Execute batch? [y/N]: ").strip().lower()
                        if ans not in {"y","yes","s","si","sí"}:
                            print("Batch skipped.", flush=True)
                            messages+= [{"role":"assistant","content":json.dumps(data)},
                                        {"role":"user","content":"User declined batch. Propose a single minimal safe command."}]
                            continue
                    rc,out,err=(0,"","") if args.dry_run else run_shell_command(f"bash -lc {shlex.quote(block)}", timeout=args.timeout)
                    if out: print("-- STDOUT --"); print(out.rstrip(), flush=True)
                    if err: print("-- STDERR --", file=sys.stderr); print(err.rstrip(), file=sys.stderr, flush=True)
                    messages.append({"role":"assistant","content":json.dumps(data)})
                    if rc!=0:
                        hint=diag_hints_for("BATCH", rc, err, out)
                        if hint: messages.append({"role":"system","content":hint})
                    messages.append({"role":"user","content":f"BATCH EXECUTION\nReturn code: {rc}\nSTDOUT:\n{clip(out)}\nSTDERR:\n{clip(err)}"})
                    if done:
                        print("\nTask complete.\nSummary:"); print(expl, flush=True); finished=True; break
                    continue

            skip=False
            for i,c in enumerate(cmds, start=1):
                c=sanitize_command(c)
                prev_fail=c in failed
                allow_retry = c.strip().startswith(("apt-get update","apt update","dpkg --configure -a","apt-get -f install","apt -f install"))
                apt_changed = (apt_context_signature()!=last_apt_sig) or (apt_context_signature()!=apt_sig_before)

                if prev_fail and not (allow_retry and apt_changed and failed[c]<3):
                    msg=f"The command '{c}' already failed with:\n{errors.get(c,'')}\nPropose alternative."
                    print(f"Skipping repeated failing command: {c}", flush=True)
                    messages+= [{"role":"assistant","content":json.dumps(data)}, {"role":"user","content":msg}]
                    skip=True; break

                if is_really_dangerous(c):
                    print(f"Blocked dangerous command: {c}", flush=True)
                    messages+= [{"role":"assistant","content":json.dumps(data)},
                               {"role":"system","content":"DIAGNOSTIC_HINTS:\n- Proposed command blocked as dangerous. Provide a safer alternative."}]
                    skip=True; break

                if args.confirm and sys.stdin.isatty():
                    ans=input(f"Execute command {i}/{len(cmds)}? [y/N]: ").strip().lower()
                    if ans not in {"y","yes","s","si","sí"}:
                        print("Skipped by user.", flush=True)
                        messages+= [{"role":"assistant","content":json.dumps(data)}, {"role":"user","content":f"User declined: {c}. Suggest different safe cmd."}]
                        continue

                print(f"\nExecuting command {i}/{len(cmds)}: {c}", flush=True)
                rc,stdout,stderr=(0,"","") if args.dry_run else run_shell_command(c, timeout=args.timeout)
                if stdout: print("-- STDOUT --"); print(stdout.rstrip(), flush=True)
                if stderr: print("-- STDERR --", file=sys.stderr); print(stderr.rstrip(), file=sys.stderr, flush=True)

                messages.append({"role":"assistant","content":json.dumps(data)})
                if rc!=0:
                    failed[c]=failed.get(c,0)+1; errors[c]=(stderr or stdout).strip()
                    hint=diag_hints_for(c, rc, stderr, stdout)
                    if hint: messages.append({"role":"system","content":hint})
                messages.append({"role":"user","content":f"Command: {c}\nReturn code: {rc}\nSTDOUT:\n{clip(stdout)}\nSTDERR:\n{clip(stderr)}"})
                sig=apt_context_signature()
                if sig!=last_apt_sig: last_apt_sig=sig

            if skip: continue
            if done:
                print("\nTask complete.\nSummary:"); print(expl, flush=True); finished=True; break
            time.sleep(0.2)

        if not finished and step>=args.max_steps:
            print("Maximum number of steps reached. The task may not be complete.", flush=True)

    # (Opcional) estimación de coste
    if total_prompt_tokens or total_completion_tokens:
        pricing={"gpt-4o":{"input":0.0025,"output":0.01},"gpt-4o-mini":{"input":0.00015,"output":0.0003},"gpt-5-mini":{"input":0.00025,"output":0.0020}}
        mk=args.model.lower()
        if mk in pricing:
            r=pricing[mk]; cost=(total_prompt_tokens*r["input"]+total_completion_tokens*r["output"])/1000.0
            print(f"\nApproximate API usage (chat): prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
            print(f"Estimated cost for model '{args.model}': ${cost:.4f} (USD)")
        else:
            print(f"\nToken usage (chat): prompt={total_prompt_tokens}, completion_tokens={total_completion_tokens}.")
if __name__=="__main__": main()
PYCODE
${SUDO} chmod 644 "$DEST_DIR/ai_server_agent.py"

echo "[INFO] Writing report to $DEST_DIR/report.md"
${SUDO} tee "$DEST_DIR/report.md" >/dev/null <<'REPORTDOC'
# Agente autónomo Debian (v1.4.2)
Mejoras:
- Sanitización automática de comandos (gpg --batch --yes, APT no-interactivo).
- Protocolo de repos: /tmp/ai-agent-repo-*.list → validación aislada → mover a /etc/apt/sources.list.d/ si es válido.
- Heurística web-search (auto) para tareas de “última versión”, repos oficiales, etc.
- Auto-batching agresivo y DIAGNOSTIC_HINTS ante errores típicos (APT, clave, firma, DNS, Release 404).
REPORTDOC
${SUDO} chmod 644 "$DEST_DIR/report.md"

# ==== venv & deps ====
VENV_DIR="$DEST_DIR/venv"
[ -d "$VENV_DIR" ] || { echo "[INFO] Creating virtual environment..."; ${SUDO} python3 -m venv "$VENV_DIR"; }
echo "[INFO] Installing Python dependencies into the virtual environment..."
${SUDO} "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
${SUDO} "$VENV_DIR/bin/pip" install --no-cache-dir 'openai>=1.60,<2' >/dev/null

# ==== wrapper ====
WRAPPER="/usr/local/bin/ai-agent"
echo "[INFO] Creating wrapper executable $WRAPPER"
${SUDO} tee "$WRAPPER" >/dev/null <<'EOF'
#!/bin/bash
set -euo pipefail
AGENT_DIR="/opt/ai-agent"
VENV_DIR="$AGENT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
AGENT_SCRIPT="$AGENT_DIR/ai_server_agent.py"
KEY_FILE="/etc/ai-agent/agent.env"
CONF_FILE="/etc/ai-agent/config.env"

print_help(){ cat <<'HLP'
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
- Variables por defecto (persistibles en /etc/ai-agent/config.env):
  AI_AGENT_DEFAULT_MODEL (por defecto: gpt-5-mini)
  AI_AGENT_DEFAULT_MAX_STEPS (24)
  AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP (24)
  AI_AGENT_WEB_MODE (auto|on|off; por defecto auto)
- Ayuda del agente Python:
  ai-agent --help-agent
HLP
}
mask(){ local k="$1"; local n=${#k}; [ "$n" -le 8 ] && echo "********" || echo "${k:0:4}********${k: -4}"; }
set_key(){ local v="${1:-}"; if [ -z "$v" ]; then [ -t 0 ] && { read -r -s -p "Introduce tu OPENAI_API_KEY: " v; echo; } || { echo "[ERROR] No TTY y sin clave"; exit 1; }; fi
  install -m 700 -d /etc/ai-agent; umask 077; printf 'OPENAI_API_KEY=%s\n' "$v" > "$KEY_FILE"; echo "[OK] Clave guardada en $KEY_FILE"; }
clear_key(){ [ -f "$KEY_FILE" ] && { rm -f "$KEY_FILE"; echo "[OK] Clave eliminada."; } || echo "[INFO] No hay clave persistida."; }
show_key(){ if [ -f "$KEY_FILE" ]; then . "$KEY_FILE"; [ -n "${OPENAI_API_KEY:-}" ] && echo "OPENAI_API_KEY=$(mask "$OPENAI_API_KEY")" || echo "[WARN] Archivo sin clave."; else echo "[INFO] No hay clave en $KEY_FILE"; fi; }

case "${1:-}" in -h|--help) print_help; exit 0;; --help-agent) exec "$PYTHON_BIN" "$AGENT_SCRIPT" --help;; esac
case "${1:-}" in --set-key) shift; set_key "${1:-}"; exit 0;; --clear-key) clear_key; exit 0;; --show-key) show_key; exit 0;; esac

: "${AI_AGENT_DEFAULT_MODEL:=${AI_AGENT_DEFAULT_MODEL:-gpt-5-mini}}"
: "${AI_AGENT_DEFAULT_MAX_STEPS:=${AI_AGENT_DEFAULT_MAX_STEPS:-24}}"
: "${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:=${AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP:-24}}"
: "${AI_AGENT_WEB_MODE:=${AI_AGENT_WEB_MODE:-auto}}"
[ -r "$CONF_FILE" ] && . "$CONF_FILE"
[ -z "${OPENAI_API_KEY:-}" ] && [ -r "$KEY_FILE" ] && . "$KEY_FILE"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -t 0 ]; then
  read -r -s -p "Introduce tu OPENAI_API_KEY: " OPENAI_API_KEY; echo
  if [ -n "${OPENAI_API_KEY:-}" ]; then read -r -p "¿Guardar clave en $KEY_FILE? [y/N]: " a; case "${a,,}" in y|yes|s|si|sí) install -m700 -d /etc/ai-agent; umask 077; echo "OPENAI_API_KEY=$OPENAI_API_KEY" > "$KEY_FILE"; echo "[OK] Clave guardada.";; esac; fi
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "[ERROR] Falta OPENAI_API_KEY. Usa 'ai-agent --set-key'."; exit 1; }

export PYTHONUNBUFFERED=1
exec env AI_AGENT_DEFAULT_MODEL="$AI_AGENT_DEFAULT_MODEL" \
         AI_AGENT_DEFAULT_MAX_STEPS="$AI_AGENT_DEFAULT_MAX_STEPS" \
         AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP="$AI_AGENT_DEFAULT_MAX_CMDS_PER_STEP" \
         AI_AGENT_WEB_MODE="$AI_AGENT_WEB_MODE" \
         OPENAI_API_KEY="$OPENAI_API_KEY" \
    "$PYTHON_BIN" "$AGENT_SCRIPT" "$@"
EOF
${SUDO} chmod +x "$WRAPPER"

echo "[SUCCESS] Installation complete."
if [ -t 0 ]; then
  read -r -p "¿Quieres guardar ahora tu OPENAI_API_KEY para futuros usos? [y/N]: " WANT
  case "${WANT,,}" in
    y|yes|s|si|sí) read -r -s -p "Introduce tu OPENAI_API_KEY: " K; echo
                   if [ -n "${K:-}" ]; then ${SUDO} install -m700 -d /etc/ai-agent; umask 077; echo "OPENAI_API_KEY=$K" | ${SUDO} tee /etc/ai-agent/agent.env >/dev/null; echo "[OK] Clave guardada."; fi;;
    *) echo "[INFO] Puedes usar 'ai-agent --set-key' más tarde.";;
  esac
fi
printf "\nListo. Prueba:\n  ai-agent --task \"muestra 'uname -a' y 'lsb_release -a'\"\n"

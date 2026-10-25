"""pi-voice 开关控制器 —— voiceon / voiceoff / voicestatus 的实际实现。

放在项目里而不是塞进 .bat：判断"端口归谁""进程还在不在"这类事用 Python 写更可靠，
也方便直接测。.cmd 只是转发。

用法：
    python ctl.py start [--open|--no-open] [main.py 的其他参数...]
    python ctl.py stop [--force]
    python ctl.py status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "pi-voice"
PID_FILE = STATE_DIR / "pid"
LOG_FILE = STATE_DIR / "server.log"
ERR_FILE = STATE_DIR / "server.err.log"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


# -- 基础探测 ---------------------------------------------------------------
def server_port() -> int:
    sys.path.insert(0, str(PROJ))
    import config

    return int(config.SERVER_PORT)


def is_listening(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.4)
                sock.connect((host, port))
                return True
        except OSError:
            continue
    return False


def port_owner(port: int) -> int | None:
    """从 netstat 里找出监听该端口的 PID。"""
    try:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            encoding="mbcs",
            errors="replace",
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pattern = re.compile(rf"[:.]{port}\s+\S+\s+LISTENING\s+(\d+)", re.IGNORECASE)
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def pid_alive(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            encoding="mbcs",
            errors="replace",
            timeout=15,
        )
        return str(pid) in (result.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return False


def pid_command_line(pid: int) -> tuple[str, str]:
    """返回 (本进程命令行, 父进程命令行)。venv 的 python.exe 是个转发器，
    真正监听端口的是它 spawn 出来的 base python，命令行里只有 main.py。"""
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction SilentlyContinue;"
        " if (-not $p) { '' ; '' ; exit }"
        " $pp = Get-CimInstance Win32_Process -Filter (\"ProcessId=\" + $p.ParentProcessId) -ErrorAction SilentlyContinue;"
        " $p.CommandLine; if ($pp) { $pp.CommandLine } else { '' }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return "", ""
    lines = [line.strip() for line in (result.stdout or "").splitlines()]
    while len(lines) < 2:
        lines.append("")
    return lines[0], lines[1]


def is_our_process(pid: int) -> bool:
    command, parent_command = pid_command_line(pid)
    haystack = f"{command} {parent_command}".lower()
    return "main.py" in haystack and str(PROJ).lower() in haystack


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def write_pid(pid: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="ascii")


def kill_tree(pid: int) -> bool:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


# -- 命令 -------------------------------------------------------------------
def cmd_start(args: list[str]) -> int:
    port = server_port()
    url = f"http://localhost:{port}/"
    open_browser = "--open" in args
    extra = [item for item in args if item not in {"--open", "--no-open"}]

    if is_listening(port):
        owner = port_owner(port)
        if owner and read_pid() != owner and (pid_alive(owner) and is_our_process(owner)):
            write_pid(owner)
        if owner and read_pid() == owner:
            print(f"[pi-voice] already running (pid {owner}) - {url}")
            if open_browser:
                _open(url)
            return 0
        print(f"[pi-voice] port {port} is held by another process (pid {owner}); run: voiceoff --force")
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    stdout = LOG_FILE.open("wb")
    stderr = ERR_FILE.open("wb")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py", *extra],
            cwd=str(PROJ),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )
    finally:
        stdout.close()
        stderr.close()
    write_pid(proc.pid)
    print(f"[pi-voice] starting (pid {proc.pid}) ...")

    for _ in range(120):
        if proc.poll() is not None:
            print("[pi-voice] server exited right after start. Log tail:")
            print(tail(ERR_FILE, 15) or tail(LOG_FILE, 15))
            PID_FILE.unlink(missing_ok=True)
            return 1
        if is_listening(port):
            print(f"[pi-voice] ON  pid={proc.pid}  {url}")
            print(f"[pi-voice] hold F9 to talk; logs: {LOG_FILE}")
            if open_browser:
                _open(url)
            return 0
        _sleep(0.5)

    print(f"[pi-voice] still not listening after 60s (pid {proc.pid} alive).")
    print(f"[pi-voice] check: voicestatus   logs: {LOG_FILE}")
    return 1


def cmd_stop(args: list[str]) -> int:
    port = server_port()
    force = "--force" in args
    killed: list[int] = []

    pid = read_pid()
    if pid and pid_alive(pid):
        if is_our_process(pid) or force:
            print(f"[pi-voice] stopping pid {pid} ...")
            if kill_tree(pid):
                killed.append(pid)
        else:
            print(f"[pi-voice] pid {pid} does not look like pi-voice; run: voiceoff --force")
            return 1
    PID_FILE.unlink(missing_ok=True)

    if is_listening(port):
        owner = port_owner(port)
        if owner:
            if is_our_process(owner) or force:
                print(f"[pi-voice] stopping port owner pid {owner} ...")
                if kill_tree(owner):
                    killed.append(owner)
            else:
                print(f"[pi-voice] port {port} held by pid {owner} (not pi-voice); run: voiceoff --force")
                return 1

    for _ in range(20):
        if not is_listening(port):
            break
        _sleep(0.25)

    if is_listening(port):
        print(f"[pi-voice] port {port} is still listening - stop failed")
        return 1
    print("[pi-voice] OFF" if killed else "[pi-voice] not running")
    return 0


def cmd_status(args: list[str]) -> int:
    port = server_port()
    pid = read_pid()
    alive = bool(pid and pid_alive(pid))
    listening = is_listening(port)
    print(f"[pi-voice] {'RUNNING' if alive or listening else 'STOPPED'}")
    print(f"  pid file : {pid if pid else '-'} ({'alive' if alive else 'not alive'})")
    if not alive and listening:
        owner = port_owner(port)
        print(f"  port owned by pid {owner} (pid file stale; voiceoff will adopt it)")
    print(f"  url      : http://localhost:{port}/ -> {'listening' if listening else 'no listener'}")
    print(f"  log      : {LOG_FILE}")
    tail_text = tail(LOG_FILE, 6)
    if tail_text:
        print("  -- log tail --")
        print("\n".join("  " + line for line in tail_text.splitlines()))
    err_text = tail(ERR_FILE, 6)
    if err_text:
        print("  -- stderr tail --")
        print("\n".join("  " + line for line in err_text.splitlines()))
    return 0 if (alive or listening) else 3


def _open(url: str) -> None:
    try:
        os.startfile(url)  # noqa: S606 - 打开默认浏览器
    except OSError:
        print(f"[pi-voice] open {url} manually")


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["status"]
    command, rest = argv[0], argv[1:]
    handlers = {"start": cmd_start, "stop": cmd_stop, "status": cmd_status}
    handler = handlers.get(command)
    if handler is None:
        print(f"unknown command: {command} (start|stop|status)")
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本地改好的板端源码推到 RK3588 开发板（gatekeeper_V4），可选顺手编译。

在【项目根目录】运行：
    python build-check/board_push.py                  # 只推"大小变了"的 src/ include/ 文件
    python build-check/board_push.py --all            # 全推（不比对）
    python build-check/board_push.py src/gate_id_counter.cpp include/gate_id_counter.h
    python build-check/board_push.py --build          # 推完在板上 cmake+make -j4
    python build-check/board_push.py --build --all    # 全推并编译
    python build-check/board_push.py --status         # 只看板端状态，不推

连接信息（可用环境变量覆盖，默认即现场值）：
    BOARD_HOST=192.168.1.104  BOARD_USER=topeet  BOARD_PASS=topeet
    BOARD_DIR=/home/topeet/code/gatekeeper_V4

为什么不用 scp/ssh：Windows 上没有 sshpass，非交互式带密码只能用 paramiko。
"""
import argparse
import hashlib
import os
import posixpath
import sys

try:
    import paramiko
except ImportError:
    sys.exit("缺少 paramiko，请先运行：pip install paramiko")

HOST = os.environ.get("BOARD_HOST", "192.168.1.104")
USER = os.environ.get("BOARD_USER", "topeet")
PASS = os.environ.get("BOARD_PASS", "topeet")
RDIR = os.environ.get("BOARD_DIR", "/home/topeet/code/gatekeeper_V4")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOBS = ("src", "include")          # 默认同步这两棵子树
EXTS = (".cpp", ".h", ".hpp", ".cc", ".c")  # 只推源码
# ⚠️ 构建脚本必须一起同步（2026-09-22 踩过）：
#    新增源文件时若板端 CMakeLists.txt 没更新，cmake 根本不会编译新文件，
#    直到链接期才报 undefined reference，很难一眼看出是"构建脚本没同步"。
EXTRA_FILES = ("CMakeLists.txt",)


def collect_default():
    """收集 src/ 与 include/ 下的源码文件 + 构建脚本（相对项目根的路径）。"""
    out = []
    for sub in DEFAULT_GLOBS:
        base = os.path.join(ROOT, sub)
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.lower().endswith(EXTS):
                    full = os.path.join(dirpath, fn)
                    out.append(os.path.relpath(full, ROOT).replace("\\", "/"))
    for fn in EXTRA_FILES:
        if os.path.exists(os.path.join(ROOT, fn)):
            out.append(fn)
    return sorted(out)


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=PASS, timeout=10,
                look_for_keys=False, allow_agent=False)
    return cli


def local_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_md5(cli, rels):
    """一次 md5sum 批量取远端 md5。为什么不比大小：大小相同 ≠ 内容相同
    （曾出现"大小一样就跳过"，但实际是板端文件已被别的途径替换过）。"""
    if not rels:
        return {}
    paths = " ".join(posixpath.join(RDIR, r) for r in rels)
    _rc, out, _e = run(cli, f"md5sum {paths} 2>/dev/null")
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            got[parts[1]] = parts[0]
    return got


def run(cli, cmd, timeout=600):
    _in, out, err = cli.exec_command(cmd, timeout=timeout)
    rc = out.channel.recv_exit_status()
    return rc, out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")


def report_status(cli):
    """一眼回答：板端代码是不是最新的？二进制是不是已经把最新代码编进去了？"""
    print("=== 板端状态 ===")
    rc, o, e = run(cli, "uname -sr; hostname; date '+%Y-%m-%d %H:%M:%S'")
    info = [x for x in (o or e).strip().splitlines() if x]
    print(f"--- 连接 ---\n  {USER}@{HOST}   " + "  |  ".join(info))

    # ① 代码同步状态：逐文件 md5 比对
    files = collect_default()
    rmd5 = remote_md5(cli, files)
    diff = []
    for rel in files:
        rh = rmd5.get(posixpath.join(RDIR, rel))
        if rh is None:
            diff.append((rel, "板端缺失"))
        elif rh != local_md5(os.path.join(ROOT, rel)):
            diff.append((rel, "内容不同"))
    print("--- ① 代码同步（md5 逐文件比对）---")
    if not diff:
        print(f"  ✓ {len(files)} 个文件（源码 + 构建脚本）全部与本地一致 —— 板端代码已是最新")
    else:
        print(f"  ✗ {len(diff)}/{len(files)} 个文件与本地不同：")
        for rel, why in diff[:10]:
            print(f"      {rel}  ({why})")
        if len(diff) > 10:
            print(f"      … 另外 {len(diff) - 10} 个")
        if any(r.endswith("CMakeLists.txt") for r, _ in diff):
            print("      ⚠️ CMakeLists.txt 不同 = 板端构建脚本过期！本地若新增/删除了源文件，")
            print("         不同步它会导致 cmake 不编译新文件、链接期才报 undefined reference。")
        print("  → 执行：python build-check/board_push.py --build")

    # ② 二进制新鲜度：.o 是否比源码新
    print("--- ② 板上二进制 / 编译新鲜度 ---")
    _rc, o, _e = run(cli, f"ls -l --time-style=+%m-%d_%H:%M {RDIR}/build/main_tracking 2>/dev/null"
                         f" || echo '（未编译）'")
    print("  产物   " + o.strip())
    _rc, o, _e = run(cli, f"find {RDIR}/build -name 'gate_id_counter.cpp.o' "
                          f"-printf '  %TY-%Tm-%Td %TH:%TM  %p\\n' 2>/dev/null")
    obj_line = o.strip()
    _rc, o, _e = run(cli, f"stat -c '%y' {RDIR}/src/gate_id_counter.cpp 2>/dev/null | cut -c1-16")
    src_time = o.strip()
    print("  源码   " + (src_time or "?"))
    if obj_line:
        print(obj_line)
    else:
        print("  （找不到 .o，尚未编译过）")

    # ③ 关键参数
    print("--- ③ 板端生效的计数参数 ---")
    _rc, o, _e = run(cli, f"grep -nE 'kSealMinTravel|kSealProbationFrames|kUnsealConfirmFrames|"
                          f"kStateKeepFrames|kMoveMinStep' {RDIR}/src/gate_id_counter.cpp")
    print("\n".join("  " + x for x in o.strip().splitlines()) or "  （读不到）")

    # ④ 判据速查
    print("--- ④ 判据 ---")
    if diff:
        print("  板端代码**不是**最新 → 先推再编译")
    elif not obj_line:
        print("  代码已同步，但**还没编译** → 执行 --build")
    else:
        print("  代码已同步；.o 若比源码时间新，则二进制已含最新代码。")
        print("  想进一步坐实：板端 grep 二进制里的新注记字符串，例如")
        print(f"    grep -aoE 'seal-confirmed|seal-cancelled|unsealed' {RDIR}/build/main_tracking | sort -u")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="要推送的文件（相对项目根的路径）")
    ap.add_argument("--all", action="store_true", help="不比对，全部推送")
    ap.add_argument("--build", action="store_true", help="推完在板上编译")
    ap.add_argument("--status", action="store_true", help="只打印板端状态")
    args = ap.parse_args()

    try:
        cli = connect()
    except Exception as exc:                       # noqa: BLE001
        sys.exit(f"连不上 {USER}@{HOST}: {exc}\n（确认板子已开机、与本机同网段）")

    try:
        if args.status:
            report_status(cli)
            return 0

        files = args.files or collect_default()
        files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
        if not files:
            sys.exit("没有找到要推送的文件")

        # ---- 用 md5 严格比对，决定哪些需要推 ----
        rmd5 = {} if args.all else remote_md5(cli, files)
        want = []
        for rel in files:
            if args.all:
                want.append(rel)
                continue
            if rmd5.get(posixpath.join(RDIR, rel)) != local_md5(os.path.join(ROOT, rel)):
                want.append(rel)
        skipped = [r for r in files if r not in want]

        sftp = cli.open_sftp()
        made = set()
        pushed = []
        for rel in want:
            local = os.path.join(ROOT, rel)
            remote = posixpath.join(RDIR, rel)
            rdir = posixpath.dirname(remote)

            if rdir not in made:                       # 远端目录按需创建
                parts, cur = rdir.split("/"), ""
                for p in parts:
                    if not p:
                        cur = "/"
                        continue
                    cur = posixpath.join(cur, p)
                    try:
                        sftp.stat(cur)
                    except IOError:
                        sftp.mkdir(cur)
                made.add(rdir)

            sftp.put(local, remote)
            pushed.append((rel, os.path.getsize(local)))

        sftp.close()

        print(f"=== 推送结果（{HOST}:{RDIR}）===")
        for rel, size in pushed:
            print(f"  [推送] {rel}  ({size} B)")
        if skipped:
            print(f"  [跳过] {len(skipped)} 个文件 md5 与板端一致（未变）")
        if not pushed:
            print("  板端代码已与本地完全一致，无需推送。")

        # ---- 回读校验：把推上去的文件重新取 md5 与本地比 ----
        if pushed:
            back = remote_md5(cli, [r for r, _ in pushed])
            bad = [r for r, _ in pushed
                   if back.get(posixpath.join(RDIR, r))
                   != local_md5(os.path.join(ROOT, r))]
            print("  回读校验：" + ("全部一致 ✓" if not bad else f"不一致 {bad} ✗"))

        if args.build:
            print("\n=== 板上编译（cmake + make -j4）===")
            # ⚠️ 必须走 login shell：板端 cmake 装在 ~/.local/bin，非交互式 SSH 的 PATH
            #    只有 /usr/bin 那几个，直接执行会"command not found"（退出码 127、且无输出）。
            # ⚠️ 不能用 `make ... | tail`：管道的退出码是 tail 的，make 失败也会显示"退出码 0"
            #    （2026-09-22 踩过：链接 undefined reference，脚本却报成功）。
            cmd = (f"bash -lc 'cd {RDIR}/build && (cmake .. >/tmp/_cm.log 2>&1 "
                   f"&& make -j4 >/tmp/_mk.log 2>&1); rc=$?; "
                   f"echo === cmake ===; tail -6 /tmp/_cm.log; "
                   f"echo === make ===; tail -40 /tmp/_mk.log; "
                   f"echo __MK_RC__=$rc; exit $rc'")
            rc, o, e = run(cli, cmd, timeout=1800)
            print(o.rstrip() or e.rstrip())
            print(f"=== 退出码 {rc} ===")
            if rc == 0:
                rc2, o2, _ = run(cli, f"ls -l --time-style=+%m-%d_%H:%M {RDIR}/build/main_tracking")
                print("产物：" + o2.strip())
            else:
                print("  ⚠️ 编译**失败**（原因见上面输出）。两个最常见原因：")
                print("     · undefined reference 到新模块 → 板端 CMakeLists.txt 未同步"
                      "（本脚本已把它纳入默认同步，若仍失败请确认源文件已进 add_executable）")
                print("     · 找不到头文件 → 新头文件没推上去")
            return rc
    finally:
        cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

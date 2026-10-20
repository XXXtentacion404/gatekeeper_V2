#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 RK3588 板端执行任意 shell 命令（复用 board_push.py 的连接信息）。

用法（在【项目根目录】运行）：
    python build-check/board_exec.py "ls -la build/main_tracking"
    python build-check/board_exec.py --timeout 30 "ps -ef | grep main_tracking"

环境变量（与 board_push.py 同名）：
    BOARD_HOST=192.168.1.104  BOARD_USER=topeet  BOARD_PASS=topeet
    BOARD_DIR=/home/topeet/code/gatekeeper_V4

为什么单独写一个：board_push.py 只管推源码 + 编译，没有通用执行入口；
排查现场问题时经常需要在板端敲一两条命令，为此再开一个 ssh 客户端很麻烦。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_push as bp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="在 RK3588 板端执行 shell 命令")
    ap.add_argument("cmd", help="要执行的命令（整体作为第一个位置参数传入）")
    ap.add_argument("--timeout", type=int, default=120, help="超时秒数，默认 120")
    ap.add_argument("--cwd", default=None,
                    help="执行前先 cd 到的目录（默认板端项目根 %s）" % bp.RDIR)
    ap.add_argument("--quiet", action="store_true", help="只输出 stdout")
    a = ap.parse_args()

    cwd = a.cwd if a.cwd is not None else bp.RDIR
    full = "cd %s && %s" % (cwd, a.cmd)

    cli = bp.connect()
    try:
        rc, out, err = bp.run(cli, full, timeout=a.timeout)
    finally:
        cli.close()

    if out:
        sys.stdout.write(out)
    if err and not a.quiet:
        sys.stderr.write(err)
    if not a.quiet:
        sys.stderr.write("\n[exit code] %d\n" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())

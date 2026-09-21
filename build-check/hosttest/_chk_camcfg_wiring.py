# -*- coding: utf-8 -*-
"""CameraConfig 字段搬运自检 —— 防"手写 move 构造漏字段"的复发。

背景（2026-09-16 现场事故）：
    main_tracking.cpp 的 CameraConfig 有**手写 move 构造函数**（因为含 atomic /
    unique_ptr / Mat，不能 default）。新增字段时必须在两处手工搬运：
      ① move 构造的初始化列表      → 否则 vector 扩容 / emplace_back 时静默丢失
      ② loadCameras() 里的 cc.X = c.X → 否则配置解析出来的值根本没进 CameraConfig
    当天 zone_line / zone_role 两处都漏了 → 板端表现是
      "画面不画区域 + IN/OUT/RETRO 全程 0"，而 Region/Zone 显示照旧正常，
    极具迷惑性（Region 走的是检测区 mask，不受影响）。main_tracking.cpp 依赖
    海康 SDK / X11，宿主机只能 -fsyntax-only，跑不了单测 → 用静态分析补这道网。

用法：python _chk_camcfg_wiring.py [main_tracking.cpp 路径]
退出码 0 = 全部字段都搬到了；非 0 = 有字段漏搬（打印是哪个）。
"""
import os
import re
import sys

# 需要被"声明 + move 构造转发 + loadCameras 赋值"全链路搬运的字段
GEOM_FIELDS = ["entry_line", "zone_line", "entry_zone_depth", "zone_role",
               "detection_mask", "role"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src", "main_tracking.cpp"))


def slice_between(text, start_pat, end_pat):
    m = re.search(start_pat, text)
    if not m:
        return None
    rest = text[m.end():]
    e = re.search(end_pat, rest)
    return rest[:e.start()] if e else rest


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    struct_body = slice_between(text, r"struct\s+CameraConfig\s*\{", r"\n\};")
    move_init = slice_between(text, r"CameraConfig\(CameraConfig&&\s*o\)\s*noexcept[^\n]*", r"\{")
    loader = slice_between(text, r"loadCameras\s*\([^)]*\)\s*\{", r"\n\s*return out;")

    problems = []
    if struct_body is None:
        problems.append("找不到 struct CameraConfig 定义")
    if move_init is None:
        problems.append("找不到 CameraConfig 的 move 构造函数")
    if loader is None:
        problems.append("找不到 loadCameras() 函数体")
    if problems:
        for p in problems:
            print("[FAIL] " + p)
        return 1

    print("字段搬运自检（%s）" % os.path.relpath(path))
    print("  %-18s %-8s %-10s %s" % ("字段", "已声明", "move转发", "loadCameras赋值"))
    bad = 0
    for fld in GEOM_FIELDS:
        declared = re.search(r"^\s*[A-Za-z_][\w:<>,\s\*&]*\b%s\b\s*[;={]" % re.escape(fld),
                             struct_body, re.M) is not None
        in_move = re.search(r"\b%s\s*\(" % re.escape(fld), move_init) is not None
        in_loader = re.search(r"\bcc\.%s\s*=" % re.escape(fld), loader) is not None
        ok = declared and in_move and in_loader
        if not ok:
            bad += 1
        print("  %-18s %-8s %-10s %s %s" % (
            fld, "Y" if declared else "N", "Y" if in_move else "N",
            "Y" if in_loader else "N", "" if ok else "  <== 漏搬！"))
    print()
    if bad:
        print("==== 结果: %d 个字段漏搬, 检查失败 ====" % bad)
        return 1
    print("==== 结果: %d 个字段全部已搬运, 0 项失败 ====" % len(GEOM_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

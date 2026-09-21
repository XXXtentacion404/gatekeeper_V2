# -*- coding: utf-8 -*-
"""判定线"边界取段"逻辑自检 —— 离线工具，不属于正式测试套件。

用户只做两件事：在检测区边界上点 2 个点。剩下的由 _snap_to_ring / _ring_arc 完成：
  1) 把 2 个点吸附到检测区多边形边界最近的位置；
  2) 取两点之间**较短**的那段边界（中间经过的顶点全部保留）。
这样判定线天然就是"区域的那条边缘"，边缘是折的/弯的都能跟着走。

用法: python _chk_tool_arc.py
"""
import importlib.util
import os
import sys

import numpy as np

FAIL = 0
TOTAL = 0


def check(name, got, want):
    global FAIL, TOTAL
    TOTAL += 1
    ok = (got == want)
    if not ok:
        FAIL += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name}\n        got={got}\n       want={want}")


def load_tool(path):
    spec = importlib.util.spec_from_file_location("gatedraw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只执行模块级定义，不建窗口
    return mod


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        parent = os.path.dirname(root)
        if parent == root:
            raise SystemExit("cameras_config.txt not found")
        root = parent
    mod = load_tool(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"))

    print("=== A. 吸附到边界 ===")
    ring = [(100, 100), (900, 100), (900, 600), (100, 600)]     # 不重复首点的闭合矩形
    # 点在边界上方 30px 处 → 应吸附到上边(边 0)，y 拉回 100
    a = mod._snap_to_ring(ring, (300, 70))
    check("上边外侧 30px 吸附", (a[0], a[2]), (0, (300, 100)))
    # 点在右边右侧 40px → 应吸附到右边(边 1)，x 拉回 900
    b = mod._snap_to_ring(ring, (940, 300))
    check("右边外侧 40px 吸附", (b[0], b[2]), (1, (900, 300)))
    # 点在左上角外侧 → 最近的是角的两个端点之一，只要落在这两条边上即可
    c = mod._snap_to_ring(ring, (60, 60))
    check("左上角外侧吸附落在角上", c[2], (100, 100))

    print("=== B. 取较短的那段（同一条边）===")
    # 两点都在上边 → 近距离直接子段，不该绕整圈
    arc = mod._ring_arc(ring, mod._snap_to_ring(ring, (300, 95)),
                              mod._snap_to_ring(ring, (700, 105)))
    check("同边 2 点 → 直接子段", arc, [(300, 100), (700, 100)])

    print("=== C. 取较短的那段（跨一个角）===")
    # 上边一点 + 右边一点 → 走上边->右上角->右边（短），不该绕另外三条边
    arc = mod._ring_arc(ring, mod._snap_to_ring(ring, (300, 100)),
                              mod._snap_to_ring(ring, (900, 300)))
    check("跨右上角 → 经过角顶点", arc, [(300, 100), (900, 100), (900, 300)])

    print("=== D. 取较短的那段（跨两个角，且反向）===")
    # 右边一点 + 下边一点，反着点也要走短的那段（经过右下角）
    arc = mod._ring_arc(ring, mod._snap_to_ring(ring, (900, 300)),
                              mod._snap_to_ring(ring, (500, 600)))
    check("反序点击 → 仍取短段", arc, [(900, 300), (900, 600), (500, 600)])

    print("=== E. 折线（凹角）边缘：中间顶点必须保留 ===")
    # 一个 L 形边缘：上边 -> 内凹角 -> 竖边 -> 横边；判定线必须贴着这条折线走
    bent = [(100, 100), (500, 100), (500, 300), (900, 300), (900, 600), (100, 600)]
    arc = mod._ring_arc(bent, mod._snap_to_ring(bent, (200, 100)),
                              mod._snap_to_ring(bent, (800, 300)))
    check("折线边缘保留中间顶点",
          arc, [(200, 100), (500, 100), (500, 300), (800, 300)])

    print("=== F. 区域几何：沿 x 平移，y 不变 ===")
    line = [(200, 100), (500, 100), (500, 300), (900, 300)]
    ring2 = mod._zone_ring(line, -100)
    check("区域环顶点数 = 2N", len(ring2), 2 * len(line))
    check("区域环前半 = 判定线", ring2[:len(line)], line)
    check("区域环后半 = 平移副本(逆序, 只改 x)",
          ring2[len(line):], [(x - 100, y) for x, y in reversed(line)])

    print("=== G. 弧长中点所属线段（C++ midSegment 同口径）===")
    # 段长 = 300 / 200 / 400，总长 900，中点 450 → 落在第 1 段（索引 1）
    seg, mid = mod._poly_mid(line)
    check("弧长中点所在段", (seg, mid), (1, (500.0, 250.0)))
    # 2 点直线时退化成线段中点（与旧实现一致）
    check("2 点退化为线段中点", mod._poly_mid([(0, 0), (100, 0)])[1], (50.0, 0.0))

    print(f"\n==== 结果: {TOTAL} 项检查, {FAIL} 项失败 ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

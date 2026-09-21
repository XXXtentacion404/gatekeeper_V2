# -*- coding: utf-8 -*-
"""区域几何变更的跨语言一致性自检（Python 侧）—— 离线工具，不属于正式测试套件。

用 闸门计数-网络摄像头配置文件绘制.py 里的真实几何函数，对同一份配置算出每个网格点的
区域号(0=区域外 1=区域 2=检测区)，输出与 chk_zone.cpp 相同的格式，供 diff 对拍。
复刻 C++ GateIdCounter::regionOf() 的判定顺序与容差：
    R1 = 点落在区域闭环内(带 10px 容差)  >  R2 = 点落在任一检测区多边形内(带 10px 容差)  >  R0
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

TOL = 10.0  # 与 gate_id_counter.cpp 的 kRegionGapTolerance 一致
DEFAULT_DEPTH = 100.0


def load_tool(path):
    spec = importlib.util.spec_from_file_location("gatedraw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只执行模块级定义，不建窗口
    return mod


def parse_config(path):
    cams = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tk = line.split()
            if len(tk) < 4:
                continue
            rest = tk[4:]
            W, H = 1920, 1080
            if rest and "*" in rest[0]:
                W, H = map(int, rest[0].split("*"))
                rest = rest[1:]
            pstr = " ".join(rest)
            polys, line_pts, depth, explicit = [], None, 0.0, False
            s = 0
            while True:
                a = pstr.find("[", s)
                if a < 0:
                    break
                b = pstr.find("]", a)
                if b < 0:
                    break
                pts = []
                for tok in pstr[a + 1:b].split():
                    if "," in tok:
                        x, y = map(int, tok.split(","))
                        pts.append((x, y))
                if len(pts) >= 3:
                    polys.append(pts)
                s = b + 1
            for tok in rest:
                if tok.startswith("t1,"):
                    p = tok[3:].replace(",", " ").split()
                    if len(p) < 4:
                        continue
                    line_pts = [(int(float(p[0])), int(float(p[1]))),
                                (int(float(p[2])), int(float(p[3])))]
                    if len(p) >= 5:
                        depth, explicit = float(p[4]), True
                elif tok.startswith("zin,") or tok.startswith("zout,"):
                    p = tok[4:].replace(",", " ").split()
                    pts = []
                    for k in range(0, len(p) - 1, 2):
                        pts.append((int(float(p[k])), int(float(p[k + 1]))))
                    if len(pts) >= 2:
                        line_pts = pts
                elif tok.startswith("zo,"):
                    depth, explicit = float(tok[3:]), True
            if line_pts and not explicit:
                depth = DEFAULT_DEPTH
            cams.append(dict(ip=tk[0], last3=tk[0].rsplit(".", 1)[-1], W=W, H=H,
                             polys=polys, line=line_pts, depth=depth))
    return cams


def region_of_cpp(cam, pt, ring):
    """复刻 C++ regionOf()：R1 区域 > R2 检测区 > R0"""
    if ring is not None and cv2.pointPolygonTest(ring, pt, True) >= -TOL:
        return 1
    for poly in cam["polys"]:
        arr = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(arr, pt, True) >= -TOL:
            return 2
    return 0


def zone_ring(mod, cam):
    """用工具里的真实函数算区域闭环（= C++ zonePolygon 的口径）。"""
    line = cam["line"]
    if not line:
        return None
    depth = mod._resolve_depth_sign(cam["polys"], line, cam["depth"])
    ring = mod._zone_ring(line, depth)
    return np.array(ring, dtype=np.float32).reshape(-1, 1, 2), depth


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        parent = os.path.dirname(root)
        if parent == root:
            raise SystemExit("cameras_config.txt not found")
        root = parent
    mod = load_tool(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"))
    cfg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "cameras_config.txt")
    cams = parse_config(cfg)
    for cam in cams:
        out = zone_ring(mod, cam)
        ring, depth = out if out else (None, 0.0)
        sys.stderr.write(
            f"[py] {cam['ip']} line={cam['line']} depth={cam['depth']:+.0f} "
            f"-> fixed={depth:+.0f} ring={len(cam['line']) if cam['line'] else 0} 点判定线\n")
        sys.stdout.write(f"# {cam['ip']} W={cam['W']} H={cam['H']} "
                         f"depth={depth:.0f} polys={len(cam['polys'])}\n")
        for y in range(0, cam["H"], 60):
            for x in range(0, cam["W"], 60):
                r = region_of_cpp(cam, (float(x), float(y)), ring)
                sys.stdout.write(f"{cam['last3']} {x} {y} {r}\n")


if __name__ == "__main__":
    main()

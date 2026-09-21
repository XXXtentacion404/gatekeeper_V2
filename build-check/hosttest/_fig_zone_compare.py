# -*- coding: utf-8 -*-
"""生成入口带几何「改前 vs 改后」对照图（离线工具，不属于正式测试套件）。

用 cameras_config.txt 的真实坐标算出两种带，按【列】分开展示，避免两张带叠在一起看不清：
    列1 OLD : 入口带 = t1 沿内侧法向偏移 |depth| px（斜的矩形，矩形边垂直于 t1）
    列2 NEW : 入口带 = t1 的 2 点整体沿 x 平移 |depth| px，y 不变（平行四边形，短边水平）
每行一台相机。裁剪到 t1 附近并放大，便于看清楚带的走向。
    蓝 = 检测区多边形   黄 = t1 标记线   半透明填充 = 入口带
用法: python _fig_zone_compare.py [cameras_config.txt] [out.png]
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROI = {  # 以 t1 为中心、能把两种带都装下的裁剪框 (x0,y0,x1,y1)
    "200": (1040, 620, 1560, 1110),
    "201": (1100, 600, 1620, 1090),
}
SCALE = 1.5
BG = 244


def load_tool(path):
    spec = importlib.util.spec_from_file_location("gatedraw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_config(path):
    cams = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tk = line.split()
        rest = tk[4:]
        if rest and "*" in rest[0]:
            rest = rest[1:]
        pstr = " ".join(rest)
        polys, p1, p2, depth = [], None, None, 0.0
        s = 0
        while True:
            a = pstr.find("[", s)
            if a < 0:
                break
            b = pstr.find("]", a)
            if b < 0:
                break
            pts = [tuple(map(int, t.split(","))) for t in pstr[a + 1:b].split() if "," in t]
            if len(pts) >= 3:
                polys.append(pts)
            s = b + 1
        for tok in rest:
            if tok.startswith("t1,"):
                p = tok[3:].replace(",", " ").split()
                p1, p2 = (int(p[0]), int(p[1])), (int(p[2]), int(p[3]))
                if len(p) >= 5:
                    depth = float(p[4])
        cams.append(dict(last3=tk[0].rsplit(".", 1)[-1], polys=polys,
                         p1=p1, p2=p2, depth=depth))
    return cams


def old_band(mod, cam):
    """改前的实现：沿内侧法向偏移 |depth| px 的矩形（矩形边垂直于 t1）"""
    ux, uy = mod._right_hand_normal(cam["p1"], cam["p2"])
    fixed = mod._resolve_depth_sign(cam["polys"], cam["p1"], cam["p2"], cam["depth"])
    if fixed < 0:
        ux, uy = -ux, -uy
    s = abs(cam["depth"])
    p1, p2 = cam["p1"], cam["p2"]
    return [p1, p2,
            (int(round(p1[0] + ux * s)), int(round(p1[1] + uy * s))),
            (int(round(p2[0] + ux * s)), int(round(p2[1] + uy * s)))]


def new_band(mod, cam):
    """改后的实现：2 点整体沿 x 平移 |depth| px（y 不变）"""
    fixed = mod._resolve_depth_sign(cam["polys"], cam["p1"], cam["p2"], cam["depth"])
    sx = mod._x_offset_from_depth(cam["p1"], cam["p2"], fixed)
    p1, p2 = cam["p1"], cam["p2"]
    return [p1, p2,
            (int(round(p1[0] + sx)), p1[1]),
            (int(round(p2[0] + sx)), p2[1])]


def panel(mod, cam, band, title, accent):
    x0, y0, x1, y1 = ROI[cam["last3"]]
    img = np.full((y1 - y0, x1 - x0, 3), BG, np.uint8)
    sh = np.array([-x0, -y0])

    def P(pts):
        return (np.array([[p[0], p[1]] for p in pts]) + sh).astype(np.int32)

    def ring(b):                      # 存储顺序 -> 首尾相连的环
        return [b[0], b[1], b[3], b[2]]

    for poly in cam["polys"]:
        cv2.polylines(img, [P(poly)], True, (200, 140, 60), 1)
    ov = img.copy()
    cv2.fillPoly(ov, [P(ring(band))], accent)
    cv2.addWeighted(ov, 0.40, img, 0.60, 0, img)
    cv2.polylines(img, [P(ring(band))], True, tuple(int(c * 0.75) for c in accent), 2)
    # 起点/终点之间画出平移向量，直观看出"斜着挪"还是"水平挪"
    for i in (0, 1):
        cv2.arrowedLine(img, tuple(P([band[i]])[0]), tuple(P([band[i + 2]])[0]),
                        tuple(int(c * 0.75) for c in accent), 2,
                        cv2.LINE_AA, 0, 0.14)
    cv2.line(img, tuple(P([cam["p1"]])[0]), tuple(P([cam["p2"]])[0]), (0, 200, 245), 2)
    for p in (cam["p1"], cam["p2"]):
        cv2.circle(img, tuple(P([p])[0]), 4, (0, 0, 0), -1)

    # 入口带内侧边的两个短边端点：NEW 的应落在与起点/终点同一水平线上
    for i in (0, 1):
        cv2.circle(img, tuple(P([band[i + 2]])[0]), 4, tuple(int(c * 0.75) for c in accent), -1)

    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                tuple(int(c * 0.7) for c in accent), 2)
    return np.vstack([bar, img])


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        root = os.path.dirname(root)
    cfg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "cameras_config.txt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        root, "文档", "入口带几何_改前改后对照.png")
    mod = load_tool(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"))
    cams = parse_config(cfg)

    rows = []
    for c in cams:
        bo, bn = old_band(mod, c), new_band(mod, c)
        left = panel(mod, c, bo, f"{c['last3']}  OLD: offset along NORMAL (slanted rect)",
                     (235, 90, 90))
        right = panel(mod, c, bn, f"{c['last3']}  NEW: shifted along X by "
                                  f"{bn[2][0] - c['p1'][0]}px, y unchanged",
                     (60, 175, 60))
        h = max(left.shape[0], right.shape[0])
        left = np.pad(left, ((0, h - left.shape[0]), (0, 0), (0, 0)), constant_values=255)
        right = np.pad(right, ((0, h - right.shape[0]), (0, 0), (0, 0)), constant_values=255)
        rows.append(np.hstack([left, np.full((h, 16, 3), 120, np.uint8), right]))
        print(f"  {c['last3']}  t1={c['p1']}->{c['p2']}")
        print(f"        OLD 带 内侧点 {bo[2]} / {bo[3]}   (偏移向量 {bo[2][0]-c['p1'][0]},{bo[2][1]-c['p1'][1]})")
        print(f"        NEW 带 内侧点 {bn[2]} / {bn[3]}   (偏移向量 {bn[2][0]-c['p1'][0]},{bn[2][1]-c['p1'][1]})")

    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=120) for r in rows]
    img = rows[0]
    for r in rows[1:]:
        img = np.vstack([img, np.full((16, w, 3), 120, np.uint8), r])
    # ⚠️ Windows 上 cv2.imwrite 用窄字符 fopen，中文路径会静默失败；
    #    走 imencode + ndarray.tofile() 才能正确写中文路径。
    cv2.imencode(".png", img)[1].tofile(out)
    print("saved:", out, img.shape)


if __name__ == "__main__":
    main()

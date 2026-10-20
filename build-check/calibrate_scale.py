#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A4 纸标定：量出 px_per_cm，判断 8cm×8cm 当前能做到多少像素。

为什么必须标定
------------------------------------------------------------------
`min_item_px` 是"全分辨率像素"口径，它对应的物理尺寸完全取决于相机几何
（分辨率 / 安装高度 / 视角 / 目标在画面里的位置）。不标定就只能猜：
猜大了小目标被尺寸门限滤掉（漏报），猜小了噪声当目标（误报）。

怎么量（3 分钟）
------------------------------------------------------------------
1. 把 A4 纸**横放**在检测区里（长边 29.7cm 朝向画面左右）；
2. 鼠标从纸的**左边缘拖到右边缘**（尽量水平）；
3. 松开 → 自动记录一次，屏幕上立刻显示像素数和 px_per_cm；
4. **换位置重复**：近端 / 中段 / 远端各量一次。
   ⚠️ 相机斜向下看，近处每厘米占的像素远多于远处，**平均值没有意义，三个数才有意义**；
5. 按 s 打印汇总（另存 calib_result.txt），按 q 退出（也会自动打印）。

用法
------------------------------------------------------------------
    D:/Software/Anaconda3/envs/yolov8/python.exe build-check/calibrate_scale.py --index 0
    ... --index 1                  # 第二台相机
    ... --bg path/to/frame.jpg     # 用指定图片，不重新抓帧（可反复标注）
    ... --cm 21                    # 参考物换成 A4 短边 / 别的尺子
    ... --apply                    # 把建议的 min_item_px 直接写进 abandon_config.txt

不指定 --bg 时：优先用已抓好的 abandon_bg_<IP末段>.jpg（无人干扰、可反复量），
没有就现场抓帧。图像会被缩放到 1280 宽显示，坐标自动换算回原分辨率。

按键
------------------------------------------------------------------
    s        打印汇总并保存 calib_result.txt
    z        撤销上一次测量
    g        显示 / 隐藏检测区轮廓
    q / ESC  退出（自动打印汇总）
"""
import argparse
import io
import os
import re
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("需要 opencv-python + numpy：请用 Anaconda 的 yolov8 环境运行")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grab_bg import ROOT, grab, parse_cams, rtsp_url   # noqa: E402

MAXW = 1280
A4_LONG_CM = 29.7


def read_image(path):
    """中文路径安全的读图：np.fromfile + imdecode。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def report(meas, cm, tag, quiet=False):
    if not meas:
        if not quiet:
            print("\n（还没有任何测量。在图上横向拖一条线，覆盖 A4 纸的左右边缘。）")
        return None
    vals = [px / cm for (_, _, _, _, px) in meas]
    avg = sum(vals) / len(vals)
    px8 = 8.0 * avg
    print("\n=== 标定结果（参考物 {:.1f} cm · 相机 {}）===".format(cm, tag))
    for i, (_x1, _y1, _x2, _y2, px) in enumerate(meas, 1):
        print("  #{:<2} {:>4} px   →   px_per_cm = {:.2f}".format(i, px, px / cm))
    print("  ---- 平均 px_per_cm = {:.2f} ----".format(avg))
    print("  8cm × 8cm ≈ {:.0f} × {:.0f} px".format(px8, px8))
    if px8 >= 20:
        verdict = "可靠 ✓"
    elif px8 >= 15:
        verdict = "勉强 —— 建议 morph_open_px=0、ref_diff_thresh 降到 15~20"
    else:
        verdict = "不可靠 —— 建议提高分辨率 / 调整机位 / 接受只抓大件"
    print("  判定：{}".format(verdict))
    print("  建议 min_item_px = {}（配置当前值：见 abandon_config.txt）".format(int(round(px8))))
    if len(meas) == 1:
        print("  ⚠️ 只量了一个位置。请再到**近端**和**远端**各量一次 —— 斜视角下两者差别很大。")
    elif max(vals) / max(min(vals), 1e-6) >= 1.6:
        print("  ℹ 最近/最远位置相差 {:.1f} 倍 → 检测区里像素密度很不均匀，".format(max(vals) / min(vals)))
        print("    建议按**远端**（最不利位置）的值来定 min_item_px，并接受近端会更灵敏。")
    return int(round(px8))


def apply_min_item(px, path):
    with io.open(path, encoding="utf-8") as f:
        txt = f.read()
    new, n = re.subn(r"(?m)^min_item_px\s*=\s*\d+", "min_item_px={}".format(px), txt)
    if n == 0:
        print("  ✗ 没在 {} 里找到 min_item_px，请手工修改。".format(path))
        return False
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print("  ✓ 已写入 {}：min_item_px={}".format(os.path.basename(path), px))
    print("    → 记得重启板端程序才生效；如需同步到板子：")
    print("      python build-check/board_push.py abandon_config.txt")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "cameras_config.txt"))
    ap.add_argument("--index", type=int, default=0, help="相机行号（从 0 起）")
    ap.add_argument("--bg", default="", help="用指定图片，不重新抓帧")
    ap.add_argument("--cm", type=float, default=A4_LONG_CM, help="参考物实际长度(cm)，默认 A4 长边 29.7")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--apply", action="store_true", help="把建议的 min_item_px 写进 abandon_config.txt")
    args = ap.parse_args()

    cams = parse_cams(args.config)
    if not cams:
        sys.exit("配置里没解析出相机行")
    if not (0 <= args.index < len(cams)):
        sys.exit("--index 越界（共 {} 台）".format(len(cams)))
    cam = cams[args.index]
    cm = args.cm

    # ---- 载入图像 ----
    if args.bg:
        img = read_image(args.bg)
        if img is None:
            sys.exit("读不到 " + args.bg)
        src = args.bg
    else:
        auto = os.path.join(ROOT, "abandon_bg_{}.jpg".format(cam["tag"]))
        if os.path.exists(auto):
            img = read_image(auto)
            src = auto
            print("使用已抓好的背景图：{}".format(os.path.basename(auto)))
        else:
            img = None
            src = ""
        if img is None:
            print("没有现成背景图，现场抓帧 …")
            img, err = grab(rtsp_url(cam), args.frames)
            if img is None:
                sys.exit("抓帧失败：" + err)
            src = "（实时帧）"

    h, w = img.shape[:2]
    scale = min(1.0, MAXW / float(w))
    disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()

    state = {"p1": None, "p2": None, "drag": False, "meas": [], "zone": False}
    poly = cam.get("poly") or []

    def to_img(x, y):
        return int(round(x / scale)), int(round(y / scale))

    def to_disp(v):
        return int(round(v * scale))

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["p1"] = to_img(x, y)
            state["p2"] = state["p1"]
            state["drag"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["p2"] = to_img(x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            state["p2"] = to_img(x, y)
            state["drag"] = False
            if state["p1"] and state["p2"]:
                px = abs(state["p2"][0] - state["p1"][0])
                if px >= 5:
                    state["meas"].append((state["p1"][0], state["p1"][1],
                                          state["p2"][0], state["p2"][1], px))
            state["p1"] = state["p2"] = None

    def render():
        c = disp.copy()
        if state["zone"] and len(poly) >= 2:
            pts = np.array([[to_disp(a), to_disp(b)] for a, b in poly], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(c, [pts], True, (90, 200, 90), 2)
        for (x1, y1, x2, y2, px) in state["meas"]:
            cv2.line(c, (to_disp(x1), to_disp(y1)), (to_disp(x2), to_disp(y2)), (0, 210, 220), 2)
            label = "{}px  {:.2f}px/cm".format(px, px / cm)
            cv2.putText(c, label, (min(to_disp(x1), to_disp(x2)), max(16, to_disp(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 220), 2)
        if state["drag"] and state["p1"] and state["p2"]:
            px = abs(state["p2"][0] - state["p1"][0])
            cv2.line(c, (to_disp(state["p1"][0]), to_disp(state["p1"][1])),
                     (to_disp(state["p2"][0]), to_disp(state["p2"][1])), (0, 0, 235), 2)
            cv2.putText(c, "{}px".format(px),
                        (min(to_disp(state["p1"][0]), to_disp(state["p2"][0])), max(16, to_disp(state["p1"][1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 235), 2)
        cv2.putText(c, "Drag across the A4 width ({:.1f}cm)  |  s=save  z=undo  g=zone  q=quit".format(cm),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        if state["meas"]:
            avg = sum(p for *_x, p in state["meas"]) / len(state["meas"]) / cm
            cv2.putText(c, "n={}  avg {:.2f} px/cm   8cm ~ {:.0f}px".format(
                len(state["meas"]), avg, 8 * avg), (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 130, 30), 2)
        return c

    win = "A4 calibration - {} ({}x{})".format(cam["ip"], w, h)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("图源：{}   分辨率 {}x{}（显示缩放 {:.2f}x）".format(src, w, h, scale))
    print("把 A4 纸横放进检测区，从纸的左边缘拖到右边缘；近端/远端各量一次。")

    suggested = None
    while True:
        cv2.imshow(win, render())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("z") and state["meas"]:
            state["meas"].pop()
        elif k == ord("g"):
            state["zone"] = not state["zone"]
        elif k == ord("s"):
            suggested = report(state["meas"], cm, cam["tag"]) or suggested
    cv2.destroyAllWindows()

    suggested = report(state["meas"], cm, cam["tag"]) or suggested
    if suggested and state["meas"]:
        out = os.path.join(ROOT, "calib_result.txt")
        with io.open(out, "w", encoding="utf-8") as f:
            f.write("相机 {}\n参考物 {:.1f} cm\n测量 {} 次\n".format(cam["tag"], cm, len(state["meas"])))
            for i, (_a, _b, _c, _d, px) in enumerate(state["meas"], 1):
                f.write("  #{:<2} {:>4} px  px_per_cm={:.2f}\n".format(i, px, px / cm))
            f.write("平均 px_per_cm = {:.2f}\n建议 min_item_px = {}\n".format(
                sum(p for *_x, p in state["meas"]) / len(state["meas"]) / cm, suggested))
        print("  ✓ 已保存 {}".format(os.path.basename(out)))
        if args.apply:
            apply_min_item(suggested, os.path.join(ROOT, "abandon_config.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

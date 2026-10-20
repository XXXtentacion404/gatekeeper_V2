#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从海康相机抓一张「标准空场景背景图」，作为滞留检测参考差分的基线（输入端①）。

为什么要在电脑上做这一步
------------------------------------------------------------------
参考差分（导师说的"两输入"方案）= 当前帧 与 一张"画面里没有任何滞留物、
没有行人"的标准背景图 做差分。这张图必须与运行时同一机位、同一光照时段，
所以最省事、最可控的做法就是**用本机 OpenCV 直接抓一帧** —— 等人走空、
按一下命令，比"跑到板子上想办法截图"可靠得多。

用法（在项目根目录）
------------------------------------------------------------------
    python build-check/grab_bg.py                    # 抓全部相机 → abandon_bg_<IP末段>.jpg
    python build-check/grab_bg.py --index 0          # 只抓配置里第 1 行那台
    python build-check/grab_bg.py --index 0 --out abandon_bg.jpg
    python build-check/grab_bg.py --frames 30        # 抓 30 帧逐像素取中值（默认 15，更干净）
    python build-check/grab_bg.py --preview          # 额外输出叠加检测区/判定线的预览图

它同时把「检测区 bbox 的像素尺寸」算出来打印给你 —— 这就是标定的另一半：
量一下检测区的实际物理宽度，就能算出 px_per_cm，填进 abandon_config.txt 的
region_width_cm，板端一启动就会自检 8cm 指标达不达标。

依赖：opencv-python + numpy。推荐用现成环境：
    D:/Software/Anaconda3/envs/yolov8/python.exe build-check/grab_bg.py --index 0
"""
import argparse
import io
import os
import re
import sys
import time
from urllib.parse import quote

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("需要 opencv-python 和 numpy：请用 Anaconda 的 yolov8 环境运行本脚本")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_cams(path):
    """解析 cameras_config.txt：ip user pwd channel WxH [多边形…] <判定线token> [zo,N]"""
    cams = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            p = t.split()
            if len(p) < 5:
                continue
            poly = []
            m = re.search(r"\[([^\]]*)\]", t)
            if m:
                poly = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*,\s*(\d+)", m.group(1))]
            cams.append({
                "ip": p[0], "user": p[1], "pwd": p[2], "channel": p[3],
                "res": p[4], "poly": poly, "raw": t,
                "tag": p[0].split(".")[-1],
            })
    return cams


def rtsp_url(cam):
    """海康 RTSP：主码流 = Streaming/Channels/<channel>01（channel=1 → 101）。"""
    ch = str(cam["channel"]).strip()
    if ch.isdigit() and len(ch) == 1:
        ch = ch + "01"
    elif ch.isdigit() and len(ch) == 3:
        pass
    else:
        ch = "101"
    return "rtsp://{}:{}@{}:554/Streaming/Channels/{}".format(
        quote(cam["user"], safe=""), quote(cam["pwd"], safe=""), cam["ip"], ch)


def grab(url, n_frames, warmup=8):
    """连上抓若干帧，逐像素取中值 —— 中值能自动抹掉偶然走过的零星人影。"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url)          # 退回默认后端再试一次
    if not cap.isOpened():
        return None, "连不上 RTSP（检查 IP/账号/密码/channel，或本机是否与相机同网段）"
    try:
        for _ in range(warmup):              # 丢掉开头几帧（码流刚起来时可能花屏）
            cap.read()
        frames = []
        for _ in range(max(1, n_frames)):
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
            time.sleep(0.03)
    finally:
        cap.release()
    if not frames:
        return None, "连上了但一帧都没读到（码流格式/取流权限问题）"
    if len(frames) == 1:
        return frames[0], None
    return np.median(np.stack(frames, 0), 0).astype(np.uint8), None


def save_jpg(img, path, quality=95):
    """用 imencode + tofile：cv2.imwrite 在中文路径下会静默失败。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    buf.tofile(path)
    return True


def draw_preview(img, cam):
    out = img.copy()
    poly = cam["poly"]
    if len(poly) >= 2:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (60, 200, 60), 2)
        x0, y0 = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        x1, y1 = pts[:, 0, 0].max(), pts[:, 0, 1].max()
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 220), 1)
        cv2.putText(out, "zone bbox {}x{}".format(x1 - x0, y1 - y0),
                    (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 220), 2)
    cv2.putText(out, cam["ip"], (10, out.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "cameras_config.txt"))
    ap.add_argument("--index", type=int, default=-1, help="只抓第 N 行相机（从 0 起）；默认全部")
    ap.add_argument("--out", default="", help="输出文件名（仅在 --index 指定一台时可用）")
    ap.add_argument("--frames", type=int, default=15, help="参与取中值的帧数（默认 15）")
    ap.add_argument("--preview", action="store_true", help="额外输出带检测区叠加的预览图")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit("找不到 " + args.config)
    cams = parse_cams(args.config)
    if not cams:
        sys.exit("配置里没解析出相机行")
    todo = cams if args.index < 0 else [cams[args.index]] if 0 <= args.index < len(cams) else []
    if not todo:
        sys.exit("--index 越界（共 {} 台）".format(len(cams)))
    if args.out and len(todo) > 1:
        sys.exit("--out 只能配合 --index 指定单台相机")

    print("=== 抓标准背景图（人走空、光照稳定时执行）===")
    ok_count = 0
    for cam in todo:
        name = args.out or "abandon_bg_{}.jpg".format(cam["tag"])
        path = name if os.path.isabs(name) else os.path.join(ROOT, name)
        print("\n[{}] 取流中 … {}".format(cam["ip"], rtsp_url(cam).split("@")[-1]))
        img, err = grab(rtsp_url(cam), args.frames)
        if img is None:
            print("   ✗ " + err)
            continue
        h, w = img.shape[:2]
        save_jpg(img, path)
        print("   ✓ {}   {}x{}".format(os.path.basename(path), w, h))
        ok_count += 1

        if cam["poly"]:
            xs = [p[0] for p in cam["poly"]]
            ys = [p[1] for p in cam["poly"]]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            print("   检测区 bbox = {}x{} px  （x {}~{} , y {}~{}）".format(
                bw, bh, min(xs), max(xs), min(ys), max(ys)))
            print("   → 量出检测区横向实际宽度 W_cm 后：px_per_cm = {:.1f} / W_cm".format(bw))
            print("     例：W_cm=260 → px_per_cm={:.2f} → 8cm ≈ {:.0f}px（≥20px 可靠，<15px 不可靠）"
                  .format(bw / 260.0, 8 * bw / 260.0))
            print("     把 W_cm 填进 abandon_config.txt 的 region_width_cm，板端启动会自检")
            if args.preview:
                pv = os.path.join(ROOT, "abandon_bg_{}_preview.jpg".format(cam["tag"]))
                save_jpg(draw_preview(img, cam), pv)
                print("   预览（含检测区叠加）：{}".format(os.path.basename(pv)))

    print("\n=== 完成：{} / {} 台成功 ===".format(ok_count, len(todo)))
    if ok_count:
        print("下一步：")
        print("  1. 把生成的 jpg 传到板端：")
        print("     python build-check/board_push.py abandon_bg_<末段>.jpg")
        print("  2. 板端 abandon_config.txt 里填 ref_bg_path=/home/topeet/code/gatekeeper_V4/abandon_bg_<末段>.jpg")
        print("     和 region_width_cm=<你量出的物理宽度>")
        print("  3. 两端都要改 → 本地改完再推一次：python build-check/board_push.py --build")
    return 0


if __name__ == "__main__":
    sys.exit(main())

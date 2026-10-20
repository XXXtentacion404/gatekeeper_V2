# -*- coding: utf-8 -*-
"""
第 2 课 · 合成训练数据
=====================================================================
训练一个"变化检测"网络，最重要的不是网络结构，是数据。
这一课先不碰网络，只解决一件事：**怎么造出 (背景图, 当前图, 掩膜) 这一对半的数据**。

为什么必须合成：
    现场要拍到"有人把东西放下然后走了"的画面，一天也出不了几条。
    但背景是固定的——只要有一张干净的空景图，就可以在上面**任意贴物体**，
    贴多少个、贴多大、贴哪里、贴什么颜色，全都是随机的。
    在线合成 = 数据集没有上限。

【输入】abandon_bg_200.jpg  —— 现场空景原图，1920×1080
        这是你自己已经采好的参考背景图，直接拿来当底。

【输出】学习/合成预览.jpg    —— 左：背景图  中：贴了东西的当前图  右：掩膜
        看一眼就知道合成得对不对。

【跑法】在项目根目录执行：
    D:\\Software\\Anaconda3\\envs\\yolov8\\python.exe 学习\\02_合成数据.py
"""

import cv2
import numpy as np

# ---------------- 现场标定值（来自 item_abandonment 的 ROI 裁剪） ----------------
BG_PATH = "abandon_bg_200.jpg"
ROI_X, ROI_Y, ROI_W, ROI_H = 472, 274, 1088, 806   # 200 号相机的检测区 bbox
OUT_W, OUT_H = 1024, 768                            # 网络输入分辨率（不降采样，只轻微缩放）

# 物体尺寸：8cm ≈ 18px，现场物品大概 12~60px（8cm ~ 30cm）
ITEM_MIN, ITEM_MAX = 12, 60
MAX_ITEMS = 3                                       # 一张图最多贴几个


def load_bg(path, x, y, w, h, out_w, out_h):
    """读空景图 → 裁 ROI → 缩到网络输入尺寸。这就是"背景图 ref"。"""
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(path)
    roi = raw[y:y + h, x:x + w]
    return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)


def paste_items(ref, rng):
    """
    在 ref 上随机贴几个小块，同时生成掩膜。
    返回 (当前图 cur, 掩膜 mask)。
    """
    cur = ref.copy()
    mask = np.zeros(ref.shape[:2], np.uint8)

    n = rng.integers(1, MAX_ITEMS + 1)
    for _ in range(n):
        s = int(rng.integers(ITEM_MIN, ITEM_MAX + 1))
        x = int(rng.integers(0, ref.shape[1] - s))
        y = int(rng.integers(0, ref.shape[0] - s))

        # 前景内容：从背景别处随机裁一块 → 自带真实纹理，比纯色块像东西
        sx = int(rng.integers(0, ref.shape[1] - s))
        sy = int(rng.integers(0, ref.shape[0] - s))
        patch = ref[sy:sy + s, sx:sx + s].astype(np.float32)

        # 亮度和色调抖一下，避免网络学到"只有变暗才是物体"
        patch = patch * float(rng.uniform(0.6, 1.5)) + float(rng.uniform(-25, 25))
        patch = np.clip(patch, 0, 255).astype(np.uint8)

        cur[y:y + s, x:x + s] = patch
        mask[y:y + s, x:x + s] = 255

    return cur, mask


def main():
    rng = np.random.default_rng(0)
    ref = load_bg(BG_PATH, ROI_X, ROI_Y, ROI_W, ROI_H, OUT_W, OUT_H)
    cur, mask = paste_items(ref, rng)

    print("背景图 ref.shape =", ref.shape)
    print("当前图 cur.shape =", cur.shape)
    print("掩膜  mask.shape =", mask.shape, " dtype =", mask.dtype)
    print("掩膜里为 255 的像素占比 = %.4f%%" % (100.0 * (mask > 0).mean()))

    # 拼成三联图看效果：背景 | 当前 | 掩膜
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    preview = np.hstack([ref, cur, mask3])
    preview = cv2.resize(preview, (preview.shape[1] // 2, preview.shape[0] // 2))
    ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    buf.tofile("学习/合成预览.jpg")     # ⚠️ 中文路径必须走 imencode，cv2.imwrite 会静默失败
    print("已写 学习/合成预览.jpg  尺寸 =", preview.shape[:2])


if __name__ == "__main__":
    main()

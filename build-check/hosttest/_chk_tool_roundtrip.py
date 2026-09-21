# -*- coding: utf-8 -*-
"""闸门计数绘制工具的端到端回环自检（离线工具，不属于正式测试套件）。

验证 加载(cameras_config.txt) -> 内部几何 -> 保存 之后，配置文件是否字节级往返，
以及区域是否确实是"判定线整体沿 x 方向平移（y 不变）"。

覆盖两种格式：
  A) 旧格式 t1（2 点直线 + depth）——必须原样往返，板端行为不变；
  B) 新格式 zin,/zout,（区域边缘折线，点数不限）+ zo, ——画布重画后应稳定往返。

用 offscreen 平台跑真实的 MainWindow，不弹窗、不需要显示器。

用法: python _chk_tool_roundtrip.py [cameras_config.txt]
"""
import difflib
import importlib.util
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

FAIL = 0
TOTAL = 0


def check(name, got, want):
    global FAIL, TOTAL
    TOTAL += 1
    ok = (got == want)
    if not ok:
        FAIL += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + ("" if ok else f"  got={got} want={want}"))
    return ok


def load_tool(path):
    spec = importlib.util.spec_from_file_location("gatedraw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_root():
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        parent = os.path.dirname(root)
        if parent == root:
            raise SystemExit("cameras_config.txt not found")
        root = parent
    return root


def run_roundtrip(mod, cfg, out):
    """加载 -> 保存，返回 (load 出来的 cams, 两个文件的 token 列表)"""
    mod.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (cfg, ""))
    mod.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))
    mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    mod.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = mod.MainWindow()
    w._load()
    cams = [dict(c) for c in w.cams]
    w._save()
    a = open(cfg, encoding="utf-8").read().split()
    b = open(out, encoding="utf-8").read().split()
    return cams, a, b


def main():
    global FAIL
    root = find_root()
    here = os.path.dirname(os.path.abspath(__file__))
    mod = load_tool(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"))
    app = QApplication(sys.argv)          # noqa: F841  (offscreen，仅需存在)

    # ---------- A. 真实配置文件：加载 -> 区域几何 -> token 往返 ----------
    # 注意：仓库里的 cameras_config.txt 现在是**新格式**（zin,/zout, 折线 + zo,），
    # 所以这里不能写死"2 点"，要按 N 点通用断言（2026-09-16 起为 59 顶点边界弧）。
    cfg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "cameras_config.txt")
    out = os.path.join(here, "_roundtrip_out.txt")
    print(f"=== A. 真实配置回环: {os.path.basename(cfg)} ===")
    cams, a, b = run_roundtrip(mod, cfg, out)
    print(f"  加载 {len(cams)} 台相机")
    for c in cams:
        line = c.get("zone_line")
        role = c.get("zone_role")
        if not line or len(line) < 2:
            check(f"{c['ip']} 解析出 ≥2 点判定线", line, "≥2 点")
            continue
        check(f"{c['ip']} 角色合法({role})", role in ("entry", "exit"), True)
        # 区域环 = 判定线 + 其整体沿 x 平移(dx,0) 的**逆序**副本
        sx = mod._x_offset_from_depth(line, c["zone_depth"])
        ring = mod._zone_ring(line, c["zone_depth"])
        want = list(line) + [(x + sx, y) for x, y in reversed(line)]
        check(f"{c['ip']} 区域环 = 判定线 + 逆序平移副本(N={len(line)}, dx={sx:+.0f})",
              [(round(x, 3), round(y, 3)) for x, y in ring],
              [(round(x, 3), round(y, 3)) for x, y in want])
    check("真实配置 加载->保存 token 完全一致", b, a)
    if b != a:
        for ln in list(difflib.unified_diff(a, b, "原", "新", lineterm=""))[:14]:
            print("      " + ln)

    # ---------- A2. 旧格式 t1 仍然兼容（合成配置）----------
    print("=== A2. 旧格式 t1 兼容（合成配置）===")
    poly_a = "[600,200 1400,200 1400,900 600,900]"
    cfg_a = os.path.join(here, "_t1_synth_in.txt")
    out_a = os.path.join(here, "_t1_synth_out.txt")
    with open(cfg_a, "w", encoding="utf-8") as f:
        f.write("10.60.87.201 admin peng2176 1 1920*1080 " + poly_a + " t1,600,200,1400,200,100\n")
    cams_a, ta, tb = run_roundtrip(mod, cfg_a, out_a)
    la = cams_a[0].get("zone_line") if cams_a else None
    check("旧格式 t1 仍能解析出判定线(≥2 点)", bool(la) and len(la) >= 2, True)
    if la:
        check("旧格式 t1 首尾端点落在原 t1 两端",
              (abs(la[0][0] - 600) <= 2 and abs(la[0][1] - 200) <= 2 and
               abs(la[-1][0] - 1400) <= 2 and abs(la[-1][1] - 200) <= 2), True)
    check("旧格式 t1 token 原样往返", [t for t in tb if t.startswith("t1,")],
          [t for t in ta if t.startswith("t1,")])

    # ---------- B. 新格式 zin/zout（合成配置）----------
    print("=== B. 新格式 zin/zout 回环（合成配置）===")
    poly = "[600,200 1400,200 1400,900 600,900]"
    # 201 = 入口相机：判定线是"上边 + 右上角 + 右边"这条折线（3 个顶点，不是直线）
    # 200 = 出口相机：判定线是"左下角到右下角"的下边
    zin = "zin,700,200,1400,200,1400,500"
    zout = "zout,700,900,1400,900"
    cfg2 = os.path.join(here, "_zone_synth_in.txt")
    out2 = os.path.join(here, "_zone_synth_out.txt")
    with open(cfg2, "w", encoding="utf-8") as f:
        f.write("10.60.87.201 admin peng2176 1 1920*1080 " + poly + " " + zin + "\n")
        f.write("10.60.87.200 admin peng2176 1 1920*1080 " + poly + " " + zout + "\n")
    cams2, a2, b2 = run_roundtrip(mod, cfg2, out2)
    check("解析出 2 台相机", len(cams2), 2)
    check("201 判定线 = 3 顶点折线", cams2[0]["zone_line"],
          [(700, 200), (1400, 200), (1400, 500)])
    check("201 角色 = 入口", cams2[0]["zone_role"], "entry")
    check("200 判定线 = 2 顶点", cams2[1]["zone_line"], [(700, 900), (1400, 900)])
    check("200 角色 = 出口", cams2[1]["zone_role"], "exit")
    check("201 区域环顶点数 = 2N", len(mod._zone_ring(cams2[0]["zone_line"],
                                                     cams2[0]["zone_depth"])),
          2 * len(cams2[0]["zone_line"]))
    # 新格式会显式补出 zo,<平移量>（让文件自描述），所以第一次保存不是逐字往返；
    # 要求的是**幂等**：把刚写出的文件再加载->保存，必须逐字不变。
    check("新格式已写出显式 zo,", [t for t in b2 if t.startswith("zo,")], ["zo,-100", "zo,100"])
    cams3, a3, b3 = run_roundtrip(mod, out2, os.path.join(here, "_zone_synth_out2.txt"))
    check("新格式 幂等（再加载->保存 token 一致）", b3, a3)
    if b3 != a3:
        for ln in list(difflib.unified_diff(a3, b3, "原", "新", lineterm=""))[:14]:
            print("      " + ln)
    print("  写出内容: " + open(out2, encoding="utf-8").read().strip().replace("\n", "\n            "))

    print(f"\n==== 结果: {TOTAL} 项检查, {FAIL} 项失败 ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

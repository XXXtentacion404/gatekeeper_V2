// ============================================================================
// GateIdCounter 宿主侧功能测试（合成轨迹，直接驱动真实 src/gate_id_counter.cpp）
//
// 触发源 = 本相机标记线(t1)的"侧翻转边沿"：每跨一次线恰好计一次，与 ID 生命周期解耦。
// ⚠️ 跨线还要"站住"：新的一侧连续观测 ≥ cross_dwell_frames 帧才结算（默认 3 = 历史行为；
//    板端 kCrossDwellFrames=25）。没站住 = 在线上蹭/徘徊 → 整对事件作废。见 T49~T52。
//
// 覆盖场景：
//   T1  正向进入(跨线进) + 跨线出离开（立即结算）
//   T2  入口带逆行离开(2→1→0)→ via_entry_band 标记
//   T3  遮挡致 ID 切换 → 新 ID 在检测区内出生 → 继承所在侧，不重不漏
//   T4  出生即在检测区内 → 不凭空计 IN（但仍按跨出线计一次离开类）
//   T5  边界灰区抖动(|signed|<20px) → 不改侧、不误判、保持"在区域内"标记
//   T6  门口徘徊但不跨线(始终同侧/死区内) → 0 事件
//   T7  轨迹在检测区内消亡(未跨出线) → 不补任何事件(不抢对面相机的 OUT)
//   T8  用户示例：区域内 4 人，3 号遮挡切换为 5 号 → 不重计不漏计
//   T9  出口相机角色(200) 标签映射对称
//   T10 跨出线即结算(不再需要 15 帧深在外)
//   T11 已进入者消失在入口带内(未跨出线) → 不计离开类
//   T12 人从对面一侧(远端)走出检测区 → 本相机 0 事件
//   T13 人从对面一侧(远端)进入检测区 → 本相机 0 事件
//   T14 出口相机：人从对面一侧离场 → 不得计 OUT
//   T15 跨线途中换 ID → 继承所在侧，补上进入类
//   T16 跨出线但确认帧数没凑够即消失 → 消失回补一次离开类
//   T17 ★单人同一 ID 多次往返 → 每次跨线都计数（复现并验证本次 bug 的修复）
//   T18 ★出口相机单人多次往返 → OUT/RETRO 每次都计（不再漏计 OUT）
//   T19 闸口跨度外横向走动(延长线上跨线) → 不计数（跨度约束）
//   T20 ★跟丢越线推断【默认关闭】→ 跨线瞬间跟丢不再补计（宁可漏计，不虚计；P12 已知代价）
//   T20b 显式打开开关 → 同一轨迹仍补 1 次离开类（机制保留，供将来"推断+实时观测"复用）
//   T21 反例：在检测区深处消亡(离线段远) → 开关打开时也不得推断（守住 T7 语义）
//   T22 反例：走到门口后站定再消失(帧间速度归零) → 开关打开时也不得推断（守住 T11 语义）
//   T23 ★出口相机(200) 对称：默认关闭不补 OUT；显式打开后才补
//   T24 防重复：跨线瞬间跟丢但【有后继 ID】→ 由交接计一次，推断不得再补
//   T25 ★第 2 趟跨进即换 ID(entered 已 true) → 兑付仍须补计（修 M1 守卫缺陷）
//   T26 ★第 2 次跨出即换 ID(exited 已 true) → 兑付仍须补计（出方向对称）
//   T27 ★判定线为折线(区域边缘本身就是判定线) → 跨过它恰好计一次
//   T28 折线跨度之外(延长线远端)移动 → 不计数（跨度约束对折线同样成立）
//   T29 折角处跨线(命中线段发生切换) → 仍只计一次，不因换段双计
//   T30 折线 + 出口相机角色 → 标签映射对称
//   T31 ★走到判定线附近但没跨出去就消失 → 默认与打开开关都不得计数（现场假 OUT:6→7 复现）
//   T32 T31 对照组：慢速但马上要跨过去 → 默认不计；打开开关后必须计（证明判据没被废掉）
//   T33 ★★用户方案：跨出瞬间被吞 → ByteTrack 重新跟回来且已在线外 → 恰好补 1 次 OUT
//       （判据 = 两个真实观测脚底连线穿线；有实时依据，与 T20 的纯外推推断不同）
//   T34 守卫：后继 ID 离失联位置 >180px → 不认交接（防"换了 ID 就补"变兜底规则）
//   T35 守卫：换 ID 前后始终在线外 → 无穿线 → 不计数（触发源是穿线，不是换 ID）
//   T49 ★在线上蹭/徘徊（跨过去没站住就回来）→ 这一对事件整体作废，0 事件
//   T50 dwell=25 下真跨线仍照常计（进出各 1 次）
//   T51 dwell 阈值边界：新侧站住 24 帧不计 / 26 帧计（钉死语义）
//   T52 对照：门槛退回 dwell=3(历史行为) → 同一段徘徊确实会误计（现场 IN 3→4 的成因）
//
// 几何（1920x1080）：
//   检测区多边形 = (600,300)-(1400,300)-(1400,900)-(600,900)
//   入口绊线 t1 = (600,300)->(600,900), depth=+100 → 入口带 x∈[600,700]
//   x < 600          → R0 区域外（= 出口）
//   x ∈ [600,700]     → R1 入口带
//   x ∈ (700,1400]    → R2 检测区内
//   t1 跨度判据：脚底 y ∈ [300-50, 900+50]
//   折线判定线 = (600,300)->(700,500)->(850,900)（T27~T30）
// ============================================================================
#include "gate_id_counter.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        std::printf("  [FAIL] %s\n", what.c_str());
    } else {
        std::printf("  [ ok ] %s\n", what.c_str());
    }
}

cv::Mat makeMask() {
    cv::Mat mask = cv::Mat::zeros(1080, 1920, CV_8UC1);
    std::vector<cv::Point> poly{{600, 300}, {1400, 300}, {1400, 900}, {600, 900}};
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
    return mask;
}

Tripwire makeEntryLine() { return Tripwire(600.f, 300.f, 600.f, 900.f); }

TrackedObject makeTrack(int id, float foot_x, float foot_y,
                        bool observed = true, float conf = 0.8f, int hit = 12) {
    const int w = 100, h = 200;
    TrackedObject t;
    t.track_id = id;
    t.observed = observed;
    t.confidence = conf;
    t.hit_streak = hit;
    t.class_id = 0;
    t.class_name = "people";
    t.box = cv::Rect((int)(foot_x - w / 2.0f), (int)(foot_y - h * 0.92f), w, h);
    return t;
}

struct Harness {
    GateIdCounter counter;
    int rid = 0;
    std::vector<GateCountEvent> events;
    std::string all_notes;          // 累积每帧的 GateIdCounter::notes()（诊断注记）
    int in = 0, out = 0, retro = 0;

    Harness(GateRole role, int dwell = GateIdCounter::kDefaultCrossDwellFrames)
        : counter(role, makeEntryLine(), 100.0f, makeMask(), dwell) {}

    std::vector<GateCountEvent> step(const std::vector<TrackedObject>& tracks) {
        std::vector<TrackedObject> t = tracks;
        rid++;
        for (auto& x : t) x.result_frame_id = rid;
        auto ev = counter.update(t);
        for (const auto& n : counter.notes()) { all_notes += n; all_notes += '\n'; }
        for (const auto& e : ev) {
            events.push_back(e);
            if (e.type == GATE_EVENT_IN) in++;
            else if (e.type == GATE_EVENT_OUT) out++;
            else retro++;
        }
        return ev;
    }

    void run(int frames, const std::vector<TrackedObject>& tracks) {
        for (int i = 0; i < frames; i++) step(tracks);
    }

    // 连续帧里只有部分 id 在场（其余缺席，用于制造遮挡/离场）
    void runMix(int frames, const std::vector<TrackedObject>& tracks) { run(frames, tracks); }

    GateIdMark mark(int id) {
        for (const auto& m : counter.marks())
            if (m.track_id == id) return m;
        GateIdMark empty;
        empty.track_id = -1;
        return empty;
    }

    std::vector<GateCountEvent> eventsFor(int id) const {
        std::vector<GateCountEvent> out_ev;
        for (const auto& e : events)
            if (e.track_id == id) out_ev.push_back(e);
        return out_ev;
    }
};

// ---- 折线判定线（区域边缘本身就是判定线，点数不限）----
// 3 顶点、带一个折角；弧长中点落在接近竖直的第 2 段上，故区域 = 折线 ∪ 其沿 x 平移 100px
// 的副本不会自交。用来验证"判定线不是直线"时跨线判定的行为（命中线段可能切换）。
std::vector<cv::Point2f> makeBentLine() {
    return {cv::Point2f(600.f, 300.f), cv::Point2f(700.f, 500.f),
            cv::Point2f(850.f, 900.f)};
}

struct BentHarness {
    GateIdCounter counter;
    int rid = 0;
    std::vector<GateCountEvent> events;
    int in = 0, out = 0, retro = 0;

    explicit BentHarness(GateRole role)
        : counter(role, makeBentLine(), 100.0f, makeMask()) {}

    void step(const std::vector<TrackedObject>& tracks) {
        std::vector<TrackedObject> t = tracks;
        rid++;
        for (auto& x : t) x.result_frame_id = rid;
        for (const auto& e : counter.update(t)) {
            events.push_back(e);
            if (e.type == GATE_EVENT_IN) in++;
            else if (e.type == GATE_EVENT_OUT) out++;
            else retro++;
        }
    }

    void run(int frames, const std::vector<TrackedObject>& tracks) {
        for (int i = 0; i < frames; i++) step(tracks);
    }
};

void header(const char* name) { std::printf("\n=== %s ===\n", name); }

// ---------------------------------------------------------------- T1
void t1_entry_normal_exit() {
    header("T1 正向进入(跨线进) + 跨线出离开（立即结算）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(4, {makeTrack(1, 550, 600)});   // 线外侧，确立所在侧(不发事件)
    h.run(4, {makeTrack(1, 650, 600)});   // 跨进线内 → IN
    h.run(6, {makeTrack(1, 800, 600)});   // R2 检测区内
    check(h.in == 1 && h.retro == 0 && h.out == 0, "跨进线内 → 恰好 1 次 IN");
    check(h.mark(1).inside && h.mark(1).region == GATE_REGION_DETECT,
          "ID 被标记为在区域内(R2/INSIDE)");
    h.run(4, {makeTrack(1, 500, 600)});   // 跨出本相机标记线
    check(h.retro == 1, "跨出标记线 → 立即计 1 次离开类(201 侧 = RETRO)");
    check(h.events.back().reason == "leave-cross-line" ||
              h.events.back().reason == "leave-cross-line-via-entry",
          "离开依据 = 跨线");
    h.run(5, {});                          // ID 消失
    check(h.retro == 1, "ID 消失不重复计数");
    check(h.in == 1 && h.out == 0, "IN 仍为 1，无重复计");
    check(h.mark(1).inside == false, "离开后标记翻转为区域外");
}

// ---------------------------------------------------------------- T2
void t2_retro_exit_via_entry_band() {
    header("T2 入口带逆行离开(2→1→0)");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(2, 550, 600)});
    h.run(3, {makeTrack(2, 660, 600)});
    h.run(6, {makeTrack(2, 800, 600)});
    check(h.in == 1, "先正常进入 → IN=1");
    h.run(4, {makeTrack(2, 660, 600)});   // 回到入口带
    h.run(4, {makeTrack(2, 560, 600)});   // 穿过绊线到区域外
    h.run(5, {});
    check(h.retro == 1, "逆行离开 → 恰好 1 次离开类");
    check(h.events.back().via_entry_band, "标记离开路径经过入口带");
    check(h.events.back().region == GATE_REGION_OUTSIDE, "离开判定区域 = R0");
}

// ---------------------------------------------------------------- T3
void t3_occlusion_handover() {
    header("T3 遮挡致 ID 切换(3→5)：新 ID 在检测区内出生 → 继承旧 ID 状态");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(3, 550, 600)});
    h.run(3, {makeTrack(3, 660, 600)});
    h.run(6, {makeTrack(3, 850, 600)});
    check(h.in == 1, "3 号进入 → IN=1");

    // 遮挡：先预测帧(observed=false)，再彻底消失 2 帧
    h.run(3, {makeTrack(3, 850, 600, /*observed=*/false)});
    h.run(2, {});
    check(h.in == 1 && h.retro == 0, "遮挡期间不产生任何计数");

    // 新 ID 5 在检测区内出生（距 3 号最后脚底 3px）
    h.run(2, {makeTrack(5, 853, 600)});
    check(h.in == 1, "5 号继承 entered → 不重复计 IN");
    check(h.mark(5).inherited && h.mark(5).origin_id == 3, "5 号标记继承自 3 号");
    check(h.mark(5).inside, "5 号被标记为在区域内");

    h.run(4, {makeTrack(5, 500, 600)});
    h.run(5, {});
    check(h.retro == 1, "5 号离开 → 恰好 1 次离开类(与 3 号合计仍 1 次)");
    check(h.in == 1 && h.out == 0, "全程 IN=1 / 其它=0，不重不漏");
    check(h.events.back().origin_id == 3, "离开事件携带继承来源 ID=3");
}

// ---------------------------------------------------------------- T4
void t4_born_inside_no_inherit_no_phantom_count() {
    header("T4 新生 ID 直接出现在检测区内 → 不凭空计 IN");
    Harness h(GATE_ROLE_ENTRY);
    h.run(10, {makeTrack(7, 800, 600)});           // 出生即在 R2
    check(h.in == 0, "首次确立侧(内侧)不发事件 → 不计 IN（防遮挡致重复计）");
    h.run(4, {makeTrack(7, 500, 600)});            // 跨出标记线
    check(h.retro == 1 && h.in == 0,
          "跨出本相机标记线仍计一次离开类(逆行出)，但进入类不计");
}

// ---------------------------------------------------------------- T5
void t5_boundary_jitter_keeps_inside_mark() {
    header("T5 边界灰区抖动 → 不误判离开，ID 保持'在区域内'标记");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(9, 550, 600)});
    h.run(3, {makeTrack(9, 660, 600)});
    h.run(6, {makeTrack(9, 800, 600)});
    check(h.in == 1, "进入 → IN=1");
    h.run(6, {makeTrack(9, 585, 600)});   // 多边形外 15px：在容差灰区内
    check(h.mark(9).inside, "灰区(未达清晰在外 25px) → 仍标记在区域内");
    check(h.retro == 0, "灰区不结算离开");
    h.run(6, {makeTrack(9, 800, 600)});   // 回到检测区内
    check(h.mark(9).inside && h.retro == 0, "返回后标记仍为区域内，无多余事件");
    h.run(4, {makeTrack(9, 450, 600)});
    h.run(5, {});
    check(h.retro == 1, "真正离场 → 恰好 1 次离开类");
}

// ---------------------------------------------------------------- T6
void t6_loiter_without_crossing() {
    header("T6 门口徘徊但不跨线(死区内震荡 / 始终同侧) → 不误计");
    Harness h(GATE_ROLE_ENTRY);
    for (int i = 0; i < 4; i++) {
        h.run(3, {makeTrack(11, 585, 600)});   // signed=-15，±20 死区内
        h.run(3, {makeTrack(11, 615, 600)});   // signed=+15，±20 死区内
    }
    check(h.in == 0 && h.out == 0 && h.retro == 0, "死区内震荡 → 0 事件");
    for (int i = 0; i < 4; i++) {
        h.run(4, {makeTrack(12, 400, 600)});   // 始终线外侧
        h.run(4, {makeTrack(12, 560, 600)});   // 仍在线外侧(signed=-40)
    }
    h.run(5, {});
    check(h.in == 0 && h.out == 0 && h.retro == 0, "始终同侧徘徊 → 0 事件");
}

// ---------------------------------------------------------------- T7
void t7_no_settle_when_dying_inside() {
    header("T7 轨迹在检测区内消亡 → **不**补发离开类(每台相机只认自己跨出的那一步)");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(13, 550, 600)});
    h.run(3, {makeTrack(13, 660, 600)});
    h.run(6, {makeTrack(13, 900, 600)});
    check(h.in == 1, "进入 → IN=1");
    h.run(94, {});                         // 长时间无任何观测
    check(h.events.size() == 1 && h.retro == 0,
          "人没跨出本相机标记线就消失 → 不得补离开类(否则 201 会抢 200 的 OUT)");
}

// ---------------------------------------------------------------- T8
void t8_four_people_with_id_switch() {
    header("T8 用户示例：区域内 4 人，3 号遮挡切换为 5 号(在检测区内出生)");
    Harness h(GATE_ROLE_EXIT);   // 出口相机 200：进入类=RETRO，离开类=OUT
    // 4 条互不重叠的通道（y 不同），便于构造"4 人同时在区域内"
    const float lanes[4] = {350, 500, 650, 800};
    const float inside_x[4] = {800, 950, 1100, 1250};
    std::vector<TrackedObject> all;
    auto stage = [&](float x, int id_override = -1, float x_override = -1) {
        std::vector<TrackedObject> v;
        for (int i = 0; i < 4; i++) {
            const int id = (id_override == i) ? 5 : i + 1;
            const float px = (x_override >= 0 && id_override == i) ? x_override : x;
            v.push_back(makeTrack(id, px, lanes[i]));
        }
        return v;
    };
    h.run(3, stage(550));            // 4 人都在区域外
    h.run(3, stage(660));            // 4 人都在入口带
    std::vector<TrackedObject> inside;
    for (int i = 0; i < 4; i++) inside.push_back(makeTrack(i + 1, inside_x[i], lanes[i]));
    h.run(8, inside);                // 4 人都在检测区内
    check(h.retro == 4, "4 人进入 → 4 次进入类(200 侧=RETRO)");
    check(h.counter.insideCount() == 4, "区域内 ID 计数 = 4");

    // 3 号被遮挡，切换为 5 号（在检测区内出生，距 3 号很近）
    std::vector<TrackedObject> occluded = inside;
    occluded[2] = makeTrack(3, inside_x[2], lanes[2], /*observed=*/false);
    h.run(3, occluded);              // 预测框维持 3 号
    std::vector<TrackedObject> gone = inside;
    gone.erase(gone.begin() + 2);
    h.run(2, gone);                  // 3 号彻底消失
    std::vector<TrackedObject> switched = gone;
    switched.push_back(makeTrack(5, inside_x[2] + 4, lanes[2]));
    h.run(6, switched);
    check(h.retro == 4, "切换后不重复计进入类");
    check(h.counter.insideCount() == 4, "区域内 ID 计数仍为 4(5 号继承 3 号标记)");
    check(h.mark(5).origin_id == 3 && h.mark(3).exited == false, "5 号继承自 3 号");

    // 4 人一起从检测区走出：入口带 → 区域外 → ID 消失
    std::vector<TrackedObject> out_band;
    for (int i = 0; i < 4; i++)
        out_band.push_back(makeTrack(i == 2 ? 5 : i + 1, 660, lanes[i]));
    h.run(3, out_band);
    std::vector<TrackedObject> out_side;
    for (int i = 0; i < 4; i++)
        out_side.push_back(makeTrack(i == 2 ? 5 : i + 1, 520, lanes[i]));
    h.run(4, out_side);
    check(h.counter.insideCount() == 0, "清晰离开后区域内 ID 计数归零");
    h.run(5, {});
    check(h.out == 4, "4 人离开 → 恰好 4 次离开类(OUT)");
    check(h.retro == 4, "进入类仍为 4(不重不漏)");
}

// ---------------------------------------------------------------- T9
void t9_exit_camera_role_labels() {
    header("T9 出口相机(200)角色标签对称");
    Harness h(GATE_ROLE_EXIT);
    h.run(3, {makeTrack(21, 550, 600)});
    h.run(3, {makeTrack(21, 660, 600)});
    h.run(6, {makeTrack(21, 800, 600)});
    check(h.retro == 1 && h.in == 0, "0→1→2 = RETRO");
    h.run(4, {makeTrack(21, 500, 600)});
    h.run(5, {});
    check(h.out == 1, "2→0 消失 = OUT");
}

// ---------------------------------------------------------------- T10
void t10_cross_out_immediate_settlement() {
    header("T10 跨出标记线即结算（不再需要 15 帧深在外）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(31, 550, 600)});
    h.run(3, {makeTrack(31, 660, 600)});
    h.run(6, {makeTrack(31, 900, 600)});
    check(h.in == 1, "进入 → IN=1");
    h.run(6, {makeTrack(31, 450, 600)});    // 跨出线外 150px（仍可见）
    check(h.retro == 1, "跨出线即结算，不必等 ID 消失");
    check(h.events.back().reason == "leave-cross-line", "判定依据 = 跨线");
    h.run(10, {});
    check(h.retro == 1, "后续消失不重复计数");
}

// ---------------------------------------------------------------- T11
void t11_no_settle_when_dying_in_entry_band() {
    header("T11 已进入者消失在入口带内(未跨出标记线) → 不计离开类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(41, 550, 600)});
    h.run(3, {makeTrack(41, 660, 600)});
    h.run(6, {makeTrack(41, 900, 600)});
    check(h.in == 1, "进入 → IN=1");
    h.run(4, {makeTrack(41, 660, 600)});   // 走回入口带，但没跨过标记线
    h.run(95, {});                          // 在门口被遮挡吞掉
    check(h.retro == 0 && h.events.size() == 1, "未跨线 → 不计离开类");
}

// ---------------------------------------------------------------- T12
// 现场问题①：201 相机看着人从"对面那一侧"(远端)走出检测区，旧逻辑误报 RETRO。
// 新规则要求"必须跨过本相机的标记线"，此场景应 0 事件。
void t12_far_side_exit_emits_nothing() {
    header("T12 人从对面一侧(远端)走出检测区 → 本相机不得计离开类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(21, 550, 600)});
    h.run(3, {makeTrack(21, 660, 600)});
    h.run(6, {makeTrack(21, 900, 600)});
    check(h.in == 1, "从入口跨线进入 → IN=1");
    h.run(5, {makeTrack(21, 1300, 600)});
    h.run(5, {makeTrack(21, 1500, 600)});   // 多边形外(远端)，但脚底仍在标记线内侧
    h.run(6, {});
    check(h.in == 1 && h.retro == 0 && h.out == 0,
          "从远端离场 → 本相机 0 事件(不再抢记 RETRO)");
}

// ---------------------------------------------------------------- T13
// 现场问题②：人从出口侧逆行进来，201 相机旧逻辑误报 IN。新规则应 0 事件。
void t13_far_side_entry_emits_nothing() {
    header("T13 人从对面一侧(远端)进入检测区 → 本相机不得计进入类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(4, {makeTrack(22, 1500, 600)});   // 远端多边形外，但仍在标记线内侧
    h.run(6, {makeTrack(22, 1200, 600)});
    h.run(6, {makeTrack(22, 900, 600)});
    check(h.in == 0 && h.retro == 0 && h.out == 0,
          "从远端进入 → 本相机 0 事件(不再抢记 IN)");
}

// ---------------------------------------------------------------- T14
// 对称检查：200(出口)相机不得因为"人从入口侧走掉"而计 OUT。
void t14_exit_cam_ignores_far_side_leave() {
    header("T14 200(出口)相机：人从对面一侧离场 → 不得计 OUT");
    Harness h(GATE_ROLE_EXIT);
    h.run(3, {makeTrack(31, 550, 600)});
    h.run(3, {makeTrack(31, 660, 600)});
    h.run(6, {makeTrack(31, 900, 600)});
    check(h.retro == 1 && h.out == 0, "从外面跨线进来 → RETRO=1");
    h.run(5, {makeTrack(31, 1500, 600)});   // 从远端(对面那一侧)离开
    h.run(6, {});
    check(h.out == 0, "从远端离场 → 出口相机不得计 OUT");
}

// ---------------------------------------------------------------- T15
// 用户要求：跨线途中换 ID，新 ID 必须继承旧 ID 的计数上下文，不能漏计。
void t15_handover_across_line() {
    header("T15 跨线途中换 ID(旧 ID 在线外失联) → 继承跨线证据并补上进入类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(51, 520, 600)});    // 标记线外侧(signed=-80) → 留下"从外面来"的证据
    h.run(4, {});                            // 旧 ID 短暂失联(lost=4 ≤ 30)
    h.run(6, {makeTrack(52, 690, 600)});     // 新 ID 在旧位置附近出生(距 170px < 180)
    h.run(6, {makeTrack(52, 850, 600)});     // 继续走进检测区
    check(h.in == 1, "继承旧 ID 的跨线证据 → 恰好 1 次 IN(换 ID 不漏计)");
}

// ---------------------------------------------------------------- T16
// 跨出标记线后确认帧数没凑够即消失在活跃输出 → 消失回补一次离开类。
void t16_loss_fallback_after_partial_crossing() {
    header("T16 跨出标记线但确认帧数没凑够即消失 → 消失回补离开类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(43, 660, 600)});    // 确立为线内侧
    check(h.in == 0 && h.retro == 0, "首次确立侧不发事件");
    h.run(2, {makeTrack(43, 560, 600)});    // 跨到线外(-40px)，仅 2 帧(<3)未确认
    check(h.retro == 0, "确认帧数不足 → 暂不结算");
    h.run(5, {});                            // 消失在活跃输出
    check(h.retro == 1, "消失回补 → 恰好 1 次离开类");
    check(h.events.back().settled, "标记为消失回补");
}

// ---------------------------------------------------------------- T17
// ★ 本次 bug 的核心场景：单人（ByteTrack 全程复用同一 ID）多次往返，
//   旧实现"每人至多 1 进 1 出"会把第 2 趟之后全部漏计。新实现每次跨线都计。
void t17_same_id_multiple_round_trips() {
    header("T17 单人同一 ID 多次往返 → 每次跨线都计数（1:1 封顶已废弃）");
    Harness h(GATE_ROLE_ENTRY);   // 入口相机 201
    h.run(3, {makeTrack(60, 550, 600)});    // 线外侧
    for (int r = 0; r < 3; r++) {
        h.run(3, {makeTrack(60, 660, 600)});   // 跨进 → IN
        h.run(6, {makeTrack(60, 850, 600)});   // 深入检测区
        h.run(3, {makeTrack(60, 550, 600)});   // 跨出 → RETRO
        h.run(3, {makeTrack(60, 550, 600)});   // 线外停留
    }
    check(h.in == 3, "3 次跨进 → IN=3（每次跨线都计，不再只计首次）");
    check(h.retro == 3, "3 次跨出 → RETRO=3");
    check(h.out == 0, "入口相机不产生 OUT");
}

// ---------------------------------------------------------------- T18
// ★ 出口相机(200) 同一 ID 多次往返 → OUT/RETRO 每次都计（用户反馈 OUT 漏计）。
void t18_exit_camera_multiple_round_trips() {
    header("T18 出口相机(200)：单人多次往返 → 每次跨线都计 OUT/RETRO（不再漏计 OUT）");
    Harness h(GATE_ROLE_EXIT);
    h.run(3, {makeTrack(61, 850, 600)});    // 检测区内(线内侧)
    for (int r = 0; r < 3; r++) {
        h.run(3, {makeTrack(61, 550, 600)});   // 跨出 → OUT
        h.run(3, {makeTrack(61, 550, 600)});
        h.run(3, {makeTrack(61, 850, 600)});   // 跨回 → RETRO
        h.run(3, {makeTrack(61, 850, 600)});
    }
    check(h.out == 3, "3 次跨出 → OUT=3（不再漏计）");
    check(h.retro == 3, "3 次跨回 → RETRO=3");
    check(h.in == 0, "出口相机不产生 IN");
}

// ---------------------------------------------------------------- T19
// 人在闸口跨度外、t1 延长线上横向来回走动 → 不得被判成跨线。
void t19_out_of_span_lateral_walk() {
    header("T19 闸口跨度外(t1 延长线上)横向走动 → 不产生事件");
    Harness h(GATE_ROLE_ENTRY);
    for (int i = 0; i < 4; i++) {
        h.run(4, {makeTrack(70, 500, 1400)});   // y=1400 远超 t1 跨度(300..900+50)
        h.run(4, {makeTrack(70, 750, 1400)});   // 横向跨过 x=600，但不在跨度内
    }
    h.run(5, {});
    check(h.in == 0 && h.out == 0 && h.retro == 0,
          "跨度外横向走动 → 0 事件（跨度约束生效）");
}

// ---------------------------------------------------------------- T20
// ★ 原始漏计（现场 P12）：人正跨线那一瞬间彻底跟丢，之后【再没有任何后继 ID】。
//   消失回补只兜"最后观测脚底已越过死区到线外"，交接兑付又需要后继 ID → 两者都覆盖不到。
//   §14 曾用"跟丢越线推断"补这一次。**2026-09-17 用户裁定：这是一种"没有实时观测依据的
//   补计数"，默认关闭**（见文档 §20）→ 本用例现在钉的是【默认行为 = 不补计】。
//   同一轨迹在"显式打开开关"下的行为由 T20b 钉住（防止机制腐化）。
void t20_inference_disabled_by_default() {
    header("T20 跟丢越线推断【默认关闭】→ 跨线瞬间跟丢不再补计（宁可漏计，不虚计）");
    Harness h(GATE_ROLE_ENTRY);   // 201 入口相机：离开类 = RETRO
    check(!h.counter.lostCrossingInferenceEnabled(), "默认：推断开关为关");

    h.run(3, {makeTrack(80, 550, 600)});    // 线外侧，确立所在侧
    h.run(3, {makeTrack(80, 660, 600)});    // 跨进 → IN=1
    check(h.in == 1 && h.retro == 0, "进入那条有真实跨线观测 → 照常计 IN=1");
    h.run(3, {makeTrack(80, 720, 600)});    // 深入检测区
    h.run(1, {makeTrack(80, 680, 600)});    // 转身朝线外走(帧间 -40px)
    h.run(1, {makeTrack(80, 640, 600)});    // 仍在入口带内侧(signed=+40)
    check(h.retro == 0, "尚在线内侧 → 不结算");
    h.run(200, {});                          // 彻底跟丢，无任何后继 ID
    check(h.retro == 0, "★ 从没观测到脚底出线 → 不补计（= 关闭推断的已知代价，即 P12 漏计）");
    check((int)h.events.size() == 1, "全程只有 1 条事件（进入那条），没有任何补发");
}

// ---------------------------------------------------------------- T20b
// 机制保留：显式打开开关后，同一条轨迹仍能补出那次离开类（将来做"推断+实时观测"可复用）。
void t20b_inference_opt_in_still_works() {
    header("T20b 显式打开开关 → 同一轨迹仍补 1 次离开类（机制保留，默认不生效）");
    Harness h(GATE_ROLE_ENTRY);
    h.counter.enableLostCrossingInference(true);
    check(h.counter.lostCrossingInferenceEnabled(), "开关已打开");

    h.run(3, {makeTrack(80, 550, 600)});
    h.run(3, {makeTrack(80, 660, 600)});
    h.run(3, {makeTrack(80, 720, 600)});
    h.run(1, {makeTrack(80, 680, 600)});
    h.run(1, {makeTrack(80, 640, 600)});
    h.run(200, {});
    check(h.retro == 1, "打开后 → 跟丢且无后继 ID 补 1 次离开类");
    check(h.events.back().settled, "推断事件标记为 settled");
    check(h.events.back().reason == "leave-cross-line-inferred",
          "判定依据 = 跟丢越线推断");
    check(h.in == 1, "进入类不受影响(IN 仍为 1)");
}

// ---------------------------------------------------------------- T21
// 反例：人在检测区深处(离标记线上百像素)消亡 → 不得推断。守住 T7：
// 201 不得因为"人在自己这边消失"就抢记 200 的 OUT。
void t21_no_inference_when_dying_deep_inside() {
    header("T21 反例：检测区深处消亡(离线段远) → 即使打开开关也不得推断离开类");
    Harness h(GATE_ROLE_ENTRY);
    h.counter.enableLostCrossingInference(true);   // 故意打开：本用例是"判据"的反例
    h.run(3, {makeTrack(85, 550, 600)});
    h.run(3, {makeTrack(85, 660, 600)});    // IN=1
    check(h.in == 1, "进入 → IN=1");
    h.run(3, {makeTrack(85, 900, 600)});    // 检测区深处(signed≈+300)
    h.run(1, {makeTrack(85, 860, 600)});    // 有朝外速度，但离标记线仍很远
    h.run(1, {makeTrack(85, 820, 600)});    // 帧间 -40px
    h.run(200, {});
    check(h.in == 1 && h.retro == 0,
          "离线段远 → 不推断(否则 201 会抢 200 的 OUT)");
}

// ---------------------------------------------------------------- T22
// 反例：人走到门口后站定、再消失 → 不得推断。守住 T11：
// 虽然"历史平均速度"朝外，但消失前帧间速度已归零(人是站定被吞掉的，没有离开)。
void t22_no_inference_when_stationary_at_door() {
    header("T22 反例：走到门口后站定再消失(帧间速度归零) → 即使打开开关也不得推断");
    Harness h(GATE_ROLE_ENTRY);
    h.counter.enableLostCrossingInference(true);   // 故意打开：本用例是"判据"的反例
    h.run(3, {makeTrack(90, 550, 600)});
    h.run(3, {makeTrack(90, 660, 600)});    // IN=1
    check(h.in == 1, "进入 → IN=1");
    h.run(3, {makeTrack(90, 900, 600)});
    h.run(4, {makeTrack(90, 640, 600)});    // 走回门口(有朝外历史)
    h.run(4, {makeTrack(90, 640, 600)});    // 站定：帧间位移 = 0
    h.run(200, {});
    check(h.in == 1 && h.retro == 0, "站定后消失 → 不推断离开");
}

// ---------------------------------------------------------------- T23
// 出口相机(200) 对称性：离开类 = OUT，同样要能兜"跨线瞬间跟丢且无后继"。
void t23_exit_camera_inferred_out() {
    header("T23 出口相机(200)：默认关闭 → 不补 OUT；显式打开后才补（标签映射对称）");
    const auto runTrace = [](Harness& h) {
        h.run(3, {makeTrack(95, 850, 600)});    // 检测区内，确立所在侧(内侧)
        h.run(1, {makeTrack(95, 700, 600)});
        h.run(1, {makeTrack(95, 660, 600)});    // 朝线外
        h.run(1, {makeTrack(95, 640, 600)});    // 帧间 -20px
        h.run(200, {});
    };

    Harness h_off(GATE_ROLE_EXIT);
    runTrace(h_off);
    check(h_off.out == 0 && h_off.retro == 0 && h_off.in == 0,
          "★ 默认关闭 → 没有任何出线观测，0 事件");

    Harness h_on(GATE_ROLE_EXIT);
    h_on.counter.enableLostCrossingInference(true);
    runTrace(h_on);
    check(h_on.out == 1 && h_on.retro == 0 && h_on.in == 0,
          "打开开关 → 推断补 1 次 OUT（出口相机角色标签对称）");
}

// ---------------------------------------------------------------- T24
// 防重复：跨线瞬间跟丢，但【有后继 ID】在线的另一侧出现 → 由交接兑付计一次，
// 跟丢越线推断不得再补第二次。
void t24_no_double_count_when_successor_exists() {
    header("T24 防重复：跨线瞬间跟丢但有后继 ID → 只计一次（开关打开时也不许双计）");
    Harness h(GATE_ROLE_ENTRY);
    h.counter.enableLostCrossingInference(true);   // 故意打开：验证"交接已计则推断不再补"
    h.run(3, {makeTrack(96, 550, 600)});
    h.run(3, {makeTrack(96, 660, 600)});    // IN=1
    check(h.in == 1, "进入 → IN=1");
    h.run(1, {makeTrack(96, 700, 600)});
    h.run(1, {makeTrack(96, 660, 600)});
    h.run(1, {makeTrack(96, 640, 600)});    // 朝线外(signed=+40)
    h.run(2, {});                            // 跟丢 2 帧(未达推断门限)
    h.run(3, {makeTrack(97, 540, 600)});    // 后继 ID 出现在线外(-60)
    h.run(10, {});
    check(h.retro == 1, "交接已计一次 → 推断不得重复(恰好 1 次)");
    check(h.in == 1 && h.out == 0, "不重不漏");
}

// ---------------------------------------------------------------- T25
// ★ 第 2 趟跨线即换 ID：entered/exited 已被第 1 趟置真。
//   旧实现的兑付把 `!entered` 当门 → 兑付被吞掉，且 line_side 照常提交 →
//   这条真实跨线既不发事件也再补不回来（"每跨线都计数"被破坏）。
void t25_segment_payoff_on_second_round_trip_enter() {
    header("T25 第 2 趟跨进即换 ID(entered 已为 true) → 兑付仍须补计");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(100, 550, 600)});   // 线外侧
    h.run(3, {makeTrack(100, 660, 600)});   // 第 1 趟跨进 → IN=1
    h.run(3, {makeTrack(100, 550, 600)});   // 第 1 趟跨出 → RETRO=1
    check(h.in == 1 && h.retro == 1, "第 1 趟往返正常计数");
    h.run(2, {});                            // 跟丢 2 帧
    h.run(3, {makeTrack(101, 690, 600)});   // 第 2 趟跨进换 ID(连线 550→690 真穿过 t1)
    h.run(6, {makeTrack(101, 850, 600)});   // 继续走进检测区
    check(h.in == 2, "第 2 趟跨进 → IN=2（兑付不再被 entered 守卫吞掉）");
    check(h.retro == 1, "离开类不受影响");
}

// ---------------------------------------------------------------- T26
// ★ 对称：第 2 次「离开」跨线即换 ID（exited 已为 true）→ 兑付仍须补计。
void t26_segment_payoff_on_second_round_trip_leave() {
    header("T26 第 2 次跨出即换 ID(exited 已为 true) → 兑付仍须补计");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(102, 850, 600)});   // 检测区内，确立所在侧(内侧)
    h.run(3, {makeTrack(102, 550, 600)});   // 第 1 次跨出 → RETRO=1
    h.run(3, {makeTrack(102, 850, 600)});   // 跨回 → IN=1
    check(h.retro == 1 && h.in == 1, "第 1 趟往返正常计数");
    h.run(3, {makeTrack(102, 700, 600)});   // 走到入口带内侧(+100px)
    h.run(2, {});                            // 跟丢 2 帧
    h.run(3, {makeTrack(103, 545, 600)});   // 第 2 次跨出换 ID(连线 700→545 穿线, 距155)
    h.run(6, {makeTrack(103, 500, 600)});
    check(h.retro == 2, "第 2 次跨出 → RETRO=2（兑付不再被 exited 守卫吞掉）");
    check(h.in == 1, "进入类不重复计");
}

// ---------------------------------------------------------------- T33
// ★★ 用户方案（message 6）★★：人从检测区往区域外走，**正跨线那一瞬间被吞**（ID 消失），
//   之后 ByteTrack 又把 ID 跟回来、且人已经到线外了 → 允许补计一次离开类(OUT)。
//   为什么它有实时依据：判据是"旧 ID 最后一帧脚底 + 新 ID 首帧脚底"两个**真实观测点**
//   的连线确实穿过判定线（segmentCrossesLine），并伴随所在侧翻转 —— 与 §14 那种
//   "只有运动趋势外推、没有任何观测"的补计数有本质区别（§14 已默认关闭，本条默认开启）。
void t33_exit_camera_handover_eaten_while_crossing_out() {
    header("T33 出口相机：跨出瞬间被吞 → 新 ID 已在线外 → 恰好补 1 次 OUT");
    Harness h(GATE_ROLE_EXIT);
    h.run(5, {makeTrack(61, 900, 600)});   // 检测区内，确立所在侧(signed=+300)
    h.run(3, {makeTrack(61, 700, 600)});   // 朝外走到入口带内侧(+100px)
    h.run(2, {});                          // ★ 正跨线那一瞬间被吞(lost=2 ≤ 30)
    h.run(6, {makeTrack(62, 560, 600)});   // ByteTrack 重新给 ID：在旧位置附近(140px<180)且已在线外(-40)
    const auto evs = h.eventsFor(62);
    check(h.out == 1, "两真实脚底连线穿线 + 侧翻转 → 恰好 1 次 OUT");
    check(!evs.empty() && evs.front().reason == "handover-segment-leave",
          std::string("判定依据是交接跨线兑付(不是速度外推)，实际=") +
              (evs.empty() ? std::string("<无事件>") : evs.front().reason));
    check(h.retro == 0 && h.in == 0, "不得顺手多计进入类");
    h.run(10, {makeTrack(62, 520, 600)});  // 继续走向外面
    check(h.out == 1, "此后外侧观测不得重复计 OUT(不双计)");
}

// ---------------------------------------------------------------- T34
// 守卫仍在：新 ID 出现的位置离旧 ID 末帧脚底太远(>kHandoverRadius=180px) → 不认交接。
// 防止"只要换 ID 后人在外侧就补一次"变成兜底规则（那就等于无依据补计数）。
void t34_no_payoff_when_successor_too_far() {
    header("T34 守卫：后继 ID 离失联位置 >180px → 不认交接、不补计");
    Harness h(GATE_ROLE_EXIT);
    h.run(5, {makeTrack(63, 900, 600)});   // 检测区内
    h.run(3, {makeTrack(63, 700, 600)});   // 走到入口带内侧
    h.run(2, {});                          // 被吞
    h.run(6, {makeTrack(64, 480, 600)});   // 新 ID 已在线外，但距旧脚底 220px > 180
    h.run(10, {});                         // 等它进入"消失"处理
    check(h.out == 0, "距离越界 → 不认交接 → 不补计 OUT");
    check(h.retro == 0 && h.in == 0, "也不得计进入类");
}

// ---------------------------------------------------------------- T35
// 守卫仍在：换 ID 前后**始终在线外**(如线外侧横向走动/等车) → 侧没翻转 → 不计。
// 证明触发源是"真实穿线"，而不是"换了 ID / 人在外侧"这两件事本身。
void t35_no_payoff_when_side_unchanged_across_handover() {
    header("T35 守卫：换 ID 前后始终在线外 → 无穿线 → 不计数");
    Harness h(GATE_ROLE_EXIT);
    h.run(5, {makeTrack(65, 500, 600)});   // 全程线外(signed=-100)
    h.run(2, {});                          // 被吞
    h.run(6, {makeTrack(66, 460, 600)});   // 新 ID 也在线外(距 40px，交接确实发生)
    h.run(10, {});
    check(h.out == 0, "侧未翻转 → 不补计");
    check(h.retro == 0 && h.in == 0, "全程 0 事件");
}

// ---------------------------------------------------------------- T36
// ★ 现场 2026-09-18 现象（P21 连"逆行"也丢了）。
//   **本用例原本钉的是缺陷本身（0 事件）；缺陷 A 修复后反过来钉"正确行为"。**
//   场景：人出生在区内（没被观测到跨进）→ 往外走时只被观测到 1 帧在线外（pending 跨线）
//   → 另一个 ID 在附近出生并继承了它。
//     修复前：旧 ID 被 sealed，它的 pending 证据随之作废；继承者的观测又落在死区 →
//             **整次跨线凭空消失**（现场 P21：最后观测 signed=-129px 却没有任何事件）。
//     修复后：**封存前先把这份证据兑现**（口径与消失回补完全一致，属"三条合法证据"第 ③ 条：
//             最后一次观测脚底已在另一侧 ≥ 死区），恰好补 1 次离开类；且结算后的所在侧
//             继承给新 ID，新 ID 不会把同一次跨线再计一遍。
//   现场对应的日志线索：`track=18 ... inherited_from=21`（21 是在区内出生的那个 ID）。
void t36_seal_settles_pending_crossing_first() {
    header("T36 ★封存前先兑现待确认跨线证据 → 恰好补 1 次离开类（缺陷 A 修复）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(80, 900, 600)});   // 区内出生 → 首次观测只确立 line_side=1（无 IN）
    h.run(1, {makeTrack(80, 540, 600)});   // 只被看到 1 帧在线外(-60px) → 有证据但没凑够 3 帧确认
    h.run(2, {});                          // 80 失联 2 帧（落在可继承窗口 1..30）
    check(h.retro == 0, "封存发生前：这条跨线还没被确认 → 尚无事件");
    h.run(4, {makeTrack(81, 590, 600)});   // 附近出生 → 封存 80 之前先兑现它的证据
    check(h.retro == 1, "★ 封存前兑现 → 恰好补 1 次离开类（修复前这里恒为 0，即漏计）");
    check((int)h.events.size() == 1,
          "全程只有这 1 条事件，没有多余补发");
    check(!h.events.empty() && h.events.back().reason == "leave-cross-line-on-seal",
          "判定依据 = leave-cross-line-on-seal（现场可按 reason 区分于回补/交接兑付）");
    check(!h.events.empty() && h.events.back().track_id == 80,
          "事件归给旧 ID(80)，不是继承者");
    check(!h.events.empty() && h.events.back().settled,
          "标记为 settled（终审补发，与消失回补同口径）");
    h.run(6, {});                          // 继承者也消失
    check(h.retro == 1, "继承者消失后不得再补 1 次（封存结算与消失回补不双计）");
    check(h.in == 0, "全程也不该计进入类");
    check(h.all_notes.find("sealed-with-pending") != std::string::npos,
          "注记里出现 sealed-with-pending（现场用它定位'这次跨线在交接点被处理过'）");
    check(h.all_notes.find("已在封存前兑现") != std::string::npos,
          "注记文本明确写出'已兑现'，现场一眼可判断有没有漏");
}

// ---------------------------------------------------------------- T41
// 守卫①：**"封存前结算"不得退化成"每次交接都补一次"**。
// 旧 ID 的最后一次观测仍在**它自己那一侧**（没有 pending 证据）→ 封存不得发任何事件。
void t41_no_settle_when_old_id_has_no_pending_evidence() {
    header("T41 守卫：旧 ID 最后观测仍在自己的侧（无 pending）→ 封存不得发事件");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(70, 540, 600)});   // 一直在判定线外侧(x=540, signed=-60) → 无任何事件
    h.run(2, {});                          // 失联 2 帧
    h.run(4, {makeTrack(71, 560, 600)});   // 附近出生并继承它（旧侧=外，与新脚底同侧）
    check(h.events.empty(), "无 pending 证据 → 封存不发任何事件（不虚计）");
    check(h.in == 0 && h.retro == 0, "两个计数都保持 0");
}

// ---------------------------------------------------------------- T42
// 守卫②：封存结算过一次之后，继承者**继续朝同方向走远** → 不得再计一次。
// （结算时已把所在侧提交为'线外侧'，继承者继承的就是这个新侧，因此 raw_side 与它相同，
//   不会触发侧翻转边沿。这条守住"修复不会把 1 次跨线变成 2 次"。）
void t42_no_double_count_after_seal_settlement() {
    header("T42 守卫：封存结算后继承者继续走远 → 仍只有 1 次事件（不双计）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(60, 900, 600)});   // 区内出生（无 IN）
    h.run(1, {makeTrack(60, 540, 600)});   // 1 帧在线外 → pending
    h.run(2, {});                          // 失联 2 帧
    h.run(3, {makeTrack(61, 560, 600)});   // 继承者出生 → 封存前结算 → RETRO=1
    check(h.retro == 1, "封存结算恰好 1 次离开类");
    h.run(4, {makeTrack(61, 460, 600)});   // 继续朝线外走远(-140px)
    h.run(4, {makeTrack(61, 360, 600)});   // 更远(-240px)
    check(h.retro == 1, "★ 朝同方向继续走远不再计（同一次跨线只结一次）");
    check(h.in == 0, "也不得计进入类");
}

// ---------------------------------------------------------------- T44
// ★★ 用户 2026-09-18 的核心陈述："21 才是真正的人，他的轨迹都是真人真实的轨迹；
//   因为之前的 ID 继承导致了出现错误。" ★★
//   上面的 ③（复活者不得继承）挡住的是"复活 ID 去封存别人"这一条来路；但封存还有另一条来路：
//   继承者确实是**首次出现的新 track_id**（典型：门口假框 / 瞬时误检新生了一个 ID），
//   它照样会把真人封存掉。此时唯一的救法是 —— **真人之后又有真实观测，就该解封**。
//   `!state.sealed` 是正常翻转、消失回补、交接兑付三条路径的硬门，不解封就等于把这条
//   真实轨迹永久静默（现场：被 18 封存后的 21 再无任何事件）。
//   本用例钉住：连续 kUnsealConfirmFrames(5) 帧真实观测 → 解封 → 此后真实跨线正常计数。
void t44_sealed_id_unseals_on_real_observations() {
    header("T44 ★解封：被封存的 ID 之后又收到真实观测 → 解封并恢复计数资格");
    Harness h(GATE_ROLE_ENTRY);
    // ① 80 在区内出生（无 IN）→ 只被看到 1 帧在线外(仅 pending) → 失联 → 被 81 继承封存
    h.run(5, {makeTrack(80, 900, 600)});
    h.run(1, {makeTrack(80, 540, 600)});
    h.run(2, {});
    // ★ M4 之后"封存"需要继承者自己走出轨迹才生效 → 这里让 81 动起来，制造一个**真的**封存
    h.run(3, {makeTrack(81, 560, 600)});
    check(h.retro == 1, "前置：交接点结算恰好 1 次离开类");
    h.run(3, {makeTrack(81, 460, 600)});   // 81 走出 100px 轨迹 → seal-confirmed
    check(h.all_notes.find("seal-confirmed") != std::string::npos,
          "81 自己走出了轨迹 → 确认是可动的真实目标 → 正式封存 80");
    check(h.retro == 1, "81 走动方向与继承到的侧一致 → 不额外产生事件");
    h.run(6, {});                          // 继承者 81 也消失，避免它干扰后续断言
    check(h.retro == 1, "继承者消失后不再补（账已结清，不双计）");
    // ② 真人 80 又回来了 —— 连续真实观测（预测框不算）→ 解封
    h.run(5, {makeTrack(80, 540, 600)});
    check(h.all_notes.find("unsealed") != std::string::npos,
          "★ 连续 5 帧真实观测 → 打 unsealed 注记（现场可据此确认'被封存的其实是真人'）");
    // ③ 解封只恢复"资格"：必须重新确立所在侧，不得拿封存前的旧证据补事件
    h.run(4, {makeTrack(80, 540, 600)});
    check(h.in == 0, "解封后先重新确立所在侧（线外）→ 不补发任何历史事件");
    // ④ 真人继续走 → 真的跨进检测区 → 必须计数（修复前这里恒为 0）
    h.run(4, {makeTrack(80, 900, 600)});
    check(h.in == 1, "★ 解封后真人再跨线 → 正常计 1 次进入类（修复前永久静默）");
    check(h.retro == 1, "离开类仍只有封存结算那 1 次，没有多补");
}

// ---------------------------------------------------------------- T45
// 守卫①：解封**不得退化成"补发历史"**。80 解封后直接出现在线内深处（历史上它就是
// 从线内走到线外的，那一次早已作为 on-seal 结清）→ 不得因此再补一次 IN。
void t45_unseal_does_not_backfill_events() {
    header("T45 守卫：解封只恢复资格，绝不拿封存前的旧证据补发事件");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(80, 900, 600)});
    h.run(1, {makeTrack(80, 540, 600)});
    h.run(2, {});
    h.run(3, {makeTrack(81, 560, 600)});
    check(h.retro == 1, "前置：交接点结算 1 次");
    h.run(3, {makeTrack(81, 460, 600)});   // 81 走出轨迹 → 确认封存 80
    h.run(6, {});
    // 回来时直接落在判定线内侧 300px 处 → 这正是"当年的来路"，但不是新的观测
    h.run(8, {makeTrack(80, 900, 600)});
    check(h.in == 0, "★ 解封后直接出现在线内 → 不得补 IN（首次观测只确立所在侧）");
    check(h.retro == 1, "总离开类仍为 1");
    check((int)h.events.size() == 1, "全程只有 1 条事件，没有任何补发");
}

// ---------------------------------------------------------------- T46
// 守卫②：解封门必须**比计数门更严**。只给 3~4 帧观测不得解封（否则"被继承封存"会变成
// 一条可被瞬时噪声随意推翻的软门，容易走向多计）。
void t46_unseal_needs_full_confirmation_window() {
    header("T46 守卫：观测不足 5 帧不得解封（解封门比跨线确认窗 3 帧更严）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(80, 900, 600)});
    h.run(1, {makeTrack(80, 540, 600)});
    h.run(2, {});
    h.run(3, {makeTrack(81, 560, 600)});
    h.run(3, {makeTrack(81, 460, 600)});   // 先造成一个**真的**封存
    h.run(6, {});
    h.run(4, {makeTrack(80, 540, 600)});   // 只 4 帧 < kUnsealConfirmFrames(5)
    check(h.all_notes.find("unsealed") == std::string::npos,
          "★ 只 4 帧观测 → 不解封（避免封存被瞬时噪声推翻）");
    check(h.in == 0 && h.retro == 1, "计数不受影响");
}

// ---------------------------------------------------------------- T47
// ★★ M4 核心用例：现场 2026-09-18 的真实场景 ★★
//   检测区内有一件**静止物品**被误识别成人（现场是门口的饮水机）。它产生的新 ID 会出现在
//   真人刚失联的位置附近 —— 但它**永远不动**（框只有几像素抖动），走的痕迹 ≈ 0。
//   用户的要求："用轨迹的方式来做 —— 人是有轨迹的，误识别的物品是没有轨迹的"
//                "需要根据**当前 ID 自己的轨迹**来判断"
//   ⇒ 本用例钉住：**静止 ID 无权封存别人的真实轨迹**（它会拿到 seal-cancelled），
//      而真人 21 因为从未被封存，之后继续走的那一段**照常计数**（修复前这一段全丢）。
//   ⚠️ 注意用户同时提醒的"人在区域内可能会站着不动"：本用例里的真人 21 **全程没有被要求动**，
//      它只是"不再被观测到"；而它回来后就地站着（line_signed 不变）也不会被误杀 ——
//      因为"谁能封存谁"这件事只由**发起方自己**的轨迹决定，与人是否站着完全无关。
void t47_static_id_cannot_seal_real_track() {
    header("T47 ★静止误检(饮水机)新生 ID 无权封存真人轨迹，且真人后续轨迹照常计数");
    Harness h(GATE_ROLE_ENTRY);
    // 真人 21：在区内出生（没被观测到跨进）→ 往外走，只被看到 1 帧在线外(仅 pending)
    h.run(5, {makeTrack(21, 900, 600)});
    h.run(1, {makeTrack(21, 540, 600)});
    h.run(2, {});                          // 21 失联 2 帧（落在可继承窗口 1..30）
    // 门口的静止误检框以"新 ID 18"的姿态出现在附近，**并且一直待在原地**
    h.run(8, {makeTrack(18, 560, 600)});
    check(h.retro == 1, "交接点仍按'最后一次观测已跨界'结算 1 次离开类（这一段没有丢）");
    check(h.all_notes.find("seal-cancelled") != std::string::npos,
          "★ 静止 ID 被识别出来 → seal-cancelled（放弃封存）");
    check(h.all_notes.find("seal-confirmed") == std::string::npos,
          "绝不能 seal-confirmed —— 它没有轨迹，不配终结任何人的轨迹");
    h.run(6, {});                          // 假框 18 自己消失
    // 真人 21 的轨迹完好：它再走一次跨线必须照常计数（修复前它会被静默，这一段全丢）
    h.run(4, {makeTrack(21, 540, 600)});   // 回到线外并重新确立所在侧
    h.run(4, {makeTrack(21, 900, 600)});   // 再跨进来
    check(h.in == 1, "★ 真人 21 未被静止 ID 影响：再跨线正常计 1 次进入类");
}

// ---------------------------------------------------------------- T48
// 正面对照：真人在门口换 ID（新 ID 会走路）→ 必须照常确认交接。
// 保证 M4 只拦"静止者"，不会把"正常的 ID 切换"一起挡掉。
void t48_moving_successor_confirms_seal() {
    header("T48 正面对照：继承者自己能走出轨迹 → seal-confirmed（正常换 ID 不受影响）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(30, 900, 600)});   // 30 在区内出生
    h.run(1, {makeTrack(30, 540, 600)});   // 走到线外 1 帧 → pending
    h.run(2, {});                          // 失联 2 帧
    h.run(3, {makeTrack(31, 560, 600)});   // 31 在附近出生并继承 30
    h.run(3, {makeTrack(31, 640, 600)});   // 31 继续走 → 累积位移 0→100→…（远超 30px 门限）
    check(h.all_notes.find("seal-confirmed") != std::string::npos,
          "★ 继承者走出轨迹 → 确认封存旧 ID（正常换 ID 的交接语义完好）");
    check(h.in == 1, "31 确实跨进检测区 → 恰好 1 次进入类");
    check(h.retro == 1, "30 在交接点已结算的那 1 次离开类不重复计");
}

// ---------------------------------------------------------------- T37
// 对照组：同一条轨迹，只要**没有**别的 ID 来继承封存它 → 消失回补能兜住（应当补 1 次）。
// 二者成对，证明"丢"不是消失回补本身不工作，而是被继承封存吃掉了。
void t37_same_track_without_inheritor_gets_on_loss() {
    header("T37 对照组：无继承者时，同样的轨迹由消失回补补上 1 次离开类");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(90, 900, 600)});   // 区内出生
    h.run(1, {makeTrack(90, 540, 600)});   // 只 1 帧在线外(-60px)
    h.run(6, {});                          // 直接消失（没有新 ID 来继承）
    check(h.retro == 1, "消失回补恰好补 1 次离开类");
    check(h.in == 0, "不得凭空计 IN");
}

// ---------------------------------------------------------------- T43
// ★★ 用户 2026-09-18 指出的核心问题 ★★
//   "新 18 找到附近刚失联的 21 → 继承并封存它"这件事本身就是 bug：**21 才是真人、它的轨迹
//   每一步都是真实检测**；18 是被 tracker 重连复活的旧轨迹（ID 数值还比 21 小，说明它出生更早，
//   不可能是"新人"）。让复活者去继承，代价是**把一条真实轨迹封存掉** —— 它的待确认证据、
//   以及此后每一次真实跨线全部作废。
//   本用例钉住新增的门：**只有"计数器第一次见到"的 track_id 才有资格做交接**。
void t43_revived_id_cannot_inherit_and_seal() {
    header("T43 ★守卫：被重连复活的旧 ID 不得继承并封存真实轨迹");
    Harness h(GATE_ROLE_ENTRY);
    // ① 先让计数器见过 70（它随后长期消失 → 状态被回收，但"见过"的短期记忆还在）
    h.run(3, {makeTrack(70, 540, 600)});   // 线外侧 → 确立 side=0
    h.run(3, {makeTrack(70, 900, 600)});   // 进区 → IN=1
    check(h.in == 1, "前置：70 走进门 → IN=1");
    // ② 70 长期消失（> kStateKeepFrames=240）→ 它的状态被回收
    h.run(300, {});
    // ③ 真正的人 71：区内出生（没被观测到跨进）→ 往外走只被看到 1 帧在线外
    h.run(5, {makeTrack(71, 900, 600)});
    h.run(1, {makeTrack(71, 540, 600)});
    h.run(2, {});                          // 71 失联 2 帧 → 落进可继承窗口(1..30)
    check(h.retro == 0, "此时 71 的跨线尚未结算");
    // ④ 70 被 tracker 重连复活，恰好出现在 71 附近（修复前它会继承并封存 71）
    h.run(4, {makeTrack(70, 560, 600)});
    check(h.retro == 1, "★ 复活者不参与交接 → 71 的跨线由消失回补正常补 1 次");
    check(h.all_notes.find("revived-id") != std::string::npos,
          "注记出现 revived-id（现场可据此确认'是复活轨迹在捣乱'）");
    check(h.all_notes.find("sealed-with-pending") == std::string::npos,
          "不再出现 sealed-with-pending（没有真实轨迹被复活者封存）");
    check(h.in == 1, "复活者不得凭空计进入类（IN 仍只有 70 当初那 1 次）");
}

// ---------------------------------------------------------------- T38
// ★★ 回答"视频里明明看得到完整轨迹，为什么没计数" ★★
// 计数模块只吃 observed=true 的**真实检测**帧；画面上画的框包含 tracker 的**预测框**
// （main_tracking 的 OSD 画全部输出轨迹，而 gate_id_counter 在 `if(!track.observed) continue;`
//  处直接跳过预测框）。所以"看得到轨迹"与"计数模块看到跨线"是两件事。
// 本组用例把三种"看得到却没计数"的情形各钉一条注记（GateIdCounter::notes()）。
void t38_note_first_side_when_born_far_inside() {
    header("T38 出生在区内：首次确立所在侧时就已离线很远 → 打 first-side 注记（不会计 IN）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(4, {makeTrack(210, 900, 600)});   // 出生即在判定线内侧 300px
    check(h.in == 0, "出生在区内不凭空计 IN");
    check(h.all_notes.find("first-side") != std::string::npos,
          "注记里出现 first-side（现场用来判断'这个人跨线那几帧没被看到'）");
}

void t39_note_abandoned_crossing_candidate() {
    header("T39 跨线候选只观测到 2 帧就被反向覆盖 → cand-abandoned 注记（差一点就计数）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(5, {makeTrack(211, 900, 600)});   // 区内，确立 side=in
    h.run(2, {makeTrack(211, 570, 600)});   // 只观测 2 帧在线外(-30px)，不够 3 帧确认
    h.run(4, {makeTrack(211, 900, 600)});   // 又回到区内
    check(h.retro == 0, "只 2 帧证据 → 不计（宁可漏计不虚计）");
    check(h.all_notes.find("cand-abandoned") != std::string::npos,
          "注记里出现 cand-abandoned（含 seen=N 帧，直接说明差几帧）");
}

void t40_note_predicted_frames_are_invisible_to_counter() {
    header("T40 只有预测框的帧(observed=false) → predicted-frame 注记（画面有框但计数看不到）");
    Harness h(GATE_ROLE_ENTRY);
    h.run(3, {makeTrack(212, 595, 600)});                    // 真实观测，落在判定线附近
    h.run(1, {makeTrack(212, 595, 600, /*observed=*/false)}); // 预测框（OSD 上照样画框）
    check(h.all_notes.find("predicted-frame") != std::string::npos,
          "注记里出现 predicted-frame（这就是'视频里有轨迹、日志里没事件'的根因标记）");
}

// ---------------------------------------------------------------- T27
void t27_polyline_judging_line_cross() {
    header("T27 判定线为折线：跨过带折角的判定线 → 恰好 1 次进入类/1 次离开类");
    BentHarness h(GATE_ROLE_ENTRY);
    check(h.counter.judgingLine().size() == 3, "判定线解析为 3 顶点折线");
    check(h.counter.zoneRing().size() == 6, "区域环 = 折线 ∪ 其 x 平移副本(2N 顶点)");
    // y 恒定（帧间 dy=0）→ 不会触发"遮挡脚底冻结"；两侧 signed 都远超死区
    h.run(5, {makeTrack(1, 650, 650)});    // 折线外侧 signed≈-100
    check(h.in == 0, "首次观测只确立所在侧，不发事件");
    h.run(6, {makeTrack(1, 900, 650)});    // 折线内侧 signed≈+135
    check(h.in == 1, "跨过折线 → 恰好 1 次 IN");
    h.run(6, {makeTrack(1, 650, 650)});    // 跨回去
    check(h.retro == 1, "跨回去 → 恰好 1 次离开类");
    h.run(5, {});
    check(h.in == 1 && h.retro == 1, "ID 消失不重复计数");
}

// ---------------------------------------------------------------- T28
void t28_polyline_out_of_span_no_count() {
    header("T28 判定线为折线：跨度之外扫动不计数，跨度之内同样动作计数");
    BentHarness h(GATE_ROLE_ENTRY);
    // 同样的纵向扫动做两组对比：一组在折线跨度外(右远处)，一组落在跨度内。
    // 逐帧只走 50px（< 遮挡脚底冻结门限），避免把"冻结"误当成测试结论。
    h.run(5, {makeTrack(1, 1800, 900)});
    for (int y = 900; y >= 700; y -= 50) h.run(3, {makeTrack(1, 1800, (float)y)});
    check(h.in == 0 && h.retro == 0,
          "脚底投影超出折线两端(含 ±50px 余量) → 不参与跨线判定，0 事件");
    h.run(5, {makeTrack(1, 820, 900)});
    for (int y = 900; y >= 700; y -= 50) h.run(3, {makeTrack(1, 820, (float)y)});
    check(h.in == 1, "同样的纵向动作落在折线跨度内 → 恰好 1 次 IN（对照，非空测试）");
}

// ---------------------------------------------------------------- T29
void t29_polyline_cross_at_bend_counts_once() {
    header("T29 判定线为折线：跨线时命中线段切换(折角处) → 仍只计一次");
    BentHarness h(GATE_ROLE_ENTRY);
    // (560,550) 归第 1 段判定(signed≈-148)，(820,550) 归第 2 段判定(signed≈+95)：
    // 一次跨越里"最近线段"发生了变化，这是折线特有的情形，必须只计一次。
    h.run(5, {makeTrack(7, 560, 550)});
    h.run(6, {makeTrack(7, 820, 550)});
    check(h.in == 1, "命中线段从第 1 段切到第 2 段 → 恰好 1 次 IN（不因换段双计）");
    h.run(6, {makeTrack(7, 560, 550)});
    check(h.retro == 1, "跨回 → 恰好 1 次离开类");
    check(h.in == 1, "进入类不重复计");
}

// ---------------------------------------------------------------- T30
void t30_polyline_exit_role_labels() {
    header("T30 折线判定线 + 出口相机(200)：标签映射对称");
    BentHarness h(GATE_ROLE_EXIT);
    h.run(5, {makeTrack(9, 650, 650)});
    h.run(6, {makeTrack(9, 900, 650)});    // 跨进折线内侧
    check(h.retro == 1, "出口相机跨进线内侧 → RETRO=1");
    h.run(6, {makeTrack(9, 650, 650)});    // 跨出
    check(h.out == 1, "出口相机跨出 → OUT=1");
    check(h.in == 0, "出口相机不产生 IN");
}

// ---------------------------------------------------------------- T31
// ★ 现场假 OUT（2026-09-17，200 出口相机）：人从区域外进区、在区内转圈，走到判定线
//   **末端附近**但并没有跨出去，随后跟丢 → 旧逻辑补发了一次 OUT（OUT:6→7）。
//   复算现场几何：最后脚底 line_signed≈+63px、帧间速度≈-12px/帧，外推需要 ~6.9 帧
//   才越过死区；旧判据允许窗口 20 帧，于是第 7 个丢失帧凑够就误判。
//   "跨线在望"(need ≤ 5 帧)生效后必须拒绝 —— 他只是走到线边、并没有跨。
void t31_no_inference_when_just_near_line_not_crossing() {
    header("T31 反例（现场假 OUT 复现）：走到判定线附近但没跨出去就消失 → 默认与打开开关都不得计数");
    const auto runTrace = [](Harness& h) {
        h.run(4, {makeTrack(110, 900, 600)});   // 出生在检测区深处 → 只确立"内侧"，不发事件
        h.run(2, {makeTrack(110, 800, 600)});   // 朝线外走
        h.run(2, {makeTrack(110, 720, 600)});
        h.run(2, {makeTrack(110, 680, 600)});   // signed=+80
        h.run(1, {makeTrack(110, 675, 600)});   // signed=+75
        h.run(1, {makeTrack(110, 663, 600)});   // signed=+63，帧间 -12px/帧（≈现场数值）
        h.run(200, {});                          // 彻底跟丢（现场在第 7 个丢失帧误判）
    };

    Harness h_off(GATE_ROLE_EXIT);
    runTrace(h_off);
    check(h_off.out == 0, "★ 默认关闭：走到线上方 63px 后跟丢 → 不计数");
    check(h_off.events.empty(), "全程 0 事件（人没有跨过判定线）");

    Harness h_on(GATE_ROLE_EXIT);
    h_on.counter.enableLostCrossingInference(true);
    runTrace(h_on);
    check(h_on.out == 0, "★ 开关打开时也拒绝：需要 ~6.9 帧才越线 > 5 → 不在望");
    check(h_on.events.empty(), "打开开关后仍是 0 事件（④ 跨线在望判据生效）");
}

// ---------------------------------------------------------------- T32
// T31 的对照组：证明"跨线在望"没有把推断废掉。
// 慢速但确实马上要跨过去（signed=+18、帧间 -8px/帧 → need≈4.8 帧 ≤ 5）→ 必须补 OUT。
void t32_inference_still_works_when_crossing_imminent() {
    header("T32 T31 对照组：慢速但马上要跨过去(need≈4.8 帧) → 关闭时不计，打开后必须计");
    const auto runTrace = [](Harness& h) {
        h.run(4, {makeTrack(120, 900, 600)});
        h.run(2, {makeTrack(120, 700, 600)});
        h.run(2, {makeTrack(120, 640, 600)});
        h.run(1, {makeTrack(120, 626, 600)});   // signed=+26
        h.run(1, {makeTrack(120, 618, 600)});   // signed=+18，帧间 -8px/帧
        h.run(200, {});
    };

    Harness h_off(GATE_ROLE_EXIT);
    runTrace(h_off);
    check(h_off.out == 0, "★ 默认关闭 → 仍然 0 事件（只认观测）");

    Harness h_on(GATE_ROLE_EXIT);
    h_on.counter.enableLostCrossingInference(true);
    runTrace(h_on);
    check(h_on.out == 1, "打开开关 → 跨线在望(need 4.8 帧) 补 1 次 OUT");
    check(h_on.events.back().reason == "leave-cross-line-inferred",
          "判定依据 = 跟丢越线推断");
    check(h_on.events.back().settled, "推断事件标记为 settled");
    check(h_on.retro == 0 && h_on.in == 0, "不产生其它类型事件");
}

// ---------------------------------------------------------------- T49
// ★在线上蹭/徘徊：跨出去"没站住"就回来 → 这一对事件整体作废。
// 现场原始日志（201 入口相机，2026-09-18 单人测试）：
//   ts=2048853 事件=RETRO；ts=2049605 事件=IN（IN 3→4）—— 两次只隔 752ms、
//   脚底只差 126px。即人"在入口区域徘徊了一点点"，却被算了一次进入。
// 机制：翻转候选即使已"确认"(≥kLineSideConfirmFrames)，也要在新的一侧
//   连续站住 ≥ cross_dwell_frames 帧才结算；中途回到原侧 = 整对事件作废。
void t49_flip_abandoned_when_not_standing_on_new_side() {
    header("T49 ★在线上蹭/徘徊（跨过去没站住就回来）→ 不计数（dwell=25）");
    Harness h(GATE_ROLE_ENTRY, 25);
    h.run(6,  {makeTrack(1, 800, 600)});   // 出生就在检测区内 → 首次确立 side=in，不发事件
    check(h.in == 0 && h.retro == 0, "出生在区内 → 0 事件（首次观测只确立所在侧）");

    h.run(10, {makeTrack(1, 560, 600)});   // 跨到线外，只站住 10 帧（< 25）
    h.run(10, {makeTrack(1, 820, 600)});   // 又回到线内（= 现场"徘徊了一点点"）
    check(h.in == 0 && h.retro == 0 && h.out == 0,
          "跨出去只待 10 帧又回来 → 离开+进入成对作废，0 事件");
    check(h.all_notes.find("flip-abandoned") != std::string::npos,
          "注记出现 flip-abandoned（判成\"没站住\"，不是\"没看到\"）");
    check(h.all_notes.find("已确认但只站住") != std::string::npos,
          "注记里带实际站住帧数与阈值 → 现场一眼看出离阈值差多少");
}

// ---------------------------------------------------------------- T50
void t50_real_crossing_still_counts_with_dwell() {
    header("T50 dwell=25 下真跨线仍照常计数（进出各 1 次）");
    Harness h(GATE_ROLE_ENTRY, 25);
    h.run(6,  {makeTrack(1, 550, 600)});   // 线外侧 → 首次确立 side=out
    h.run(30, {makeTrack(1, 800, 600)});   // 跨进线内并站住 30 帧（≥ 25）
    check(h.in == 1, "跨进线内并站住 → 恰好 1 次 IN");
    check(h.events.size() == 1 && h.events.back().reason == "enter-cross-line",
          "依据 = 正常跨线（不是消失回补）");
    check(h.retro == 0 && h.out == 0, "不产生其它类型事件");
    h.run(30, {makeTrack(1, 500, 600)});   // 跨出线外并站住 30 帧
    check(h.retro == 1, "跨出线外并站住 → 1 次离开类（201 = RETRO）");
    check(h.in == 1, "IN 未被重复计");
}

// ---------------------------------------------------------------- T51
void t51_dwell_threshold_boundary() {
    header("T51 dwell 阈值边界：新侧只站住 24 帧 → 不计；站住 26 帧 → 计（钉死语义）");
    {
        Harness h(GATE_ROLE_ENTRY, 25);
        h.run(6,  {makeTrack(1, 550, 600)});
        h.run(24, {makeTrack(1, 800, 600)});
        check(h.in == 0, "站住 24 帧(< 25) → 不计（宁漏勿虚）");
    }
    {
        Harness h(GATE_ROLE_ENTRY, 25);
        h.run(6,  {makeTrack(1, 550, 600)});
        h.run(26, {makeTrack(1, 800, 600)});
        check(h.in == 1, "站住 26 帧(≥ 25) → 计 1 次");
    }
}

// ---------------------------------------------------------------- T52
// 鉴别力对照：把门槛退回历史值(kLineSideConfirmFrames=3) → 同一段徘徊确实会误计。
// 这条把"现场 IN 3→4 的成因"永久固化下来，同时证明 T49 的通过是门槛在起作用，
// 而不是因为输入根本没构成跨线候选。
void t52_dwell_off_reproduces_field_false_count() {
    header("T52 对照：dwell=3(历史行为) 下同一段徘徊会误计 → 证明门槛真在起作用");
    Harness h(GATE_ROLE_ENTRY, 3);
    h.run(6,  {makeTrack(1, 800, 600)});
    h.run(10, {makeTrack(1, 560, 600)});
    h.run(10, {makeTrack(1, 820, 600)});
    check(h.retro == 1 && h.in == 1,
          "dwell=3 → 同一段徘徊被计成 离开 1 次 + 进入 1 次（= 现场 IN 3→4）");
    check(h.out == 0, "只有离开/进入两类，没有第三种事件");
}

}  // namespace

int main() {
    std::printf("GateIdCounter 场景测试\n");
    t1_entry_normal_exit();
    t2_retro_exit_via_entry_band();
    t3_occlusion_handover();
    t4_born_inside_no_inherit_no_phantom_count();
    t5_boundary_jitter_keeps_inside_mark();
    t6_loiter_without_crossing();
    t7_no_settle_when_dying_inside();
    t8_four_people_with_id_switch();
    t9_exit_camera_role_labels();
    t10_cross_out_immediate_settlement();
    t11_no_settle_when_dying_in_entry_band();
    t12_far_side_exit_emits_nothing();
    t13_far_side_entry_emits_nothing();
    t14_exit_cam_ignores_far_side_leave();
    t15_handover_across_line();
    t16_loss_fallback_after_partial_crossing();
    t17_same_id_multiple_round_trips();
    t18_exit_camera_multiple_round_trips();
    t19_out_of_span_lateral_walk();
    t20_inference_disabled_by_default();
    t20b_inference_opt_in_still_works();
    t21_no_inference_when_dying_deep_inside();
    t22_no_inference_when_stationary_at_door();
    t23_exit_camera_inferred_out();
    t24_no_double_count_when_successor_exists();
    t25_segment_payoff_on_second_round_trip_enter();
    t26_segment_payoff_on_second_round_trip_leave();
    t33_exit_camera_handover_eaten_while_crossing_out();
    t34_no_payoff_when_successor_too_far();
    t35_no_payoff_when_side_unchanged_across_handover();
    t36_seal_settles_pending_crossing_first();
    t37_same_track_without_inheritor_gets_on_loss();
    t41_no_settle_when_old_id_has_no_pending_evidence();
    t42_no_double_count_after_seal_settlement();
    t43_revived_id_cannot_inherit_and_seal();
    t44_sealed_id_unseals_on_real_observations();
    t45_unseal_does_not_backfill_events();
    t46_unseal_needs_full_confirmation_window();
    t47_static_id_cannot_seal_real_track();
    t48_moving_successor_confirms_seal();
    t38_note_first_side_when_born_far_inside();
    t39_note_abandoned_crossing_candidate();
    t40_note_predicted_frames_are_invisible_to_counter();
    t27_polyline_judging_line_cross();
    t28_polyline_out_of_span_no_count();
    t29_polyline_cross_at_bend_counts_once();
    t30_polyline_exit_role_labels();
    t31_no_inference_when_just_near_line_not_crossing();
    t32_inference_still_works_when_crossing_imminent();
    t49_flip_abandoned_when_not_standing_on_new_side();
    t50_real_crossing_still_counts_with_dwell();
    t51_dwell_threshold_boundary();
    t52_dwell_off_reproduces_field_false_count();

    std::printf("\n==== 结果: %d 项检查, %d 项失败 ====\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

/**
 * gatekeeper_V4 - 主程序
 * YOLOv8检测 -> ByteTrack决策级融合跟踪 -> 区域人数统计 + 虚拟跨线IN/RETRO逆行检测(默认开)
 * 双海康摄像头 RTSP 实时流
 *
 * 用法: ./main_tracking <model.rknn> <config.txt> [npu_threads=2]
 */
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <sqlite3.h>
#include <X11/Xlib.h>

#include "HCNetSDK.h"
#include "LinuxPlayM4.h"

#include "types.h"
#include "yolov8_custom.h"
#include "yolov8_thread_pool.h"
#include "bytetrack/tracker.h"
#include "gate_id_counter.h"
#include "item_abandonment.h"
#include "mask_utils.h"
#include "comm.h"

// ==================== 全局 ====================
std::mutex g_gui_mutex;
std::atomic<bool> g_running{true};
std::mutex g_hik_mutex;
std::string g_model_path;
int g_npu_threads_per_camera = 2;
bool g_enable_display = true;
bool g_enable_zone_trace = false;
constexpr int kDisplayRefreshMs = 50;
constexpr int kDisplayEveryResults = 2;
const int MAX_CAMERAS = 4;

// ★ 跨线"站住"确认帧数（board 取值，2026-09-18）。
// 一次确认的侧翻转，只有新的一侧被连续观测 ≥ 该帧数才结算（发事件 + 推进所在侧）；
// 中途回到原侧 = 在线上蹭/徘徊 → 本次跨线整体作废（不发事件），进出计数保持配平。
// 现场依据：2026-09-18 单人测试日志里 201 出现 RETRO 之后 752ms 紧跟一个 IN
//  （IN 3→4），两次事件脚底只差 126px —— 典型的"跨过去又马上回来"。
// 25 帧 ≈ 1.0~1.4s（板端实测 17~25 FPS）。真人过闸后在新侧会停留数秒，不受影响；
// 万一真人快速穿过而站不够帧数，仍有「消失回补」「交接兑付」两条独立证据兜底。
// 调大 = 更不容易把徘徊算成跨线，但更快穿过的人更容易漏计；调小反之。
constexpr int kCrossDwellFrames = 25;

// comm.cpp 依赖的全局变量
std::atomic<uint16_t> g_max_box_count(0);
std::atomic<bool> g_need_clear_queues{false};

// 双向闸机 + 每摄像头独立计数：201 计 IN/RETRO，200 计 RETRO/OUT。
// 计数与 ID 生命周期解耦：同一 ID 每跨一次判定线就计一次（跨线边沿触发），
// 不再有"每个 ID 至多 1 进 1 出"的封顶（旧 reportFusedEvent / g_fused_* 已移除）。
// ⚠️ 跨线必须"站住" kCrossDwellFrames 帧才算（见上面的常量），防在线上蹭被误计。

// ==================== 摄像头配置 ====================
struct CameraConfig {
    std::string unique_id, ip, username, password;
    int channel;
    int cam_index = 0;  // 在cameras数组中的索引，用于融合去重
    LONG userID = -1, realPlayHandle = -1;
    std::unique_ptr<Yolov8ThreadPool> yolov8_pool;
    std::unique_ptr<ByteTrackTracker> tracker;
    std::atomic<int> frame_id{0};
    std::atomic<uint64_t> decoded_frame_seq{0};
    std::atomic<int64_t> last_capture_ms{0};
    std::atomic<bool> stop_flag{false};
    sqlite3* send_db = nullptr;
    time_t last_minute = 0;
    
    int g_nPort = -1;
    cv::Mat g_BGRImage;
    std::mutex g_frame_mutex;
    
    std::atomic<int> frame_counter{0};
    std::chrono::steady_clock::time_point last_stat_time;
    double fps{0};
    
    cv::Mat detection_mask;
    cv::Mat mask_overlay;
    
    // 标记线 / 判定线（区域边缘折线，点数不限；区域 = 折线 ∪ 其沿 x 平移的副本）
    Tripwire entry_line;             // 旧格式 t1（2 点直线），作为区域边缘折线的一种特例
    std::vector<cv::Point2f> zone_line;  // 新格式 zin,/zout, 的判定线折线；非空时优先
    float entry_zone_depth = 100.0f; // 区域沿 x 平移量(px, 带符号)：朝检测区侧；<1 = 无区域
    int zone_role = -1;              // 配置文件里是否显式指定了角色(-1=未指定，按 IP 判定)
    int role = 0;                    // 0=入口摄像头(201) 1=出口摄像头(200)
    std::atomic<int> in_count{0};    // 进入累计
    std::atomic<int> out_count{0};   // 离开累计
    std::atomic<int> retro_count{0}; // 逆行累计
    
    std::deque<int> recent_counts;
    std::mutex count_mutex;
    int filtered_count{0};
    std::atomic<int> zone_count{0};
    std::atomic<int> mode_count{0};
    std::atomic<int> total_count{0};

    // 物品滞留检测（纯 CV 背景差分，不识别物品类别；与计数逻辑完全解耦）。
    // 见 include/item_abandonment.h / src/item_abandonment.cpp。
    std::unique_ptr<ItemAbandonmentDetector> item_detector;
    std::atomic<int> item_alarm{0};   // 处于告警态(L1+L2)的滞留物数
    std::atomic<int> item_l1{0};
    std::atomic<int> item_l2{0};

    std::shared_ptr<cv::Mat> display_frame_ptr;
    std::mutex display_ptr_mutex;
    int display_w = 640, display_h = 360;

    CameraConfig() = default;
    CameraConfig(const CameraConfig&) = delete;
    CameraConfig& operator=(const CameraConfig&) = delete;
    
    // ⚠️ 手写 move 构造：**新增字段必须在这里同步转发**，否则 loadCameras 里赋的值会被
    // 静默丢掉（vector 扩容 / emplace_back 都会走这里）。
    // 2026-09-16 实际踩过：zone_line / zone_role 漏转发 + loadCameras 漏赋值 →
    // 板端"画面不画区域 + IN/OUT/RETRO 全程 0"（Region/Zone 显示仍正常，极具迷惑性）。
    CameraConfig(CameraConfig&& o) noexcept
        : unique_id(std::move(o.unique_id)), ip(std::move(o.ip)),
          username(std::move(o.username)), password(std::move(o.password)),
          channel(o.channel), userID(o.userID), realPlayHandle(o.realPlayHandle),
          yolov8_pool(std::move(o.yolov8_pool)), tracker(std::move(o.tracker)),
          frame_id(o.frame_id.load()), decoded_frame_seq(o.decoded_frame_seq.load()),
          last_capture_ms(o.last_capture_ms.load()),
          stop_flag(o.stop_flag.load()),
          send_db(o.send_db), last_minute(o.last_minute),
          g_nPort(o.g_nPort), g_BGRImage(std::move(o.g_BGRImage)),
          frame_counter(o.frame_counter.load()), last_stat_time(o.last_stat_time),
          fps(o.fps), detection_mask(std::move(o.detection_mask)),
          mask_overlay(std::move(o.mask_overlay)),
          entry_line(o.entry_line),
          zone_line(std::move(o.zone_line)),
          entry_zone_depth(o.entry_zone_depth),
          zone_role(o.zone_role),
          role(o.role),
          in_count(o.in_count.load()), out_count(o.out_count.load()),
          retro_count(o.retro_count.load()),
          recent_counts(std::move(o.recent_counts)), filtered_count(o.filtered_count),
          zone_count(o.zone_count.load()),
          mode_count(o.mode_count.load()),
          total_count(o.total_count.load()),
          item_detector(std::move(o.item_detector)),
          item_alarm(o.item_alarm.load()),
          item_l1(o.item_l1.load()),
          item_l2(o.item_l2.load())
    { o.send_db = nullptr; o.g_nPort = -1; }
    
    ~CameraConfig() {
        stop_flag = true;
        if (send_db) sqlite3_close(send_db);
        if (g_nPort != -1) { PlayM4_Stop(g_nPort); PlayM4_CloseStream(g_nPort); PlayM4_FreePort(g_nPort); }
    }
    
    int calculateMode(int c) {
        std::lock_guard<std::mutex> lk(count_mutex);
        recent_counts.push_back(c);
        if (recent_counts.size() > 10) recent_counts.pop_front();
        std::unordered_map<int,int> freq;
        int m=0, mf=0;
        for (int x : recent_counts) { freq[x]++; if (freq[x] > mf) { mf=freq[x]; m=x; } }
        return filtered_count = m;
    }
    void clearRecent() { std::lock_guard<std::mutex> lk(count_mutex); recent_counts.clear(); }
};
std::vector<CameraConfig> cameras;

// 本相机角色的"有意义计数"文本：入口(role=0/201) 只会出 IN+RETRO；出口(role=1/200) 只会出 OUT+RETRO。
// 另一项在整个运行期恒为 0（入口相机不会产生 OUT 事件，出口相机不会产生 IN 事件），
// 显示它只是噪音 —— 日志 / OSD / 状态行三处都用这个函数，保证口径一致。
static std::string gateCountSummary(const CameraConfig& cfg) {
    if (cfg.role == 0)   // 入口：跨线进入类=IN，离开类=RETRO
        return "IN:" + std::to_string(cfg.in_count.load())
             + " RETRO:" + std::to_string(cfg.retro_count.load());
    return "OUT:" + std::to_string(cfg.out_count.load())
         + " RETRO:" + std::to_string(cfg.retro_count.load());
}

// ==================== 海康回调 ====================
void CALLBACK DecCBFun(int nPort, char* pBuf, int nSize, FRAME_INFO* pFrameInfo, void* nUser, int) {
    CameraConfig* cfg = reinterpret_cast<CameraConfig*>(nUser);
    if (pFrameInfo->nType == T_YV12 && !cfg->stop_flag) {
        std::unique_lock<std::mutex> lk(cfg->g_frame_mutex, std::try_to_lock);
        if (lk.owns_lock()) {
            cv::Mat yuv(pFrameInfo->nHeight + pFrameInfo->nHeight/2, pFrameInfo->nWidth, CV_8UC1, (uchar*)pBuf);
            // cvtColor 输出到全新 Mat 再 move 走：解码器下一帧会复用 pBuf，
            // 线程池(submitTask 按值)拿到的必须是独立缓冲
            cv::Mat bgr;
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YV12);
            cfg->g_BGRImage = std::move(bgr);
            cfg->last_capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            cfg->decoded_frame_seq++;
        }
    }
}
void CALLBACK RealDataCallBack_V30(LONG h, DWORD t, BYTE* p, DWORD sz, void* u) {
    CameraConfig* cfg = static_cast<CameraConfig*>(u);
    if (t == NET_DVR_STREAMDATA && sz > 0 && cfg->g_nPort != -1 && !cfg->stop_flag)
        PlayM4_InputData(cfg->g_nPort, p, sz);
}

// ==================== 数据库 ====================
sqlite3* getDB(const std::string& name) {
    static std::mutex mx;
    static std::map<std::string, sqlite3*> pool;
    std::lock_guard<std::mutex> lk(mx);
    if (pool.count(name)) return pool[name];
    sqlite3* db = nullptr;
    sqlite3_open_v2(name.c_str(), &db, SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_FULLMUTEX, nullptr);
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS detection_results(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "device TEXT NOT NULL, timestamp TEXT NOT NULL, box_count INTEGER NOT NULL);",
                 nullptr, nullptr, nullptr);
    return pool[name] = db;
}

// ==================== 通信模块依赖 ====================
sqlite3* GetDatabaseConnection(const std::string& db_name) {
    return getDB(db_name);
}
int GetMaxValidCount() {
    return g_max_box_count.load();
}

// ==================== 读配置 ====================
std::vector<CameraConfig> loadCameras(const std::string& f) {
    auto cfgs = parseCameraConfig(f);
    std::vector<CameraConfig> out;
    std::unordered_map<std::string,int> cnt;
    for (auto& c : cfgs) {
        CameraConfig cc;
        cc.ip = c.ip; cc.username = c.username; cc.password = c.password; cc.channel = c.channel;
        cc.detection_mask = createDetectionMask(c.width, c.height, c.detection_zones);
        cc.entry_line = c.entry_line;
        cc.zone_line = c.zone_line;                // 新格式 zin,/zout, 的判定线折线（≥2 点时优先）
        cc.zone_role = c.zone_role;                // -1=未指定(按 IP) / 0=入口 / 1=出口
        cc.entry_zone_depth = c.entry_zone_depth;  // 区域沿 x 平移量(带符号)
        if (!cc.detection_mask.empty()) {
            std::vector<std::vector<cv::Point>> ct;
            cv::findContours(cc.detection_mask, ct, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cc.mask_overlay = cv::Mat(c.height, c.width, CV_8UC3, cv::Scalar(0,0,0));
            for (auto& x : ct) cv::polylines(cc.mask_overlay, x, true, cv::Scalar(0,0,255), 3);
        }
        std::string base = c.ip + "_Ch" + std::to_string(c.channel);
        cc.unique_id = base + "_" + std::to_string(++cnt[base]);
        // 相机角色：优先用配置文件里显式写的角色（zin,=入口 / zout,=出口，由用户选择
        // 哪台相机画入口、哪台画出口）；没写角色时退回 IP 末段：200=出口，201=入口。
        if (cc.zone_role >= 0) {
            cc.role = cc.zone_role;
        } else {
            std::string tail = c.ip;
            auto dot = tail.find_last_of('.');
            if (dot != std::string::npos) tail = tail.substr(dot + 1);
            cc.role = (tail == "200") ? 1 : 0;  // 默认入口（含 201 及其他）
        }
        // 判定线来源自检：折线(zin,/zout,) > 旧格式(t1,) > 无。
        // "无"意味着这台相机永远不会产生 IN/OUT/RETRO —— 必须吵出来，不能静默失败。
        const bool has_poly_line = (cc.zone_line.size() >= 2);
        const bool has_trip_line = cc.entry_line.isValid();
        std::cout << "[INFO] " << cc.unique_id << " 判定线="
                  << (has_poly_line
                          ? ("折线(zin/zout) " + std::to_string(cc.zone_line.size()) + " 点")
                          : (has_trip_line ? std::string("旧格式 t1 (2 点)")
                                           : std::string("**无**")))
                  << ", 角色=" << (cc.role == 0 ? "入口" : "出口")
                  << ", 区域沿 x 平移 " << cc.entry_zone_depth << "px" << std::endl;
        if (!has_poly_line && !has_trip_line) {
            std::cerr << "[GATE] ERROR: " << cc.unique_id
                      << " 配置里既没有 zin,/zout, 折线也没有 t1, 直线 → 本相机不会产生任何"
                         " IN/OUT/RETRO 事件。请检查 cameras_config.txt 这一行末尾的判定线 token"
                         "（顶点必须成对且 ≥2 点）。\n";
        }
        out.emplace_back(std::move(cc));
    }
    return out;
}

// ==================== 绘制 ====================
// 框上 "P<id>" 标签的字号/线宽（看不清就只调这两个数）
static const double kIdLabelScale     = 1.30;   // 原 0.70
static const int    kIdLabelThickness = 3;      // 原 2

void draw(cv::Mat& img,
          const std::vector<Detection>& det,
          const std::vector<TrackedObject>& trk,
          const cv::Mat& overlay) {
    if (!overlay.empty() && overlay.size() == img.size()) {
        cv::Mat g, b; cv::cvtColor(overlay, g, cv::COLOR_BGR2GRAY);
        cv::threshold(g, b, 1, 255, cv::THRESH_BINARY); overlay.copyTo(img, b);
    }
    for (auto& d : det) { if (d.className != "people") cv::rectangle(img, d.box, cv::Scalar(0,255,0), 2); }
    for (auto& t : trk) {
        cv::Rect s = t.box & cv::Rect(0,0,img.cols,img.rows);
        if (s.area() <= 0) continue;
        cv::rectangle(img, t.box, t.color, 3);
        std::string lb = "P" + std::to_string(t.track_id);
        int bl; cv::Size ts = cv::getTextSize(lb, cv::FONT_HERSHEY_SIMPLEX, kIdLabelScale, kIdLabelThickness, &bl);
        int ty = t.box.y - 4;                       // 文字基线（默认画在框上方）
        if (ty < ts.height + 6) ty = ts.height + 6;  // 贴到画面顶边时下移，避免被裁掉
        cv::rectangle(img, cv::Point(t.box.x, ty-ts.height-4),
                      cv::Point(t.box.x+ts.width, ty), t.color, cv::FILLED);
        cv::putText(img, lb, cv::Point(t.box.x, ty),
                    cv::FONT_HERSHEY_SIMPLEX, kIdLabelScale, cv::Scalar(255,255,255), kIdLabelThickness);
    }
}

// ==================== 处理线程 ====================
// 滞留检测配置查找：优先当前目录，其次上一级 —— 兼容「在 build/ 下运行」和
// 「在项目根运行」两种习惯。09-22 教训：写死相对路径时，从 build/ 启动会
// 找不到文件 → 静默退回内置默认值（ref_bg_path 为空、mode 退回 mog2），
// 表现为"新功能好像没生效"。这里显式探测，找不到也由 loadAbandonConfig 打印提示。
static std::string resolveAbandonConfigPath() {
    static const char* kCands[] = {
        "abandon_config.txt",
        "../abandon_config.txt",
        "../../abandon_config.txt",
    };
    for (const char* p : kCands) {
        std::ifstream f(p);
        if (f.is_open()) return p;
    }
    return kCands[0];
}

void processCamera(CameraConfig& cfg) {
    cfg.send_db = getDB("results_v4.db");
    cfg.yolov8_pool = std::make_unique<Yolov8ThreadPool>();
    // 按相机绑核：cam0 主线程→NPU0, cam1 主线程→NPU2；两路副线程共享 NPU1。
    // 主线程永不冲突，副线程错峰使用剩余核（必须在 setUp 之前设置）。
    cfg.yolov8_pool->setNpuCores({cfg.cam_index == 0 ? 0 : 2, 1});
    if (cfg.yolov8_pool->setUp(g_model_path, g_npu_threads_per_camera) != NN_SUCCESS) {
        std::cerr << "YOLOv8 fail: " << cfg.ip << std::endl; return;
    }
    cfg.tracker = std::make_unique<ByteTrackTracker>(0.18f, 0.35f, 0.25f, 180, 2);

    // 物品滞留检测器（纯 CV 双背景差分，不用 NPU）。参数来自 abandon_config.txt，
    // 支持 roleN. 前缀按角色覆盖；文件不存在则用内置默认值。
    cfg.item_detector = std::make_unique<ItemAbandonmentDetector>(
        loadAbandonConfig(resolveAbandonConfigPath(), cfg.role), cfg.unique_id);

    // 海康登录
    NET_DVR_USER_LOGIN_INFO li = {0}; NET_DVR_DEVICEINFO_V40 di = {0};
    strncpy(li.sDeviceAddress, cfg.ip.c_str(), sizeof(li.sDeviceAddress)-1);
    li.wPort = 8000;
    strncpy(li.sUserName, cfg.username.c_str(), sizeof(li.sUserName)-1);
    strncpy(li.sPassword, cfg.password.c_str(), sizeof(li.sPassword)-1);
    { std::lock_guard<std::mutex> lk(g_hik_mutex); cfg.userID = NET_DVR_Login_V40(&li, &di); }
    if (cfg.userID < 0) { std::cerr << "Login fail: " << cfg.ip << std::endl; return; }
    if (!PlayM4_GetPort(&cfg.g_nPort) || !PlayM4_SetStreamOpenMode(cfg.g_nPort, STREAME_REALTIME)
        || !PlayM4_OpenStream(cfg.g_nPort, NULL, 0, 1024*1024)) {
        NET_DVR_Logout(cfg.userID); return;
    }
    PlayM4_SetDecCallBackExMend(cfg.g_nPort, DecCBFun, NULL, 0, &cfg);
    PlayM4_Play(cfg.g_nPort, 0);
    NET_DVR_PREVIEWINFO pi = {0}; pi.lChannel = cfg.channel; pi.dwStreamType = 0; pi.dwLinkMode = 0; pi.bBlocked = 1;
    { std::lock_guard<std::mutex> lk(g_hik_mutex); cfg.realPlayHandle = NET_DVR_RealPlay_V40(cfg.userID, &pi, RealDataCallBack_V30, &cfg); }

    cfg.last_stat_time = std::chrono::steady_clock::now();
    int dc = 0;
    int display_result_counter = 0;
    uint64_t last_submitted_decode_seq = 0;
    uint64_t last_perf_decode_seq = 0;
    int last_submitted_frame_id = -1;
    int last_processed_result_id = -1;
    // 跨标记线闸机计数（与 ByteTrack ID 绑定，ID 全程维护区域标记）。
    // 几何：R0 区域外(判定线外侧 = 出口侧) / R1 区域(判定线 ∪ 其 x 平移副本) / R2 检测区内。
    // 判定线 = 区域边缘折线（zin,/zout, 给出，点数不限；无则退回 t1 的 2 点直线）。
    // 触发源 = 本相机判定线的"侧翻转边沿"：每跨一次线恰好计一次（与 ID 生命周期解耦，
    // 同一 ID 多次往返每次都计）。角色只负责给事件贴标签：
    //   入口：线外→线内 = IN，线内→线外 = RETRO
    //   出口：线外→线内 = RETRO，线内→线外 = OUT
    GateRole gate_role = (cfg.role == 1) ? GATE_ROLE_EXIT : GATE_ROLE_ENTRY;
    // 最后一个参数 = 跨线"站住"确认帧数：只有在新的一侧连续观测 ≥ kCrossDwellFrames 帧
    // 才结算这次跨线；中途回到原侧（在线上蹭/徘徊）则整对事件作废。见该常量处的注释。
    GateIdCounter gate_counter = cfg.zone_line.size() >= 2
        ? GateIdCounter(gate_role, cfg.zone_line, cfg.entry_zone_depth, cfg.detection_mask,
                        kCrossDwellFrames)
        : GateIdCounter(gate_role, cfg.entry_line, cfg.entry_zone_depth, cfg.detection_mask,
                        kCrossDwellFrames);
    int gate_flash = 0;  // 计数事件后的标记线闪烁帧数

    while (!cfg.stop_flag && g_running) {
        {
            std::lock_guard<std::mutex> lk(cfg.g_frame_mutex);
            const uint64_t decode_seq = cfg.decoded_frame_seq.load();
            if (!cfg.g_BGRImage.empty() && decode_seq != last_submitted_decode_seq) {
                last_submitted_frame_id = cfg.frame_id++;
                // move 交出所有权：帧缓冲零拷贝入队（DecCBFun 已保证是独立缓冲）
                cfg.yolov8_pool->submitTask(std::move(cfg.g_BGRImage), last_submitted_frame_id,
                                            cfg.last_capture_ms.load());
                last_submitted_decode_seq = decode_seq;
            }
        }

        cv::Mat raw; int rid=0, rbc=0; int64_t capture_ms = 0;
        std::vector<Detection> dets;
        // 阻塞等待最新推理结果（条件变量，非 sleep 忙等待）。20ms 超时回顶部提交新帧。
        if (cfg.yolov8_pool->waitLatestResultRaw(raw, rid, rbc, dets,
                                                 &capture_ms, 20) != NN_SUCCESS) {
            continue;
        }
        if (rid <= last_processed_result_id) continue;
        last_processed_result_id = rid;
        cfg.frame_counter++;
        std::vector<TrackedObject> tracks;
        if (!dets.empty()) {
            std::vector<Detection> people;
            for (auto& d : dets) {
                if (d.className != "people") continue;
                people.push_back(d);
            }
            tracks = cfg.tracker->update(people, raw, raw.cols, raw.rows);
        } else {
            // Advance Kalman tracks on empty detector output so stale tracks expire in real time.
            tracks = cfg.tracker->update({}, raw, raw.cols, raw.rows);
        }
        // Preserve acquisition time through asynchronous inference. Fusion
        // must not depend on when an overloaded worker returned this result.
        for (auto& track : tracks) {
            track.result_frame_id = rid;
            track.capture_timestamp_ms = capture_ms;
        }
        // 检测区域内人数（脚底在检测区域 mask 内），仅用于显示。
        int zn = 0;
        if (!cfg.detection_mask.empty()) {
            for (const auto& t : tracks) {
                int cx = t.box.x + t.box.width / 2;
                int cy = t.box.y + (int)(t.box.height * 0.92f);
                cx = std::max(0, std::min(cx, cfg.detection_mask.cols - 1));
                cy = std::max(0, std::min(cy, cfg.detection_mask.rows - 1));
                if (cfg.detection_mask.at<uchar>(cy, cx) != 0) zn++;
            }
        }
        cfg.zone_count = zn;
        cfg.mode_count = cfg.calculateMode(zn);
        cfg.total_count = (int)tracks.size();

        // ==================== 跨标记线闸机计数（与 ByteTrack ID 绑定）====================
        // 触发源 = 本相机标记线(t1)的侧翻转边沿：线外→线内 = 进入类，线内→线外 = 离开类。
        // 201(入口)：进入类=IN，离开类=RETRO；200(出口)：进入类=RETRO，离开类=OUT。
        // 同一 ID 可反复往返，每跨一次线计一次；遮挡期间预测框只维持 ID 连续、不改变所在侧；
        // ID 被切换时由交接继承"所在侧"（不继承一次性封顶），不重计不漏计。
        auto gate_events = gate_counter.update(tracks);
        for (const auto& event : gate_events) {
            gate_flash = 15;
            const char* name = nullptr;
            if (event.type == GATE_EVENT_IN) {
                cfg.in_count++;
                name = "IN";
            } else if (event.type == GATE_EVENT_OUT) {
                cfg.out_count++;
                name = "OUT";
            } else {
                cfg.retro_count++;
                name = "RETRO";
            }
            const std::string origin_tag = event.origin_id >= 0
                ? (" origin=" + std::to_string(event.origin_id)) : std::string();
            std::cerr << "[GATE] ts=" << capture_ms << " " << cfg.unique_id
                      << " track=" << event.track_id << " event=" << name
                      << " reason=" << event.reason
                      << " region=R" << event.region
                      << (event.via_entry_band ? " via_entry_band" : "")
                      << (event.settled ? " settled" : "")
                      << origin_tag
                      << " lost=" << event.lost_frames
                      << " foot=(" << event.foot.x << "," << event.foot.y << ")"
                      << " " << gateCountSummary(cfg) << "\n";
        }
        // 每个 ID 的区域标记（req1）：R0=区域外(出口侧) R1=入口带 R2=检测区内，
        // 以及"是否标记在区域内"（粘滞，只有确认离开才翻转）。
        std::map<int, GateIdMark> id_marks;
        for (const auto& m : gate_counter.marks()) id_marks[m.track_id] = m;
        const int ids_inside = gate_counter.insideCount();

        // ==================== 物品滞留检测（纯CV背景差分，与计数解耦）====================
        // ⚠️ 必须在 draw(raw,...) 之前调用：raw 一旦被画上框/掩膜，就会污染背景差分。
        // 二级告警(PUSH) = "物品滞留认定"，对应硬指标默认 60s，见 AbandonConfig::push_sec。
        std::vector<ItemEvent> item_events;
        if (cfg.item_detector) {
            item_events = cfg.item_detector->update(raw, tracks, cfg.detection_mask,
                                                    capture_ms, rid);
            for (const auto& e : item_events) {
                std::cerr << "[ITEM] ts=" << e.ts_ms << " " << cfg.unique_id
                          << " item=" << e.item_id
                          << " event=" << itemEventName(e.type)
                          << " static=" << (int)(e.static_sec + 0.5) << "s"
                          << " owner=" << e.owner_id
                          << " box=(" << e.box.x << "," << e.box.y << ","
                          << e.box.width << "," << e.box.height << ")"
                          << (e.note.empty() ? "" : (" note=" + e.note))
                          << (e.evidence_dir.empty() ? "" : (" evidence=" + e.evidence_dir))
                          << "\n";
            }
            cfg.item_alarm = cfg.item_detector->alarmCount();
            cfg.item_l1 = cfg.item_detector->level1Count();
            cfg.item_l2 = cfg.item_detector->level2Count();
        }
        if (g_enable_zone_trace && !gate_events.empty()) {
            for (const auto& kv : id_marks) {
                const GateIdMark& m = kv.second;
                // 只打印当前仍在 tracker 输出里的 ID：已消失(lost>0)或已被交接封存的旧 ID
                // 不再刷屏，避免日志里出现"幽灵 ID"（交接来源已由 inherited_from= 表达）
                if (m.lost_frames > 0) continue;
                std::cerr << "[GATEMARK] " << cfg.unique_id << " track=" << m.track_id
                          << " region=R" << m.region
                          << " mark=" << (m.inside ? "INSIDE" : "OUTSIDE")
                          << " entered=" << (m.entered ? 1 : 0)
                          << " exited=" << (m.exited ? 1 : 0)
                          << (m.inherited ? " inherited_from=" : " origin=")
                          << m.origin_id
                          << " lost=" << m.lost_frames << "\n";
            }
        }
        if (g_enable_display &&
            (++display_result_counter % kDisplayEveryResults == 0)) {
        draw(raw, dets, tracks, cfg.mask_overlay);

        cv::Mat dp; cv::resize(raw, dp, cv::Size(cfg.display_w, cfg.display_h));
        // 画面: 绘制入口/出口矩形 (按外侧边法向 + depth 符号, 不依赖 gate_dir)
        {
            float sx = (float)cfg.display_w / raw.cols;
            float sy = (float)cfg.display_h / raw.rows;
            auto scalePoly = [&](const std::vector<cv::Point2f>& src) {
                std::vector<cv::Point> out;
                out.reserve(src.size());
                for (const auto& p : src)
                    out.emplace_back((int)(p.x * sx), (int)(p.y * sy));
                return out;
            };
            // 角色配色：入口绿 / 出口红（与工具端一致）
            const bool is_entry_cam = (gate_role == GATE_ROLE_ENTRY);
            const cv::Scalar line_col = is_entry_cam ? cv::Scalar(0,255,0)
                                                     : cv::Scalar(0,0,255);
            const cv::Scalar zone_col = is_entry_cam ? cv::Scalar(0,160,0)
                                                     : cv::Scalar(0,0,160);
            // 区域 = 判定线 ∪ 其沿 x 平移副本。几何直接取自 counter（与计数同源，
            // 避免"画面画的区域"与"实际判定用的几何"不一致）。
            const std::vector<cv::Point2f>& ring = gate_counter.zoneRing();
            if (ring.size() >= 3) {
                std::vector<cv::Point> pts = scalePoly(ring);
                cv::Mat overlay; dp.copyTo(overlay);
                cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{pts}, zone_col);
                cv::addWeighted(overlay, 0.4, dp, 0.6, 0, dp);
                cv::polylines(dp, pts, true, zone_col, 2);
                // 平移量标签（带符号）
                cv::Moments m = cv::moments(pts);
                if (m.m00 > 1.0) {
                    char dz[24];
                    snprintf(dz, 24, "x%+.0fpx", cfg.entry_zone_depth);
                    cv::putText(dp, dz, cv::Point((int)(m.m10/m.m00) - 24, (int)(m.m01/m.m00)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, zone_col, 1);
                }
            }
            // 判定线（区域边缘折线）：跨过它 = 一次计数
            const std::vector<cv::Point2f>& jline = gate_counter.judgingLine();
            if (jline.size() >= 2) {
                std::vector<cv::Point> pts = scalePoly(jline);
                cv::polylines(dp, pts, false, line_col, 3);
                for (const auto& p : pts) cv::circle(dp, p, 3, line_col, -1);   // 折点
            }
        }
        // 轨迹调试: 脚底点 + 每 ID 区域标记 + 计数闪烁
        {
            float sx = (float)cfg.display_w / raw.cols;
            float sy = (float)cfg.display_h / raw.rows;
            // 计数事件后闪烁判定线
            if (gate_flash > 0) {
                gate_flash--;
                const auto& jl = gate_counter.judgingLine();
                if (jl.size() >= 2) {
                    std::vector<cv::Point> pts;
                    pts.reserve(jl.size());
                    for (const auto& p : jl)
                        pts.emplace_back((int)(p.x*sx), (int)(p.y*sy));
                    cv::polylines(dp, pts, false, cv::Scalar(0,255,255), 5, cv::LINE_AA);  // 亮青闪烁
                }
            }
            // 脚底点 + 区域标记：绿色=标记在区域内, 橙色=标记在区域外(R0)
            for (auto& t : tracks) {
                cv::Point2f foot(t.box.x + t.box.width/2.0f, t.box.y + t.box.height*0.92f);
                auto mit = id_marks.find(t.track_id);
                const bool has_mark = (mit != id_marks.end());
                // 没有标记的 ID（极少数：状态已清理但 tracker 仍在输出预测框）用实时区域兜底，
                // 保证画面上不会再出现标不出区域的 "R?" 幽灵框
                const int reg = has_mark ? mit->second.region : gate_counter.regionOf(foot);
                const bool inside_mark = has_mark ? mit->second.inside
                                                  : (reg == GATE_REGION_DETECT);
                const cv::Scalar foot_col = inside_mark ? cv::Scalar(0,255,0)
                                                        : cv::Scalar(0,165,255);
                cv::circle(dp, cv::Point((int)(foot.x*sx), (int)(foot.y*sy)), 4, foot_col, -1);
                // ID 区域标记文本：R0 区域外 / R1 入口带 / R2 检测区内 + INSIDE/OUTSIDE 标记
                const std::string tag = "R" + std::to_string(reg)
                                      + (inside_mark ? " INSIDE" : " OUTSIDE");
                const int bx = (int)(t.box.x * sx);
                const int by = (int)((t.box.y + t.box.height) * sy) + 14;
                cv::putText(dp, tag, cv::Point(bx, by), cv::FONT_HERSHEY_SIMPLEX,
                            0.45, foot_col, 1);
            }
        }
        // 滞留物框 + 主人连线：红=L2推送, 黄=L1提醒, 橙=静止候选, 灰=移动中。
        // ⚠️ OpenCV 的 putText 只支持 ASCII —— 标签一律用英文短词，中文会画成 "???"。
        if (cfg.item_detector) {
            const float sx = (float)cfg.display_w / raw.cols;
            const float sy = (float)cfg.display_h / raw.rows;
            for (const auto& it : cfg.item_detector->items()) {
                if (it.missing_frames > 0) continue;   // 本帧未观测（遮挡/已消失）不画
                cv::Scalar col;
                std::string tag;
                const int sec = it.unattended_since_ms > 0
                    ? (int)((capture_ms - it.unattended_since_ms) / 1000) : 0;
                switch (it.state) {
                    case ItemState::LEVEL2_PUSH:
                        col = cv::Scalar(0,0,255);     tag = "L2 PUSH " + std::to_string(sec) + "s"; break;
                    case ItemState::LEVEL1_REMIND:
                        col = cv::Scalar(0,215,255);   tag = "L1 REMIND " + std::to_string(sec) + "s"; break;
                    case ItemState::STATIC_CANDIDATE:
                        col = cv::Scalar(0,165,255);   tag = "STATIC " + std::to_string(sec) + "s"; break;
                    default:
                        col = cv::Scalar(160,160,160); tag = "MOVING"; break;
                }
                const cv::Rect rb((int)(it.box.x*sx), (int)(it.box.y*sy),
                                  (int)(it.box.width*sx), (int)(it.box.height*sy));
                cv::rectangle(dp, rb, col, 2);
                const int ty = (rb.y - 6 < 12) ? (rb.y + 16) : (rb.y - 6);
                cv::putText(dp, tag, cv::Point(rb.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 2);
                // 主人连线（黄线：人框中心 → 物品质心），一眼看出"这是谁的东西"
                if (it.owner_id >= 0 && !it.owner_left) {
                    for (const auto& t : tracks) {
                        if (t.track_id != it.owner_id) continue;
                        cv::line(dp,
                                 cv::Point((int)((t.box.x + t.box.width*0.5f)*sx),
                                           (int)((t.box.y + t.box.height*0.5f)*sy)),
                                 cv::Point((int)(it.centroid.x*sx), (int)(it.centroid.y*sy)),
                                 cv::Scalar(0,255,255), 1, cv::LINE_AA);
                        break;
                    }
                }
            }
        }
        // 第一行：黄色，基础信息（含滞留告警数 ITEM）
        std::string item_tag = " ITEM:" + std::to_string(cfg.item_alarm.load());
        if (cfg.item_l2.load() > 0) item_tag += "(L2:" + std::to_string(cfg.item_l2.load()) + ")";
        else if (cfg.item_l1.load() > 0) item_tag += "(L1:" + std::to_string(cfg.item_l1.load()) + ")";
        std::string info1 = cfg.unique_id + " Zone:" + std::to_string(zn)
                          + " Tot:" + std::to_string((int)tracks.size())
                          + " IDsIn:" + std::to_string(ids_inside)
                          + item_tag;
        cv::putText(dp, info1, cv::Point(5, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,255), 1);
        // 第二行：红色大字，只显示本相机角色的计数（入口=IN/RETRO，出口=OUT/RETRO）
        std::string info2 = gateCountSummary(cfg);
        cv::putText(dp, info2, cv::Point(5, 42), cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0,0,255), 2);
        { std::lock_guard<std::mutex> lk(cfg.display_ptr_mutex); cfg.display_frame_ptr = std::make_shared<cv::Mat>(dp); }
        }

        if (dc++ % 30 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double el = std::chrono::duration_cast<std::chrono::milliseconds>(
                now-cfg.last_stat_time).count()/1000.0;
            const uint64_t decode_seq = cfg.decoded_frame_seq.load();
            const double decode_fps = el > 0
                ? (decode_seq - last_perf_decode_seq) / el : 0.0;
            if (el>0) cfg.fps=30.0/el;
            cfg.last_stat_time = now;
            last_perf_decode_seq = decode_seq;
            const auto perf = cfg.yolov8_pool->getStats();
            std::cout << cfg.unique_id << " FPS:" << cfg.fps << " Zone:" << zn
                     << " Tot:" << tracks.size()
                     << " IDsIn:" << ids_inside
                     << " ITEM:" << cfg.item_alarm.load()
                     << "(L1:" << cfg.item_l1.load() << " L2:" << cfg.item_l2.load() << ")"
                     << " " << gateCountSummary(cfg)
                     << std::endl;
            std::cout << "[PERF] " << cfg.unique_id
                      << " decode=" << decode_seq
                      << " decode_fps=" << decode_fps
                      << " submitted=" << perf.submitted
                      << " processed=" << perf.processed
                      << " drop_pending=" << perf.dropped_pending
                      << " drop_result=" << perf.dropped_results
                      << " queue=" << perf.queue_depth
                      << " result_id=" << rid
                      << " lag=" << std::max(0, last_submitted_frame_id - rid)
                      << std::endl;
            // --trace：周期打印每个活跃 ID 到【本相机判定线】的几何关系。
            // 排查"某台相机全程 0 事件"时看这一行就够：
            //   line_signed 一直同号且数值很大 → 人在线的同一侧，根本没跨过它；
            //   带 (跨度外) 标记 → 人没走到这条折线覆盖的那段边界上（不是漏计，是线画的位置/长度问题）。
            if (g_enable_zone_trace) {
                // 诊断注记（GateIdCounter::notes()）：只在"关键瞬间"产生，用来回答
                // "这个人在视频里明明有轨迹，为什么没计数" —— 例如某几帧只有预测框
                // (observed=false，画面上有框但计数模块看不到)、某 ID 出生时就在线内侧等。
                for (const auto& n : gate_counter.notes()) {
                    std::cout << "[CAND] " << cfg.unique_id << " " << n << std::endl;
                }
                for (const auto& kv : id_marks) {
                    const GateIdMark& m = kv.second;
                    if (m.lost_frames > 0) continue;
                    std::cout << "[TRACE] " << cfg.unique_id << " track=" << m.track_id
                              << " region=R" << m.region
                              << " line_signed=" << (int)m.line_signed << "px"
                              << " side=" << (m.line_side < 0 ? "?" : (m.line_side == 1 ? "in" : "out"))
                              << (m.line_active ? "" : " (跨度外)")
                              << " mark=" << (m.inside ? "INSIDE" : "OUTSIDE")
                              << std::endl;
                }
            }
        }
    }
    { std::lock_guard<std::mutex> lk(g_hik_mutex);
      if (cfg.realPlayHandle>=0) NET_DVR_StopRealPlay(cfg.realPlayHandle);
      if (cfg.userID>=0) NET_DVR_Logout(cfg.userID); }
}

void sig(int) { g_running = false; }

int main(int argc, char** argv) {
    std::signal(SIGINT,sig); std::signal(SIGTERM,sig);
    cv::setNumThreads(0);
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <model> <config> [threads=2] [--headless|-n] [--trace]" << std::endl; return -1; }
    g_model_path = argv[1];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless" || arg == "-n") {
            g_enable_display = false;
        } else if (arg == "--trace") {
            g_enable_zone_trace = true;
        } else if (!arg.empty() && arg[0] != '-') {
            g_npu_threads_per_camera = std::max(1, std::min(atoi(arg.c_str()), 4));
        }
    }
    if (g_enable_display) {
        XInitThreads();
        setenv("DISPLAY",":0",1);
    }

#ifdef ENABLE_SERIAL
    init_serial_comm("/dev/ttyS9");
#endif
    NET_DVR_Init(); NET_DVR_SetConnectTime(3000,3);
    cameras = loadCameras(argv[2]);
    if (cameras.empty()) { NET_DVR_Cleanup(); return -1; }
    for (size_t i = 0; i < cameras.size(); i++) cameras[i].cam_index = (int)i;
    std::cout << "V4 Starting " << cameras.size() << " cameras (" << g_npu_threads_per_camera << " threads each)" << std::endl;

    std::vector<std::thread> ths;
    for (auto& c : cameras) ths.emplace_back(processCamera, std::ref(c));

    std::string wn = "V4 Combined";
    if (g_enable_display) {
        cv::namedWindow(wn, cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(wn, cameras.size()*cameras[0].display_w, cameras[0].display_h);
    }
    auto last_display_time = std::chrono::steady_clock::now();

    while (g_running) {
        if (g_enable_display) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_display_time).count();
            if (elapsed >= kDisplayRefreshMs) {
                last_display_time = now;
                std::vector<std::shared_ptr<cv::Mat>> frs;
                for (auto& c : cameras) {
                    std::lock_guard<std::mutex> lk(c.display_ptr_mutex);
                    frs.push_back(c.display_frame_ptr);
                }
                cv::Mat comb; bool ok=false;
                for (size_t i=0; i<frs.size(); i++) {
                    if (frs[i] && !frs[i]->empty()) {
                        if (!ok) { comb=frs[i]->clone(); ok=true; }
                        else cv::hconcat(comb, *frs[i], comb);
                    }
                }
                if (ok) {
                    int max_zone = 0, total_in = 0, total_out = 0, total_retro = 0, total_item = 0;
                    for (auto& c : cameras) {
                        max_zone = std::max(max_zone, c.zone_count.load());
                        total_in += c.in_count.load();
                        total_out += c.out_count.load();
                        total_retro += c.retro_count.load();
                        total_item += c.item_alarm.load();
                    }
                    char mx[64]; snprintf(mx, 64, "Max Zone: %d", max_zone);
                    cv::putText(comb, mx, cv::Point(5, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
                    char fused[96]; snprintf(fused, 96, "Total IN:%d OUT:%d RETRO:%d ITEM:%d",
                                             total_in, total_out, total_retro, total_item);
                    cv::putText(comb, fused, cv::Point(5, 80),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
                    cv::imshow(wn, comb);
                }
            }
            if (cv::waitKey(1)==27) g_running=false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    for (auto& t : ths) if (t.joinable()) t.join();
    NET_DVR_Cleanup();
    int total_in = 0, total_out = 0, total_retro = 0, total_item = 0;
    for (auto& c : cameras) { total_in += c.in_count.load(); total_out += c.out_count.load(); total_retro += c.retro_count.load(); total_item += c.item_alarm.load(); }
    std::cout << "Total Final IN:" << total_in
              << " OUT:" << total_out
              << " RETRO:" << total_retro
              << " ITEM:" << total_item << std::endl;
    std::cout << "V4 stopped." << std::endl;
    return 0;
}

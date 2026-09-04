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
#include "tripwire_counter.h"
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

// comm.cpp 依赖的全局变量
std::atomic<uint16_t> g_max_box_count(0);
std::atomic<bool> g_need_clear_queues{false};

// ==================== 双摄像头融合去重 ====================
struct FusedEvent {
    int64_t ts_ms;       // 事件毫秒时间戳
    int cam_idx;         // 摄像头索引 0/1
    int event_type;      // 0=IN, 1=OUT, 2=RETRO
    int line_type;       // 1=ENTRY, 2=EXIT，避免两种RETRO互相误配
};
std::deque<FusedEvent> g_event_queue;
std::mutex g_event_mutex;
std::atomic<int> g_fused_in{0};
std::atomic<int> g_fused_out{0};
std::atomic<int> g_fused_retro{0};
static const int64_t DEDUP_WINDOW_MS = 1200; // 实测双摄同一事件最大延迟约1079ms

bool reportFusedEvent(int cam_idx, int event_type, int line_type, int64_t capture_ms) {
    const auto now_ms = capture_ms;
    std::lock_guard<std::mutex> lk(g_event_mutex);
    while (!g_event_queue.empty() && (now_ms - g_event_queue.front().ts_ms) > DEDUP_WINDOW_MS)
        g_event_queue.pop_front();
    bool found = false;
    for (auto it = g_event_queue.begin(); it != g_event_queue.end(); ) {
        if (it->event_type == event_type && it->line_type == line_type
            && it->cam_idx != cam_idx) {
            std::cerr << "[FUSE_MATCH] type="
                      << (event_type == 0 ? "IN" : (event_type == 1 ? "OUT" : "RETRO"))
                      << " line=" << (line_type == 1 ? "ENTRY" : "EXIT")
                      << " cams=" << it->cam_idx << "+" << cam_idx
                      << " dt=" << (now_ms - it->ts_ms) << "ms\n";
            it = g_event_queue.erase(it); found = true; break;
        } else { ++it; }
    }
    if (!found) g_event_queue.push_back({now_ms, cam_idx, event_type, line_type});
    return !found;
}

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
    
    // 绊线区域（人行方向统计）
    Tripwire entry_line;
    Tripwire exit_line;
    float entry_zone_depth = 50.0f;  // 入口区域深度(px), 绊线向内延伸
    float exit_zone_depth = 50.0f;   // 出口区域深度(px)
    std::atomic<int> forward_count{0};    // 顺行累计
    std::atomic<int> exit_count{0};       // 出门累计
    std::atomic<int> retrograde_count{0}; // 逆行累计
    
    std::deque<int> recent_counts;
    std::mutex count_mutex;
    int filtered_count{0};
    std::atomic<int> zone_count{0};
    std::atomic<int> mode_count{0};
    std::atomic<int> total_count{0};

    std::shared_ptr<cv::Mat> display_frame_ptr;
    std::mutex display_ptr_mutex;
    int display_w = 640, display_h = 360;

    CameraConfig() = default;
    CameraConfig(const CameraConfig&) = delete;
    CameraConfig& operator=(const CameraConfig&) = delete;
    
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
          entry_line(o.entry_line), exit_line(o.exit_line),
          entry_zone_depth(o.entry_zone_depth), exit_zone_depth(o.exit_zone_depth),
          forward_count(o.forward_count.load()), exit_count(o.exit_count.load()),
          retrograde_count(o.retrograde_count.load()),
          recent_counts(std::move(o.recent_counts)), filtered_count(o.filtered_count),
          zone_count(o.zone_count.load())
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
        cc.exit_line = c.exit_line;
        cc.entry_zone_depth = c.entry_zone_depth;  // 用户画的区域深度, 0=线模式
        cc.exit_zone_depth = c.exit_zone_depth;
        if (!cc.detection_mask.empty()) {
            std::vector<std::vector<cv::Point>> ct;
            cv::findContours(cc.detection_mask, ct, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cc.mask_overlay = cv::Mat(c.height, c.width, CV_8UC3, cv::Scalar(0,0,0));
            for (auto& x : ct) cv::polylines(cc.mask_overlay, x, true, cv::Scalar(0,0,255), 3);
        }
        std::string base = c.ip + "_Ch" + std::to_string(c.channel);
        cc.unique_id = base + "_" + std::to_string(++cnt[base]);
        out.emplace_back(std::move(cc));
    }
    return out;
}

// ==================== 绘制 ====================
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
        int bl; cv::Size ts = cv::getTextSize(lb, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
        cv::rectangle(img, cv::Point(t.box.x, t.box.y-ts.height-4),
                      cv::Point(t.box.x+ts.width, t.box.y), t.color, cv::FILLED);
        cv::putText(img, lb, cv::Point(t.box.x, t.box.y-4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
    }
}

// ==================== 处理线程 ====================
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
    // Legacy state declarations are kept below for display compatibility.
    enum CrossLine { CL_NONE = 0, CL_ENTRY = 1, CL_EXIT = 2 };
    static const int TRACK_REARM_GAP_FRAMES = 4;   // 轨迹断裂 >=4 帧即重臂(与 HEAD 一致)
    struct CrossState {
        std::deque<cv::Point2f> trail;
        CrossLine first_line = CL_NONE;
        int first_frame = 0;
        cv::Point2f first_pt;
        cv::Point2f last_foot;
        int side_entry = 0;
        int side_exit = 0;
        int cooldown_entry = 0;  // 入口线计数后冷却帧数
        int cooldown_exit = 0;   // 出口线计数后冷却帧数
        bool virt_entry_done = false;  // 轨迹虚拟入口只触发一次
        bool virt_exit_done = false;   // 轨迹虚拟出口只触发一次
        bool entry_seen = false;       // 已真实或虚拟经过入口，禁止后续虚拟出口误判
        CrossLine last_cross_line = CL_NONE;  // 最近跨的线(画面闪烁用)
        int flash_frames = 0;                  // 闪烁剩余帧数
        int lost_frames = 0;
        int age_frames = 0;                    // track 从本次出现开始累计的有效帧数
        bool armed_from_outside = false;        // 入口: 先确认人在检测区外, 再允许入区计数
        int inside_zone_frames = 0;
        int outside_zone_frames = 0;
        // 检测区(脚底跳变)计数字段
        int region = 0;             // 当前区域 0区域外 1入口区域 2检测区 3出口区域
        int enter_side = 0;
        int exit_side = 0;
        int cooldown_zone = 0;
        bool last_was_retro = false;
        int last_event = 0;
    };
    std::map<int, CrossState> cs_map;
    enum { S_OUT, S_ENTER, S_IN, S_LEAVE };
    struct TM { int s = S_OUT; int n = 0; cv::Point2f p; int out_n = 0; };
    std::map<int, TM> zone_states;
    TripwireCounter tripwire_counter(cfg.entry_line, cfg.exit_line,
                                     cfg.entry_zone_depth, cfg.exit_zone_depth,
                                     cfg.detection_mask);
    tripwire_counter.setDebugEnabled(g_enable_zone_trace);
    // (注: 区域/带符号距离判定已全部收拢到 TripwireCounter, 此处不再有本地几何 lambda)

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
        if (cfg.yolov8_pool->tryGetLatestResultRaw(raw, rid, rbc, dets,
                                                   &capture_ms) != NN_SUCCESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue;
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
        // Zone count with ByteTrack ID continuity.
        for (auto it = zone_states.begin(); it != zone_states.end(); ) {
            bool al = false;
            for (auto& t : tracks) if (t.track_id == it->first) { al = true; break; }
            if (!al && it->second.s <= S_ENTER) it = zone_states.erase(it);
            else if (!al && it->second.n++ > 60) it = zone_states.erase(it);
            else ++it;
        }
        int zn = 0;
        for (auto& t : tracks) {
            int tid = t.track_id;
            auto& m = zone_states[tid];
            cv::Point2f c(t.box.x+t.box.width/2.0f, t.box.y+t.box.height*0.92f);
            // 区域内 = 同时在 entry 矩形内 AND exit 矩形内 (用户画的两根矩形围成的通道)
            bool inZ = tripwire_counter.inZone(c);
            switch (m.s) {
            case S_OUT: case S_ENTER:
                if (inZ) { m.n++; if (m.n>=2) { m.s=S_IN; m.p=c; m.n=0; } }
                else { m.s=S_OUT; m.n=0; } break;
            case S_IN:
                if (!inZ) {
                    if (cv::norm(c-m.p)>80) { m.s=S_LEAVE; m.n=1; m.out_n=1; }
                } else { m.p=c; m.out_n=0; } break;
            case S_LEAVE:
                if (!inZ) {
                    m.n++; m.out_n++;
                    // 快速离开判定: 用脚底点判定是否仍在通道内 (0/1)
                    bool fast_exit = (m.out_n>=4 && !tripwire_counter.inZone(c));
                    if (fast_exit || m.n>=8) { m.s=S_OUT; m.n=0; }
                } else { m.s=S_IN; m.p=c; m.n=0; m.out_n=0; } break;
            }
            if (m.s==S_IN || m.s==S_LEAVE) zn++;
        }
        cfg.zone_count = zn;
        cfg.mode_count = cfg.calculateMode(zn);
        cfg.total_count = (int)tracks.size();

        // ==================== 检测区(区域状态机)计数 ====================
        // 区域模型: 0区域外 1入口区域 2检测区(入口线↔出口线走廊) 3出口区域
        // 触发: 脚底所在"区域"的跳变(4帧平滑后), 进入/离开检测区即计(保留双摄融合).
        //   IN  : 非出口侧 -> 进入检测区   (区域外/入口区域 侧进)
        //   RETRO: 出口区域侧 -> 进入检测区 (逆行进)
        //   OUT  : 离开检测区 -> 出口区域/区域外
        //   RETRO: 离开检测区 -> 入口区域   (逆行出)
        // 首帧仅建立区域(不计数); 若一出生已在检测区, 按所处半区推断方向补计.
        // (实现全部在 TripwireCounter::update 内, 原 if(false&&...) 死代码块已清除)

        // Shared counter: only real observations can create an event.
        auto counter_events = tripwire_counter.update(tracks);
        if (g_enable_zone_trace) {
        auto trace_events = tripwire_counter.takeDebugEvents();
        for (const auto& trace : trace_events) {
            auto ts = capture_ms;
            std::cerr << "[ZONE_TRACE] ts=" << ts << " " << cfg.unique_id
                      << " track=" << trace.track_id
                      << " reason=" << trace.reason
                      << " raw=" << trace.raw_region
                      << " stable=" << trace.stable_region
                      << " pending=" << trace.pending_region
                      << " pending_n=" << trace.pending_frames
                      << " approach=" << trace.approach_side
                      << " approach_n=" << trace.approach_observations
                      << " departure=" << trace.departure_side
                      << " departure_n=" << trace.departure_observations
                      << " committed=" << (trace.inside_committed ? 1 : 0)
                      << " observed_n=" << trace.observed_streak
                      << " hit=" << trace.hit_streak
                      << " lost=" << trace.lost_frames
                      << " conf=" << trace.confidence
                      << " move=" << trace.movement
                      << " foot=(" << trace.foot.x << "," << trace.foot.y << ")\n";
        }
        } else {
            tripwire_counter.takeDebugEvents();
        }
        for (const auto& event : counter_events) {
            auto ts = capture_ms;
            if (event.type == TW_EVENT_IN) {
                cfg.forward_count++;
                if (reportFusedEvent(cfg.cam_index, 0, 1, capture_ms)) g_fused_in++;
            } else if (event.type == TW_EVENT_OUT) {
                cfg.exit_count++;
                if (reportFusedEvent(cfg.cam_index, 1, 2, capture_ms)) g_fused_out++;
            } else {
                cfg.retrograde_count++;
                int line = event.side == 2 ? 2 : 1;
                if (reportFusedEvent(cfg.cam_index, 2, line, capture_ms)) g_fused_retro++;
            }
            std::cerr << "[ZONE] ts=" << ts << " " << cfg.unique_id
                      << " track=" << event.track_id
                      << " event=" << (event.type == TW_EVENT_IN ? "IN" :
                                         event.type == TW_EVENT_OUT ? "OUT" : "RETRO")
                      << " side=" << (event.side == 1 ? "entry" : "exit")
                      << " source=" << (event.birth_confirmed
                                             ? "region-birth-confirmed"
                                             : event.loss_confirmed
                                                   ? "region-loss-confirmed"
                                                   : "region-transition")
                      << " observed=" << (event.loss_confirmed ? 0 : 1)
                      << " conf=" << event.confidence
                      << " move=" << event.movement
                      << " foot=(" << event.foot.x << "," << event.foot.y << ")"
                      << " IN:" << cfg.forward_count.load()
                      << " OUT:" << cfg.exit_count.load()
                      << " RETRO:" << cfg.retrograde_count.load() << "\n";
        }
        if (g_enable_display &&
            (++display_result_counter % kDisplayEveryResults == 0)) {
        draw(raw, dets, tracks, cfg.mask_overlay);

        cv::Mat dp; cv::resize(raw, dp, cv::Size(cfg.display_w, cfg.display_h));
        // 画面: 绘制入口/出口矩形 (按外侧边法向 + depth 符号, 不依赖 gate_dir)
        {
            float sx = (float)cfg.display_w / raw.cols;
            float sy = (float)cfg.display_h / raw.rows;
            auto scalePt = [&](const cv::Point2f& p){ return cv::Point2f(p.x*sx,p.y*sy); };
            auto drawTripLine = [&](const Tripwire& tw, cv::Scalar col) {
                if (!tw.isValid()) return;
                cv::Point p1(scalePt(tw.p1)), p2(scalePt(tw.p2));
                cv::line(dp, p1, p2, col, 3);
            };
            auto drawZone = [&](const Tripwire& tw, float depth, cv::Scalar fill) {
                if (!tw.isValid() || std::abs(depth) < 1.0f) return;
                // v 沿外侧边法向 (右手系), 方向由 depth 符号决定
                // 这样 5 参 t1,x1,y1,x2,y2,depth 唯一确定矩形朝向, 不依赖 gate_dir
                cv::Point2f dir = tw.p2 - tw.p1;
                float L = std::sqrt(dir.x*dir.x + dir.y*dir.y);
                if (L < 1.0f) return;
                cv::Point2f n(dir.y/L, -dir.x/L);  // 右手系法向
                float s = std::abs(depth);
                cv::Point2f v = n * (depth / s);   // 符号决定朝向
                cv::Point2f o1=scalePt(tw.p1), o2=scalePt(tw.p2);
                cv::Point2f i1=scalePt(tw.p1+cv::Point2f(v.x*s,v.y*s));
                cv::Point2f i2=scalePt(tw.p2+cv::Point2f(v.x*s,v.y*s));
                std::vector<cv::Point> pts={{(int)o1.x,(int)o1.y},{(int)o2.x,(int)o2.y},
                                            {(int)i2.x,(int)i2.y},{(int)i1.x,(int)i1.y}};
                cv::Mat overlay; dp.copyTo(overlay);
                cv::fillConvexPoly(overlay, pts, fill);
                cv::addWeighted(overlay,0.4,dp,0.6,0,dp);
                cv::line(dp, {(int)o1.x,(int)o1.y},{(int)o2.x,(int)o2.y}, fill*0.8, 2);
                // 深度标签 (带符号, + 朝法向, - 朝反法向)
                char dz[16]; snprintf(dz, 16, "%+.0fpx", depth);
                cv::Point2f mid = (o1+o2+i1+i2)*0.25f;
                cv::putText(dp, dz, cv::Point((int)mid.x-22,(int)mid.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, fill*0.8, 1);
            };
            // 两个区域都按各自矩形深度 (带符号) 朝外侧边法向方向拉
            drawZone(cfg.entry_line, cfg.entry_zone_depth, cv::Scalar(0,160,0));
            drawZone(cfg.exit_line, cfg.exit_zone_depth, cv::Scalar(0,0,160));
            drawTripLine(cfg.entry_line, cv::Scalar(0,255,0));
            drawTripLine(cfg.exit_line, cv::Scalar(0,0,255));
        }
        // 轨迹调试: 脚底点 + 跨线闪烁
        {
            float sx = (float)cfg.display_w / raw.cols;
            float sy = (float)cfg.display_h / raw.rows;
            // 跨线闪烁: 遍历cs_map, 刚刚跨的线画亮色
            for (auto& kv : cs_map) {
                if (kv.second.flash_frames > 0) {
                    kv.second.flash_frames--;
                    const Tripwire& tw = (kv.second.last_cross_line == CL_ENTRY)
                        ? cfg.entry_line : cfg.exit_line;
                    cv::Point p1((int)(tw.p1.x*sx), (int)(tw.p1.y*sy));
                    cv::Point p2((int)(tw.p2.x*sx), (int)(tw.p2.y*sy));
                    cv::line(dp, p1, p2, cv::Scalar(0,255,255), 5, cv::LINE_AA);  // 亮青闪烁
                }
            }
            // 脚底点
            for (auto& t : tracks) {
                cv::Point2f foot(t.box.x + t.box.width/2.0f, t.box.y + t.box.height*0.92f);
                cv::circle(dp, cv::Point((int)(foot.x*sx), (int)(foot.y*sy)), 3, cv::Scalar(0,255,0), -1);
            }
        }
        // 第一行：黄色，基础信息
        std::string info1 = cfg.unique_id + " Zone:" + std::to_string(zn)
                          + " Tot:" + std::to_string((int)tracks.size());
        cv::putText(dp, info1, cv::Point(5, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,255), 1);
        // 第二行：红色大字，IN / RETRO 跨线计数
        std::string info2 = "IN:" + std::to_string(cfg.forward_count.load())
                          + "  OUT:" + std::to_string(cfg.exit_count.load())
                          + "  RETRO:" + std::to_string(cfg.retrograde_count.load());
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
                     << " IN:" << cfg.forward_count.load()
                     << " OUT:" << cfg.exit_count.load()
                     << " RETRO:" << cfg.retrograde_count.load()
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
                    int max_zone = 0;
                    for (auto& c : cameras) max_zone = std::max(max_zone, c.zone_count.load());
                    char mx[64]; snprintf(mx, 64, "Max Zone: %d", max_zone);
                    cv::putText(comb, mx, cv::Point(5, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
                    char fused[64]; snprintf(fused, 64, "Fused IN:%d OUT:%d RETRO:%d",
                                             g_fused_in.load(), g_fused_out.load(), g_fused_retro.load());
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
    std::cout << "Fused Final IN:" << g_fused_in.load()
              << " OUT:" << g_fused_out.load()
              << " RETRO:" << g_fused_retro.load() << std::endl;
    std::cout << "V4 stopped." << std::endl;
    return 0;
}

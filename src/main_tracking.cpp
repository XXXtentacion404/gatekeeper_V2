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

// 双向闸机 + 每摄像头独立计数：201 计 IN/RETRO，200 计 RETRO/OUT，
// 每个 ByteTrack ID 至多一次"进入类"+一次"离开类"事件，无需双摄融合去重
// （旧 reportFusedEvent / g_fused_* 已移除）。

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
    
    // 标记线区域（单向闸机：单标记线 + 出入标记区域）
    Tripwire entry_line;             // 进出标记线（config 里 t1）
    Tripwire exit_line;              // 保留字段（新逻辑不使用）
    float entry_zone_depth = 100.0f; // 入标记区域深度(px)：标记线向检测区内延伸
    float exit_zone_depth = 50.0f;   // 出标记区域深度(px)：标记线向检测区外延伸
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
          role(o.role),
          in_count(o.in_count.load()), out_count(o.out_count.load()),
          retro_count(o.retro_count.load()),
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
        cc.entry_zone_depth = c.entry_zone_depth;  // 入标记区域深度(带符号表入侧方向)
        cc.exit_zone_depth = c.exit_zone_depth;    // 出标记区域深度
        if (cc.exit_zone_depth < 1.0f) cc.exit_zone_depth = 50.0f;  // 出标记区域默认50px
        if (!cc.detection_mask.empty()) {
            std::vector<std::vector<cv::Point>> ct;
            cv::findContours(cc.detection_mask, ct, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cc.mask_overlay = cv::Mat(c.height, c.width, CV_8UC3, cv::Scalar(0,0,0));
            for (auto& x : ct) cv::polylines(cc.mask_overlay, x, true, cv::Scalar(0,0,255), 3);
        }
        std::string base = c.ip + "_Ch" + std::to_string(c.channel);
        cc.unique_id = base + "_" + std::to_string(++cnt[base]);
        // 相机角色：IP 末段 200=出口摄像头(只计OUT)，201=入口摄像头(只计IN)。
        {
            std::string tail = c.ip;
            auto dot = tail.find_last_of('.');
            if (dot != std::string::npos) tail = tail.substr(dot + 1);
            cc.role = (tail == "200") ? 1 : 0;  // 默认入口（含 201 及其他）
        }
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
    // 双向闸机计数（与 ByteTrack ID 绑定）。标记线 = cfg.entry_line(t1)。
    // 每个摄像头都做双向判定：外→内=进入类，内→外消失=离开类，角色决定标签。
    // 201(入口)：外→内=IN，内→外消失=RETRO；200(出口)：外→内=RETRO，内→外消失=OUT。
    GateIdCounter gate_counter(cfg.role == 1 ? GATE_ROLE_EXIT : GATE_ROLE_ENTRY,
                               cfg.entry_line,
                               cfg.entry_zone_depth, cfg.exit_zone_depth);
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

        // ==================== 双向闸机计数（与 ByteTrack ID 绑定）====================
        // 201(入口)：外→内=IN，内→外消失=RETRO；200(出口)：外→内=RETRO，内→外消失=OUT。
        // 遮挡期间 ByteTrack 保持同一 ID 预测输出，不会重复计数；只有 ID 真正消失才结算。
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
            std::cerr << "[GATE] ts=" << capture_ms << " " << cfg.unique_id
                      << " track=" << event.track_id << " event=" << name
                      << " foot=(" << event.foot.x << "," << event.foot.y << ")"
                      << " IN:" << cfg.in_count.load()
                      << " OUT:" << cfg.out_count.load()
                      << " RETRO:" << cfg.retro_count.load() << "\n";
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
            // 入标记区域(绿, 向检测区内) + 出标记区域(红, 向检测区外) + 标记线
            drawZone(cfg.entry_line, cfg.entry_zone_depth, cv::Scalar(0,160,0));
            drawZone(cfg.entry_line, -cfg.exit_zone_depth, cv::Scalar(0,0,160));
            drawTripLine(cfg.entry_line, cv::Scalar(0,255,0));
        }
        // 轨迹调试: 脚底点 + 计数闪烁
        {
            float sx = (float)cfg.display_w / raw.cols;
            float sy = (float)cfg.display_h / raw.rows;
            // 计数事件后闪烁标记线
            if (gate_flash > 0) {
                gate_flash--;
                cv::Point p1((int)(cfg.entry_line.p1.x*sx), (int)(cfg.entry_line.p1.y*sy));
                cv::Point p2((int)(cfg.entry_line.p2.x*sx), (int)(cfg.entry_line.p2.y*sy));
                cv::line(dp, p1, p2, cv::Scalar(0,255,255), 5, cv::LINE_AA);  // 亮青闪烁
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
        // 第二行：红色大字，IN / OUT / RETRO 计数
        std::string info2 = "IN:" + std::to_string(cfg.in_count.load())
                          + "  OUT:" + std::to_string(cfg.out_count.load())
                          + "  RETRO:" + std::to_string(cfg.retro_count.load());
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
                     << " IN:" << cfg.in_count.load()
                     << " OUT:" << cfg.out_count.load()
                     << " RETRO:" << cfg.retro_count.load()
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
                    int max_zone = 0, total_in = 0, total_out = 0, total_retro = 0;
                    for (auto& c : cameras) {
                        max_zone = std::max(max_zone, c.zone_count.load());
                        total_in += c.in_count.load();
                        total_out += c.out_count.load();
                        total_retro += c.retro_count.load();
                    }
                    char mx[64]; snprintf(mx, 64, "Max Zone: %d", max_zone);
                    cv::putText(comb, mx, cv::Point(5, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
                    char fused[64]; snprintf(fused, 64, "Total IN:%d OUT:%d RETRO:%d",
                                             total_in, total_out, total_retro);
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
    int total_in = 0, total_out = 0, total_retro = 0;
    for (auto& c : cameras) { total_in += c.in_count.load(); total_out += c.out_count.load(); total_retro += c.retro_count.load(); }
    std::cout << "Total Final IN:" << total_in
              << " OUT:" << total_out
              << " RETRO:" << total_retro << std::endl;
    std::cout << "V4 stopped." << std::endl;
    return 0;
}

/**
 * gatekeeper_V4 — 无头版本（无显示画面）
 * 仅做：YOLOv8检测 → ByteTrack跟踪 → 区域人数统计
 * 终端输出 + SQLite + 串口
 *
 * 用法: ./main_headless <model.rknn> <config.txt> [npu_threads_per_camera=2]
 */
#include <iostream>
#include <csignal>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <sqlite3.h>

#include "HCNetSDK.h"
#include "LinuxPlayM4.h"

#include "types.h"
#include "yolov8_custom.h"
#include "yolov8_thread_pool.h"
#include "bytetrack/tracker.h"
#include "mask_utils.h"
#include "comm.h"

// ==================== 全局 ====================
std::atomic<bool> g_running{true};
std::mutex g_hik_mutex;
std::string g_model_path;
int g_npu_threads_per_camera = 2;

// comm.cpp 依赖的全局变量
std::atomic<uint16_t> g_max_box_count(0);
std::atomic<bool> g_need_clear_queues{false};

// ==================== 摄像头配置 ====================
struct CameraConfig {
    std::string unique_id, ip, username, password;
    int channel;
    LONG userID = -1, realPlayHandle = -1;
    std::unique_ptr<Yolov8ThreadPool> yolov8_pool;
    std::unique_ptr<ByteTrackTracker> tracker;
    std::atomic<int> frame_id{0};
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

    std::deque<int> recent_counts;
    std::mutex count_mutex;
    int filtered_count{0};
    std::atomic<int> zone_count{0};
    std::atomic<int> mode_count{0};
    std::atomic<int> total_count{0};

    CameraConfig() = default;
    CameraConfig(const CameraConfig&) = delete;
    CameraConfig& operator=(const CameraConfig&) = delete;

    CameraConfig(CameraConfig&& o) noexcept
        : unique_id(std::move(o.unique_id)), ip(std::move(o.ip)),
          username(std::move(o.username)), password(std::move(o.password)),
          channel(o.channel), userID(o.userID), realPlayHandle(o.realPlayHandle),
          yolov8_pool(std::move(o.yolov8_pool)), tracker(std::move(o.tracker)),
          frame_id(o.frame_id.load()), last_capture_ms(o.last_capture_ms.load()), stop_flag(o.stop_flag.load()),
          send_db(o.send_db), last_minute(o.last_minute),
          g_nPort(o.g_nPort), g_BGRImage(std::move(o.g_BGRImage)),
          frame_counter(o.frame_counter.load()), last_stat_time(o.last_stat_time),
          fps(o.fps), detection_mask(std::move(o.detection_mask)),
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
            // 输出到全新 Mat 再 move 走：解码器会复用 pBuf，线程池需独立缓冲
            cv::Mat bgr;
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YV12);
            cfg->g_BGRImage = std::move(bgr);
            cfg->last_capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
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
        std::string base = c.ip + "_Ch" + std::to_string(c.channel);
        cc.unique_id = base + "_" + std::to_string(++cnt[base]);
        out.emplace_back(std::move(cc));
    }
    return out;
}

// ==================== 处理线程 ====================
void processCamera(CameraConfig& cfg) {
    cfg.send_db = getDB("results_v4.db");
    cfg.yolov8_pool = std::make_unique<Yolov8ThreadPool>();
    cfg.yolov8_pool->setNpuCores({0, 1});
    if (cfg.yolov8_pool->setUp(g_model_path, g_npu_threads_per_camera) != NN_SUCCESS) {
        std::cerr << "YOLOv8 fail: " << cfg.ip << std::endl; return;
    }
    cfg.tracker = std::make_unique<ByteTrackTracker>(0.25f, 0.5f, 0.5f, 60, 3);

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
    int last_processed_result_id = -1;
    while (!cfg.stop_flag && g_running) {
        { std::lock_guard<std::mutex> lk(cfg.g_frame_mutex);
          // move 交出所有权(零拷贝入队); move 后 g_BGRImage 变空, 同一帧不会被重复提交
          if (!cfg.g_BGRImage.empty()) cfg.yolov8_pool->submitTask(
              std::move(cfg.g_BGRImage), cfg.frame_id++, cfg.last_capture_ms.load()); }

        cv::Mat raw; int rid=0, rbc=0; int64_t capture_ms=0; std::vector<Detection> dets;
        if (cfg.yolov8_pool->tryGetLatestResultRaw(raw, rid, rbc, dets, &capture_ms) != NN_SUCCESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue;
        }
        if (rid <= last_processed_result_id) continue;
        last_processed_result_id = rid;
        std::vector<TrackedObject> tracks;
        if (!dets.empty()) {
            std::vector<Detection> people;
            for (auto& d : dets) if (d.className=="people") people.push_back(d);
            if (!people.empty()) tracks = cfg.tracker->update(people, raw, raw.cols, raw.rows);
        }
        for (auto& track : tracks) {
            track.result_frame_id = rid;
            track.capture_timestamp_ms = capture_ms;
        }

        // 区域人数：跟踪位置记忆状态机
        enum { S_OUT, S_ENTER, S_IN, S_LEAVE };
        struct TM { int s = S_OUT; int n = 0; cv::Point2f p; int out_n = 0; };
        static std::map<int, TM> mm;
        for (auto it = mm.begin(); it != mm.end(); ) {
            bool al = false;
            for (auto& t : tracks) if (t.track_id == it->first) { al = true; break; }
            if (!al && it->second.s <= S_ENTER) it = mm.erase(it);
            else if (!al && it->second.n++ > 60) it = mm.erase(it);
            else ++it;
        }
        int zn = 0;
        for (auto& t : tracks) {
            int tid = t.track_id;
            auto& m = mm[tid];
            cv::Point2f c(t.box.x+t.box.width/2.0f, t.box.y+t.box.height*0.92f);
            bool inZ = isInsideZone(t.box, cfg.detection_mask, 0.2f);
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
                    cv::Rect vb = t.box & cv::Rect(0,0,cfg.detection_mask.cols,cfg.detection_mask.rows);
                    float iou_fast = 0; if(vb.area()>0){cv::Mat r=cfg.detection_mask(vb); iou_fast=(float)cv::countNonZero(r)/vb.area();}
                    bool fast_exit = (m.out_n>=4 && iou_fast<0.01f);
                    if (fast_exit || m.n>=8) { m.s=S_OUT; m.n=0; }
                } else { m.s=S_IN; m.p=c; m.n=0; m.out_n=0; } break;
            }
            if (m.s==S_IN || m.s==S_LEAVE) zn++;
        }
        cfg.zone_count = zn;
        cfg.mode_count = cfg.calculateMode(zn);
        cfg.total_count = (int)tracks.size();

        // 更新全局最大人数（供串口查询）
        int max_zone = 0;
        for (auto& cam : cameras) max_zone = std::max(max_zone, cam.zone_count.load());
        g_max_box_count.store(max_zone);

        // 每30帧输出终端统计
        if (dc++ % 30 == 0) {
            double el = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-cfg.last_stat_time).count()/1000.0;
            if (el>0) { cfg.fps=30.0/el; cfg.last_stat_time=std::chrono::steady_clock::now(); }
            int max_zone = 0;
            for (auto& cam : cameras) max_zone = std::max(max_zone, cam.zone_count.load());
            std::cout << cfg.unique_id << " FPS:" << cfg.fps
                     << " Zone:" << zn << " Tot:" << tracks.size()
                     << " Max:" << max_zone << std::endl;
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
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <model.rknn> <config.txt> [threads=2]" << std::endl; return -1; }
    g_model_path = argv[1];
    if (argc>3) g_npu_threads_per_camera = std::max(1,std::min(atoi(argv[3]),4));

#ifdef ENABLE_SERIAL
    init_serial_comm("/dev/ttyS9");
#endif
    NET_DVR_Init(); NET_DVR_SetConnectTime(3000,3);
    cameras = loadCameras(argv[2]);
    if (cameras.empty()) { NET_DVR_Cleanup(); return -1; }
    std::cout << "V4 Headless " << cameras.size() << " cameras (" << g_npu_threads_per_camera << " threads each)" << std::endl;

    std::vector<std::thread> ths;
    for (auto& c : cameras) ths.emplace_back(processCamera, std::ref(c));

    // 无画面，仅等待信号退出
    while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (auto& t : ths) if (t.joinable()) t.join();
    NET_DVR_Cleanup();
    std::cout << "V4 stopped." << std::endl;
    return 0;
}

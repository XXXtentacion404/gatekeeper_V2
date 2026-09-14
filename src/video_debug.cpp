/**
 * gatekeeper_V4 - video debug program
 * YOLOv8 detection -> ByteTrack tracking -> zone count + virtual tripwire IN/RETRO detection
 * Usage: ./video_debug <model.rknn> <config.txt> <video.mp4> [threads=2] [-c 0|1]
 */
#include <iostream>
#include <csignal>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <thread>
#include <memory>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <opencv2/opencv.hpp>

#include "types.h"
#include "yolov8_custom.h"
#include "yolov8_thread_pool.h"
#include "bytetrack/tracker.h"
#include "tripwire_counter.h"
#include "mask_utils.h"


std::atomic<bool> g_running{true};
std::string g_model_path;
int g_npu_threads = 2;
std::atomic<uint16_t> g_max_box_count(0);
void signal_handler(int) { g_running = false; }


struct VideoConfig {
    std::string name;
    int cam_idx = 0;
    cv::VideoCapture cap;
    std::unique_ptr<Yolov8ThreadPool> pool;
    std::unique_ptr<ByteTrackTracker> tracker;
    cv::Mat detection_mask;
    cv::Mat mask_overlay;
    cv::Mat mask_overlay_disp;
    std::atomic<int> frame_id{0};
    int disp_w = 640, disp_h = 360;
    std::chrono::steady_clock::time_point last_stat;
    double fps = 0;
    int frame_counter = 0;
    int zone_count = 0;
    int mode_count = 0;
    int flash_frames = 0;


    Tripwire entry_line;
    Tripwire exit_line;
    float entry_zone_depth = 50.0f;
    float exit_zone_depth = 50.0f;
    int forward_count = 0;
    int exit_count = 0;
    int retrograde_count = 0;

    std::deque<int> recent_counts;

    std::mutex count_mutex;
    int calculateMode(int c) {
        std::lock_guard<std::mutex> lk(count_mutex);
        recent_counts.push_back(c);
        if (recent_counts.size() > 10) recent_counts.pop_front();
        std::unordered_map<int,int> freq;
        int m=0, mf=0;
        for (int x : recent_counts) { freq[x]++; if (freq[x] > mf) { mf=freq[x]; m=x; } }
        return m;
    }

    ~VideoConfig() { if (cap.isOpened()) cap.release(); }
};


void drawFrame(cv::Mat& img,
               const std::vector<TrackedObject>& tracks,
               const cv::Mat& mask_overlay,
               int zone_count, int total_tracks,
               int fwd_count, int out_count, int retro_count,
               const Tripwire& entry, const Tripwire& exit,
               const std::string& cam_id, double fps,
               float entry_zone_depth, float exit_zone_depth,
               int cam_idx, int flash_frames,
               float sx, float sy) {

    if (!mask_overlay.empty() && mask_overlay.size() == img.size()) {
        cv::Mat gray, bin;
        cv::cvtColor(mask_overlay, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, bin, 1, 255, cv::THRESH_BINARY);
        mask_overlay.copyTo(img, bin);
    }

    auto scalePt = [&](const cv::Point2f& p) {
        return cv::Point2f(p.x * sx, p.y * sy);
    };


    auto drawTripLine = [&](const Tripwire& tw, const cv::Scalar& color) {
        if (!tw.isValid()) return;
        cv::Point2f a = scalePt(tw.p1), b = scalePt(tw.p2);
        cv::line(img, a, b, color, 2);
    };

    auto drawZone = [&](const Tripwire& tw, float depth, const cv::Scalar& fill) {
        if (!tw.isValid() || std::abs(depth) < 1.0f) return;
        cv::Point2f dir = tw.p2 - tw.p1;
        float L = std::sqrt(dir.x*dir.x + dir.y*dir.y);
        if (L < 1.0f) return;
        float s = std::abs(depth);
        cv::Point2f n(dir.y/L, -dir.x/L);
        cv::Point2f v = n * (depth / s);
        cv::Point2f o1 = scalePt(tw.p1), o2 = scalePt(tw.p2);
        cv::Point2f i1 = scalePt(tw.p1 + cv::Point2f(v.x*s, v.y*s));
        cv::Point2f i2 = scalePt(tw.p2 + cv::Point2f(v.x*s, v.y*s));
        std::vector<cv::Point> pts = {{(int)o1.x,(int)o1.y},{(int)o2.x,(int)o2.y},
                                      {(int)i2.x,(int)i2.y},{(int)i1.x,(int)i1.y}};
        cv::Mat overlay; img.copyTo(overlay);
        cv::fillConvexPoly(overlay, pts, fill);
        cv::addWeighted(overlay, 0.4, img, 0.6, 0, img);
        cv::line(img, {(int)o1.x,(int)o1.y}, {(int)o2.x,(int)o2.y}, fill*0.8, 2);

        char dz[16]; snprintf(dz, 16, "%+.0fpx", depth);
        cv::Point2f mid = (o1 + o2 + i1 + i2) * 0.25f;
        cv::putText(img, dz, cv::Point((int)mid.x - 22, (int)mid.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, fill*0.8, 1);
    };


    drawZone(entry, entry_zone_depth, cv::Scalar(0, 160, 0));
    drawZone(exit, exit_zone_depth, cv::Scalar(0, 0, 160));
    drawTripLine(entry, cv::Scalar(0, 255, 0));
    drawTripLine(exit, cv::Scalar(0, 0, 255));


    if (flash_frames > 0 && entry.isValid() && exit.isValid()) {
        cv::Point2f e1 = scalePt(entry.p1), e2 = scalePt(entry.p2);
        cv::Point2f x1 = scalePt(exit.p1), x2 = scalePt(exit.p2);
        cv::line(img, e1, e2, cv::Scalar(0, 255, 255), 5, cv::LINE_AA);
        cv::line(img, x1, x2, cv::Scalar(0, 255, 255), 5, cv::LINE_AA);
    }

    // Draw tracked boxes.
    for (const auto& t : tracks) {
        cv::Rect sbox((int)(t.box.x * sx), (int)(t.box.y * sy),
                      (int)(t.box.width * sx), (int)(t.box.height * sy));
        sbox = sbox & cv::Rect(0, 0, img.cols, img.rows);
        if (sbox.area() <= 0) continue;
        cv::rectangle(img, sbox, t.color, 2);
        std::string label = "P" + std::to_string(t.track_id);
        int baseline;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.20, 2, &baseline);
        cv::rectangle(img, cv::Point(sbox.x, sbox.y - ts.height - 4),
                      cv::Point(sbox.x + ts.width, sbox.y), t.color, cv::FILLED);
        cv::putText(img, label, cv::Point(sbox.x, sbox.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 1.20, cv::Scalar(255, 255, 255), 2);

        // foot point
        cv::Point2f foot(sbox.x + sbox.width/2.0f, sbox.y + sbox.height*0.92f);
        cv::circle(img, cv::Point((int)foot.x, (int)foot.y), 2, cv::Scalar(0,255,0), -1);
    }


    std::string line1 = cam_id + " Zone:" + std::to_string(zone_count)
                      + " Tot:" + std::to_string(total_tracks);
    cv::putText(img, line1, cv::Point(5, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    std::string line2 = "IN:" + std::to_string(fwd_count)
                      + "  OUT:" + std::to_string(out_count)
                      + "  RETRO:" + std::to_string(retro_count);
    cv::putText(img, line2, cv::Point(5, 42),
                cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0, 0, 255), 2);

    char buf[32]; snprintf(buf, 32, "FPS:%.0f", fps);
    cv::putText(img, buf, cv::Point(5, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
}


void processVideo(VideoConfig& cfg, int orig_w, int orig_h, double vid_fps, bool save_video) {
    cfg.last_stat = std::chrono::steady_clock::now();

    enum { S_OUT, S_ENTER, S_IN, S_LEAVE };
    struct TM { int s = S_OUT; int n = 0; cv::Point2f p; int out_n = 0; };
    std::map<int, TM> zone_states;

    // (注: 区域/带符号距离判定已全部收拢到 TripwireCounter, 本地几何 lambda 已清除)
    TripwireCounter tripwire_counter(cfg.entry_line, cfg.exit_line,
                                     cfg.entry_zone_depth, cfg.exit_zone_depth,
                                     cfg.detection_mask);

    float sx = cfg.disp_w / (float)orig_w;
    float sy = cfg.disp_h / (float)orig_h;

    int writer_fps = (int)std::max(1.0, std::min(vid_fps, 30.0));
    cv::VideoWriter vw;
    if (save_video) {
        vw.open("output_v4.avi", cv::VideoWriter::fourcc('X','V','I','D'),
                writer_fps, cv::Size(cfg.disp_w, cfg.disp_h));
        if (!vw.isOpened()) std::cerr << "WARN: cannot write video" << std::endl;
    }

    cv::Mat rawImg;
    while (cfg.cap.read(rawImg) && g_running) {
        auto loop_t0 = std::chrono::steady_clock::now();
        if (rawImg.empty()) break;
        cfg.frame_counter++;

        cv::Mat frame;
        if (rawImg.cols != orig_w || rawImg.rows != orig_h)
            cv::resize(rawImg, frame, cv::Size(orig_w, orig_h));
        else
            frame = rawImg;

        int submitted_id = cfg.frame_id++;
        const int64_t capture_ms = static_cast<int64_t>(submitted_id);
        // clone: 无 resize 时 frame 是 rawImg 的浅别名, cap.read 下一帧会覆写
        // 其缓冲; submitTask 已按值 move, 必须交出独立缓冲
        cfg.pool->submitTask(frame.clone(), submitted_id, capture_ms);

        std::vector<Detection> detections;

        cv::Mat resultImg;
        int rawBoxCount = 0;
        while (g_running) {
            if (cfg.pool->tryGetResult(resultImg, submitted_id, rawBoxCount, detections) == NN_SUCCESS)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!g_running) break;

// ByteTrack tracking
        std::vector<Detection> people;
        for (const auto& d : detections) {
            if (d.className != "people") continue;
            people.push_back(d);
        }
        std::vector<TrackedObject> tracks =
            cfg.tracker->update(people, resultImg, resultImg.cols, resultImg.rows);
        for (auto& track : tracks) {
            track.result_frame_id = submitted_id;
            track.capture_timestamp_ms = capture_ms;
        }

        if ((int)tracks.size() > g_max_box_count) g_max_box_count = (uint16_t)tracks.size();


        for (auto it = zone_states.begin(); it != zone_states.end(); ) {
            bool alive = false;
            for (auto& t : tracks) if (t.track_id == it->first) { alive = true; break; }
            if (!alive && it->second.s <= S_ENTER) it = zone_states.erase(it);
            else if (!alive && it->second.n++ > 60) it = zone_states.erase(it);
            else ++it;
        }

        int zn = 0;
        for (const auto& t : tracks) {
            int tid = t.track_id;
            auto& m = zone_states[tid];
            cv::Point2f c(t.box.x + t.box.width/2.0f, t.box.y + t.box.height*0.92f);

            bool inZ = tripwire_counter.inZone(c);

            switch (m.s) {
            case S_OUT: case S_ENTER:
                if (inZ) { m.n++; if (m.n >= 2) { m.s = S_IN; m.p = c; m.n = 0; } }
                else { m.s = S_OUT; m.n = 0; } break;
            case S_IN:
                if (!inZ) {
                    if (cv::norm(c - m.p) > 80) { m.s = S_LEAVE; m.n = 1; m.out_n = 1; }
                } else { m.p = c; m.out_n = 0; } break;
            case S_LEAVE:
                if (!inZ) {
                    m.n++; m.out_n++;
                    // fast exit check: foot still outside
                    bool fast_exit = (m.out_n >= 4 && !tripwire_counter.inZone(c));
                    if (fast_exit || m.n >= 8) { m.s = S_OUT; m.n = 0; }
                } else { m.s = S_IN; m.p = c; m.n = 0; m.out_n = 0; } break;
            }
            if (m.s == S_IN || m.s == S_LEAVE) zn++;
        }
        cfg.zone_count = zn;
        cfg.mode_count = cfg.calculateMode(zn);

        // ==================== 检测区(区域状态机)计数 ====================
        // 区域模型: 0区域外 1入口区域 2检测区(入口线↔出口线走廊) 3出口区域
        // 触发: 脚底所在"区域"的跳变(4帧平滑后), 进入/离开检测区即计.
        //   IN  : 非出口侧 -> 进入检测区   (区域外/入口区域 侧进)
        //   RETRO: 出口区域侧 -> 进入检测区 (逆行进)
        //   OUT  : 离开检测区 -> 出口区域/区域外
        //   RETRO: 离开检测区 -> 入口区域   (逆行出)
        // 首帧仅建立区域(不计数); 若一出生已在检测区, 按所处半区推断方向补计, 解决
        // "首帧检测框已落在入口侧走廊导致 IN 漏计"的问题. 重臂/合并/4帧平滑/cooldown 保留.
        // (实现全部在 TripwireCounter::update 内, 原 if(false&&...) 死代码块已清除)

        auto counter_events = tripwire_counter.update(tracks);
        auto trace_events = tripwire_counter.takeDebugEvents();
        for (const auto& trace : trace_events) {
            std::cout << "[ZONE_TRACE] " << cfg.name
                      << " frame=" << cfg.frame_counter
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
                      << " foot=(" << trace.foot.x << "," << trace.foot.y << ")"
                      << std::endl;
        }
        for (const auto& event : counter_events) {
            if (event.type == TW_EVENT_IN) cfg.forward_count++;
            else if (event.type == TW_EVENT_OUT) cfg.exit_count++;
            else cfg.retrograde_count++;
            cfg.flash_frames = 15;
            std::cout << "[ZONE] " << cfg.name
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
                      << " IN:" << cfg.forward_count
                      << " OUT:" << cfg.exit_count
                      << " RETRO:" << cfg.retrograde_count << std::endl;
        }
        // ==================== 检测区计数 end ====================

        // resize to display size
        cv::Mat disp;
        cv::resize(resultImg, disp, cv::Size(cfg.disp_w, cfg.disp_h));

        // boxes are in orig (input) coords; drawFrame will scale them.
        if (cfg.flash_frames > 0) cfg.flash_frames--;
        drawFrame(disp, tracks, cfg.mask_overlay_disp,
                  zn, (int)tracks.size(),
                  cfg.forward_count, cfg.exit_count, cfg.retrograde_count,
                  cfg.entry_line, cfg.exit_line, cfg.name, cfg.fps,
                  cfg.entry_zone_depth, cfg.exit_zone_depth,
                  cfg.cam_idx, cfg.flash_frames, sx, sy);

        if (save_video && vw.isOpened()) vw.write(disp);


        if (cfg.frame_counter % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - cfg.last_stat).count() / 1000.0;
            if (elapsed > 0) { cfg.fps = 30.0 / elapsed; cfg.last_stat = now; }
            std::cout << cfg.name << " FPS:" << cfg.fps
                      << " Zone:" << zn << " Tot:" << tracks.size()
                      << " IN:" << cfg.forward_count
                      << " OUT:" << cfg.exit_count
                      << " RETRO:" << cfg.retrograde_count << std::endl;
        }

        cv::imshow(cfg.name, disp);
        int wk = cv::waitKey(1);
        if (wk == 27) { g_running = false; break; }

        // sync to video fps
        if (vid_fps > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - loop_t0).count();
            int frame_ms = (int)(1000.0 / vid_fps);

            int rem = frame_ms - (int)elapsed - 1;
            if (rem > 0) std::this_thread::sleep_for(std::chrono::milliseconds(rem));
        }
    }
    vw.release();
    std::cout << "[V4] Video saved: output_v4.avi" << std::endl;
}


int main(int argc, char** argv) {

    int cam_idx = 0;
    bool save_video = true;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-c" || a == "--cam") && i + 1 < argc) {
            cam_idx = std::stoi(argv[++i]);
            if (cam_idx < 0) cam_idx = 0;
            if (cam_idx > 1) cam_idx = 1;
            continue;
        }
        if (a == "-n" || a == "--nosave") { save_video = false; continue; }
        args.push_back(a);
    }

    if (args.size() < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.rknn> <config.txt> <video.mp4> [threads=2] [-c 0|1] [-n|--nosave]" << std::endl;
        return -1;
    }
    g_model_path = args[0];
    std::string cfgFile = args[1], vidPath = args[2];
    if (args.size() > 3) g_npu_threads = std::max(1, std::stoi(args[3]));

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto configs = parseCameraConfig(cfgFile);
    if (configs.empty()) { std::cerr << "No config!" << std::endl; return -1; }
    cam_idx = std::min(cam_idx, static_cast<int>(configs.size()) - 1);
    auto& cfg = configs[cam_idx];

    std::cout << "[V4] Config: " << cfg.ip << " " << cfg.width << "x" << cfg.height << std::endl;
    std::cout << "[V4] Video: " << vidPath << " threads:" << g_npu_threads << " cam:" << cam_idx << std::endl;

    VideoConfig vdc;
    vdc.name = cfg.ip + "_v4";
    vdc.cam_idx = cam_idx;
    vdc.cap.open(vidPath);
    if (!vdc.cap.isOpened()) { std::cerr << "Cannot open video" << std::endl; return -1; }

    double vidFps = vdc.cap.get(cv::CAP_PROP_FPS);
    int vw = (int)vdc.cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int vh = (int)vdc.cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "[V4] Source: " << vw << "x" << vh << " @ " << vidFps << "fps" << std::endl;


    vdc.detection_mask = createDetectionMask(cfg.width, cfg.height, cfg.detection_zones);
    vdc.entry_line = cfg.entry_line;
    // req4：出口区域已删除，CameraConfigEntry 不再有 exit_line/exit_zone_depth（配置文件
    // 也不再写 t2）。本工具仍用旧 tripwire_counter（其四区域模型必须有出口线才能计数），
    // 故这里不再赋值——出口线保持无效，video_debug 退化为"检测/追踪可视化"，不产生计数事件。
    // （main_tracking / main_headless 走新的 GateIdCounter，不受影响。）
    vdc.entry_zone_depth = cfg.entry_zone_depth;
    std::cerr << "[V4] 注意：出口线已随出口区域一并删除，"
                 "video_debug 不再输出 IN/OUT/RETRO 计数（仅可视化）。"
              << std::endl;
    if (!vdc.detection_mask.empty()) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(vdc.detection_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        vdc.mask_overlay = cv::Mat(cfg.height, cfg.width, CV_8UC3, cv::Scalar(0, 0, 0));
        for (const auto& c : contours)
            cv::polylines(vdc.mask_overlay, c, true, cv::Scalar(0, 0, 255), 3);
        cv::resize(vdc.mask_overlay, vdc.mask_overlay_disp, cv::Size(vdc.disp_w, vdc.disp_h));
    }

    // YOLOv8 + ByteTrack
    vdc.pool = std::make_unique<Yolov8ThreadPool>();
    if (vdc.pool->setUp(g_model_path, g_npu_threads) != NN_SUCCESS) {
        std::cerr << "YOLOv8 init failed!" << std::endl; return -1;
    }
    vdc.tracker = std::make_unique<ByteTrackTracker>(0.18f, 0.35f, 0.25f, 180, 2);

    std::cout << "[V4] Starting..." << std::endl;
    cv::namedWindow(vdc.name, cv::WINDOW_NORMAL);
    cv::resizeWindow(vdc.name, vdc.disp_w, vdc.disp_h);
    processVideo(vdc, cfg.width, cfg.height, vidFps, save_video);

    std::cout << "\n========== Final ==========" << std::endl;
    std::cout << "  Mode: " << vdc.mode_count << std::endl;
    std::cout << "  Zone: " << vdc.zone_count << std::endl;
    std::cout << "  Total tracks: " << vdc.tracker->getTrackCount() << std::endl;
    std::cout << "  Max detections: " << g_max_box_count << std::endl;
    std::cout << "  IN: " << vdc.forward_count << std::endl;
    std::cout << "  OUT: " << vdc.exit_count << std::endl;
    std::cout << "  RETRO: " << vdc.retrograde_count << std::endl;
    std::cout << "===========================" << std::endl;

    cv::destroyAllWindows();
    return 0;
}

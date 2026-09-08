#ifndef YOLOV8_THREAD_POOL_H
#define YOLOV8_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <map>
#include "types.h"
#include "yolov8_custom.h"

class Yolov8ThreadPool {
public:
    struct Stats {
        uint64_t submitted = 0;
        uint64_t processed = 0;
        uint64_t dropped_pending = 0;
        uint64_t dropped_results = 0;
        size_t queue_depth = 0;
    };

    Yolov8ThreadPool();
    ~Yolov8ThreadPool();

    nn_error_e setUp(const std::string &model_path, int num_threads);
    // 按值接收：调用方应传 std::move(img)（帧缓冲归线程池所有），
    // 内部不再 clone —— 省去每帧 6.2MB 深拷贝。
    nn_error_e submitTask(cv::Mat img, int id, int64_t capture_timestamp_ms = 0);
    // 按线程序号绑定 NPU 核（下标 i 的线程用 cores[i % cores.size()] 的核，
    // 元素含义见 RKEngine::setCoreIndex：0/1/2 = NPU0/1/2，-1 = 自动轮询）
    void setNpuCores(const std::vector<int> &cores);
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);
    nn_error_e getTargetImgResult(cv::Mat &img, int id);
    nn_error_e getTargetImgResultWithCount(cv::Mat &img, int id, int& box_count);
    nn_error_e getTargetImgResultWithDetections(cv::Mat& img, int id, int& box_count, std::vector<Detection>& detections);
    nn_error_e tryGetResult(cv::Mat& img, int id, int& box_count, std::vector<Detection>& detections,
                            int64_t* capture_timestamp_ms = nullptr);
    nn_error_e tryGetLatestResult(cv::Mat& img, int& out_id, int& box_count, std::vector<Detection>& detections,
                                  int64_t* capture_timestamp_ms = nullptr);
    
    // 新增：获取原始图像（未绘制）的最新结果
    nn_error_e tryGetLatestResultRaw(cv::Mat& img, int& out_id, int& box_count,
                                     std::vector<Detection>& detections,
                                     int64_t* capture_timestamp_ms = nullptr);

    // 新增：阻塞等待最新结果（用条件变量代替 sleep 忙等待，降低 CPU）。
    // timeout_ms 内无结果返回 NN_TIMEOUT；有结果等价于 tryGetLatestResultRaw。
    nn_error_e waitLatestResultRaw(cv::Mat& img, int& out_id, int& box_count,
                                   std::vector<Detection>& detections,
                                   int64_t* capture_timestamp_ms, int timeout_ms);

    void stopAll();
    Stats getStats() const;

private:
    void worker(int id);
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;
    std::vector<int> npu_cores_;             // 每线程绑定的 NPU 核（-1 = 自动）
    std::vector<std::thread> threads;
    struct FrameTask {
        int id;
        int64_t capture_timestamp_ms;
        cv::Mat image;
    };
    std::queue<FrameTask> tasks;
    mutable std::mutex mtx1;
    std::condition_variable cv_task;
    std::atomic<bool> stop;
    mutable std::mutex mtx2;
    std::condition_variable cv_result;   // 结果就绪通知（worker 插入后 notify_all）
    std::map<int, std::vector<Detection>> results;
    std::map<int, int64_t> capture_timestamps_ms_;
    std::map<int, cv::Mat> img_results;      // 保存原始图像（未绘制）
    std::atomic<int> submitted_frames{0};
    std::atomic<int> processed_frames{0};
    int last_delivered_id_ = -1;
    std::atomic<uint64_t> dropped_pending_frames{0};
    std::atomic<uint64_t> dropped_result_frames{0};
};

#endif

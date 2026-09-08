#include "yolov8_thread_pool.h"
#include <chrono>
#include <algorithm>

Yolov8ThreadPool::Yolov8ThreadPool() { stop = false; }

Yolov8ThreadPool::~Yolov8ThreadPool() {
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }
}

nn_error_e Yolov8ThreadPool::setUp(const std::string &model_path, int num_threads) {
    const int actual_threads = npu_cores_.empty()
        ? num_threads
        : std::max(1, std::min(num_threads, (int)npu_cores_.size()));
    if (actual_threads != num_threads)
        NN_LOG_WARN("NPU core binding limits worker threads from %d to %d",
                    num_threads, actual_threads);
    for (int i = 0; i < actual_threads; ++i) {
        std::shared_ptr<Yolov8Custom> det = std::make_shared<Yolov8Custom>();
        // 每线程实例绑定独立 NPU 核（-1 = 内部自动轮询）
        int core = npu_cores_.empty() ? -1
                                      : npu_cores_[i % npu_cores_.size()];
        if (det->LoadModel(model_path.c_str(), core) != NN_SUCCESS)
            return NN_LOAD_MODEL_FAIL;
        Yolov8_instances.push_back(det);
    }
    for (int i = 0; i < actual_threads; ++i)
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    return NN_SUCCESS;
}

void Yolov8ThreadPool::setNpuCores(const std::vector<int> &cores) {
    // 须在 setUp 之前调用；运行期修改不影响已加载实例
    npu_cores_ = cores;
}

void Yolov8ThreadPool::worker(int id) {
    while (!stop) {
        FrameTask task;
        auto instance = Yolov8_instances[(size_t)id];
        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&]{ return !tasks.empty() || stop; });
            if (stop) return;
            task = tasks.front();
            tasks.pop();
        }
        std::vector<Detection> detections;
        instance->Run(task.image, detections);
        processed_frames++;
        {
            std::lock_guard<std::mutex> lock(mtx2);
            if (task.id <= last_delivered_id_) {
                dropped_result_frames++;
            } else {
                results.insert({task.id, detections});
                img_results.insert({task.id, std::move(task.image)});
                capture_timestamps_ms_[task.id] = task.capture_timestamp_ms;
            }
            cv_result.notify_all();  // 结果就绪，唤醒阻塞等待的消费线程
        }
    }
}

nn_error_e Yolov8ThreadPool::submitTask(cv::Mat img, int id,
                                        int64_t capture_timestamp_ms) {
    if (img.empty()) return NN_TIMEOUT;
    // 按值接收 + move 入队：帧缓冲所有权转移给线程池，不再 clone。
    // 前提：调用方提交的是独立缓冲（main_tracking 用 std::move(g_BGRImage)，
    // video_debug 用 frame.clone()），解码器不会覆写线程池持有的数据。
    submitted_frames++;
    {
        std::lock_guard<std::mutex> lock(mtx1);
        dropped_pending_frames.fetch_add(tasks.size());
        std::queue<FrameTask> empty;
        tasks.swap(empty);
        tasks.push({id, capture_timestamp_ms, std::move(img)});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    while (results.find(id) == results.end())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    results.erase(id);
    capture_timestamps_ms_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 10) return NN_TIMEOUT;
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    img_results.erase(id);
    capture_timestamps_ms_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetImgResultWithCount(cv::Mat &img, int id, int& box_count) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end() || results.find(id) == results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 10) return NN_TIMEOUT;
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    box_count = (int)results[id].size();
    img_results.erase(id);
    results.erase(id);
    capture_timestamps_ms_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetImgResultWithDetections(cv::Mat& img, int id, int& box_count, std::vector<Detection>& detections) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end() || results.find(id) == results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 10) return NN_TIMEOUT;
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    detections = results[id];
    box_count = (int)detections.size();
    img_results.erase(id);
    results.erase(id);
    capture_timestamps_ms_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::tryGetResult(cv::Mat& img, int id, int& box_count,
                                           std::vector<Detection>& detections,
                                           int64_t* capture_timestamp_ms) {
    std::lock_guard<std::mutex> lock(mtx2);
    auto it = img_results.find(id);
    if (it == img_results.end() || results.find(id) == results.end())
        return NN_TIMEOUT;
    img = it->second;
    detections = results[id];
    if (capture_timestamp_ms) *capture_timestamp_ms = capture_timestamps_ms_[id];
    box_count = (int)detections.size();
    img_results.erase(id);
    results.erase(id);
    capture_timestamps_ms_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::tryGetLatestResult(cv::Mat& img, int& out_id, int& box_count,
                                                 std::vector<Detection>& detections,
                                                 int64_t* capture_timestamp_ms) {
    std::lock_guard<std::mutex> lock(mtx2);
    int best_id = -1;
    for (const auto& [fid, _] : img_results) {
        if (fid > last_delivered_id_ && fid > best_id &&
            results.find(fid) != results.end())
            best_id = fid;
    }
    if (best_id < 0) {
        for (auto it = img_results.begin(); it != img_results.end();) {
            if (it->first <= last_delivered_id_) {
                results.erase(it->first);
                capture_timestamps_ms_.erase(it->first);
                it = img_results.erase(it);
                dropped_result_frames++;
            } else {
                ++it;
            }
        }
        return NN_TIMEOUT;
    }
    
    out_id = best_id;
    img = img_results[best_id];
    detections = results[best_id];
    if (capture_timestamp_ms) *capture_timestamp_ms = capture_timestamps_ms_[best_id];
    box_count = (int)detections.size();
    last_delivered_id_ = best_id;
    
    for (auto it = img_results.begin(); it != img_results.end(); ) {
        if (it->first != best_id) {
            results.erase(it->first);
            capture_timestamps_ms_.erase(it->first);
            it = img_results.erase(it);
            dropped_result_frames++;
        } else {
            ++it;
        }
    }
    img_results.erase(best_id);
    results.erase(best_id);
    capture_timestamps_ms_.erase(best_id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::tryGetLatestResultRaw(cv::Mat& img, int& out_id, int& box_count,
                                                    std::vector<Detection>& detections,
                                                    int64_t* capture_timestamp_ms) {
    std::lock_guard<std::mutex> lock(mtx2);
    int best_id = -1;
    for (const auto& [fid, _] : img_results) {
        if (fid > last_delivered_id_ && fid > best_id &&
            results.find(fid) != results.end())
            best_id = fid;
    }
    if (best_id < 0) {
        for (auto it = img_results.begin(); it != img_results.end();) {
            if (it->first <= last_delivered_id_) {
                results.erase(it->first);
                capture_timestamps_ms_.erase(it->first);
                it = img_results.erase(it);
                dropped_result_frames++;
            } else {
                ++it;
            }
        }
        return NN_TIMEOUT;
    }

    out_id = best_id;
    img = img_results[best_id];
    detections = results[best_id];
    if (capture_timestamp_ms) *capture_timestamp_ms = capture_timestamps_ms_[best_id];
    box_count = (int)detections.size();
    last_delivered_id_ = best_id;

    for (auto it = img_results.begin(); it != img_results.end(); ) {
        if (it->first != best_id) {
            results.erase(it->first);
            capture_timestamps_ms_.erase(it->first);
            it = img_results.erase(it);
            dropped_result_frames++;
        } else {
            ++it;
        }
    }
    results.erase(best_id);
    capture_timestamps_ms_.erase(best_id);
    img_results.erase(best_id);
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::waitLatestResultRaw(cv::Mat& img, int& out_id, int& box_count,
                                                  std::vector<Detection>& detections,
                                                  int64_t* capture_timestamp_ms, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mtx2);
    // 阻塞等待：直到出现可交付的最新结果，或超时。用条件变量避免 sleep 忙等待。
    cv_result.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
        for (const auto& [fid, _] : img_results) {
            if (fid > last_delivered_id_ && results.find(fid) != results.end())
                return true;
        }
        return false;
    });

    int best_id = -1;
    for (const auto& [fid, _] : img_results) {
        if (fid > last_delivered_id_ && fid > best_id &&
            results.find(fid) != results.end())
            best_id = fid;
    }
    if (best_id < 0) {
        for (auto it = img_results.begin(); it != img_results.end();) {
            if (it->first <= last_delivered_id_) {
                results.erase(it->first);
                capture_timestamps_ms_.erase(it->first);
                it = img_results.erase(it);
                dropped_result_frames++;
            } else {
                ++it;
            }
        }
        return NN_TIMEOUT;
    }

    out_id = best_id;
    img = img_results[best_id];
    detections = results[best_id];
    if (capture_timestamp_ms) *capture_timestamp_ms = capture_timestamps_ms_[best_id];
    box_count = (int)detections.size();
    last_delivered_id_ = best_id;

    for (auto it = img_results.begin(); it != img_results.end(); ) {
        if (it->first != best_id) {
            results.erase(it->first);
            capture_timestamps_ms_.erase(it->first);
            it = img_results.erase(it);
            dropped_result_frames++;
        } else {
            ++it;
        }
    }
    results.erase(best_id);
    capture_timestamps_ms_.erase(best_id);
    img_results.erase(best_id);
    return NN_SUCCESS;
}

void Yolov8ThreadPool::stopAll() {
    stop = true;
    cv_task.notify_all();
}

Yolov8ThreadPool::Stats Yolov8ThreadPool::getStats() const {
    Stats stats;
    stats.submitted = submitted_frames.load();
    stats.processed = processed_frames.load();
    stats.dropped_pending = dropped_pending_frames.load();
    stats.dropped_results = dropped_result_frames.load();
    std::lock_guard<std::mutex> lock(mtx1);
    stats.queue_depth = tasks.size();
    return stats;
}

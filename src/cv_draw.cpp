#include "cv_draw.h"

void DrawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& d : detections) {
        cv::rectangle(img, d.box, d.color, 2);
        std::string label = d.className + " " + std::to_string(d.confidence).substr(0, 4);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        cv::rectangle(img, cv::Point(d.box.x, d.box.y - ts.height - 5),
                      cv::Point(d.box.x + ts.width, d.box.y), d.color, cv::FILLED);
        cv::putText(img, label, cv::Point(d.box.x, d.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
}

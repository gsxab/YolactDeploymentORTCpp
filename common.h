#ifndef ONNXNETTEST_COMMON_H
#define ONNXNETTEST_COMMON_H

#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

// fixed
constexpr int width = 640, height = 480;
extern const cv::Vec3f MEANS, STD;
constexpr int NET_PROP = 57744, CLASS_COUNT = 80;

// coloring
extern const cv::Vec3f COLORS[];
extern const int COLORS_NUM;

struct Detection {
  struct Box {
    float x, y, x2, y2;
  } box;
  int label;
  float confidence;
  cv::Mat mask;
};

struct RawDetectionBox {
  Detection::Box box;
  float conf[CLASS_COUNT];
  cv::Mat mask;
};

inline std::ostream& operator<<(std::ostream& os, Detection& rdb) {
  os << '[' << rdb.box.x << ',' << rdb.box.y << '~' << rdb.box.x2 << ',' << rdb.box.y2 << ']' << '\n';
  os << rdb.label << ' ' << std::fixed << std::setw(5) << rdb.confidence << '\n';
  os << rdb.mask.t() << '\n';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, RawDetectionBox& rdb) {
  os << '[' << rdb.box.x << ',' << rdb.box.y << '~' << rdb.box.x2 << ',' << rdb.box.y2 << ']' << '\n';
  for (float f : rdb.conf) os << std::fixed << std::setw(5) << f << ' ';
  os << '\n';
  os << rdb.mask.t() << '\n';
  return os;
}

inline float iou(const Detection::Box& b1, const Detection::Box& b2) {
  float w = std::min(b1.x2, b2.x2) - std::max(b1.x, b2.x);
  if (w <= 0) return 0.0f;
  float h = std::min(b1.y2, b2.y2) - std::max(b1.y, b2.y);
  if (h <= 0) return 0.0f;
  float i = w * h;
  float u = (b1.x2 - b1.x) * (b1.y2 - b1.y) + (b2.x2 - b2.x) * (b2.y2 - b2.y) - i;
  return i / u;
}

inline std::pair<int, int> sanitize_coordinates_range(float x1, float x2, int padding, int max) {
  if (x1 < x2) {
    return {
        std::max((int) std::ceil(x1) - padding, 0),
        std::min((int) std::ceil(x2) + padding, max)
    };
  } else {
    return {
        std::max((int) std::ceil(x2) - padding, 0),
        std::min((int) std::ceil(x1) + padding, max)
    };
  }
}

cv::Mat decode(const cv::Mat& loc, const cv::Mat& prior);
std::vector<RawDetectionBox> detect(const cv::Mat& conf, const cv::Mat& boxes, const cv::Mat& mask);
std::vector<Detection> traditional_nms(const std::vector<RawDetectionBox>& detections, const cv::Mat& maskMat);
void postprocess(std::vector<Detection>& filtered_det, const cv::Mat& proto);
cv::Mat draw_masks(const cv::Mat& img, const std::vector<Detection>& detections);

#endif //ONNXNETTEST_COMMON_H

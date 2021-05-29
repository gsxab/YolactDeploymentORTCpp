#include "common.h"

extern const cv::Vec3f MEANS = {103.94, 116.78, 123.68}, STD = {57.38, 57.12, 58.40};

extern const cv::Vec3f COLORS[]{
    cv::Vec3f{54, 67, 244} / 255.f,
    cv::Vec3f{99, 30, 233} / 255.f,
    cv::Vec3f{176, 39, 156} / 255.f,
    cv::Vec3f{183, 58, 103} / 255.f,
    cv::Vec3f{181, 81, 63} / 255.f,
    cv::Vec3f{243, 150, 33} / 255.f,
    cv::Vec3f{244, 169, 3} / 255.f,
    cv::Vec3f{212, 188, 0} / 255.f,
    cv::Vec3f{136, 150, 0} / 255.f,
    cv::Vec3f{80, 175, 76} / 255.f,
    cv::Vec3f{74, 195, 139} / 255.f,
    cv::Vec3f{57, 220, 205} / 255.f,
    cv::Vec3f{59, 235, 255} / 255.f,
    cv::Vec3f{7, 193, 255} / 255.f,
    cv::Vec3f{0, 152, 255} / 255.f,
    cv::Vec3f{34, 87, 255} / 255.f,
    cv::Vec3f{72, 85, 121} / 255.f,
    cv::Vec3f{158, 158, 158} / 255.f,
    cv::Vec3f{139, 125, 96} / 255.f,
};
extern const int COLORS_NUM = sizeof(COLORS) / sizeof(COLORS[0]);

// parameters
constexpr float conf_thresh = 0.05f, mns_thresh = 0.5f, score_thresh = 0.15f, mask_alpha = 0.45f;
constexpr int top_k = 15, max_num_detections = 100;

cv::Mat decode(const cv::Mat& loc, const cv::Mat& prior) {
  const float varianceXY = 0.1, varianceWH = 0.2;
  cv::Mat locXY = loc.colRange(0, 2), locWH = loc.colRange(2, 4);
  cv::Mat priorXY = prior.colRange(0, 2), priorWH = prior.colRange(2, 4);

  cv::Mat boxXY = priorXY + locXY.mul(varianceXY).mul(priorWH);
  cv::Mat tmp;
  cv::exp(locWH * varianceWH, tmp);
  cv::Mat boxWH_half = cv::abs(priorWH.mul(tmp) / 2);

  cv::Mat boxNW = boxXY - boxWH_half, boxSE = boxXY + boxWH_half;
  cv::Mat ret;
  cv::hconcat(boxNW, boxSE, ret);

  return ret;
}

std::vector<RawDetectionBox> detect(const cv::Mat& conf, const cv::Mat& boxes, const cv::Mat& mask) {
  int cand = conf.rows;
  assert(conf.rows == cand && conf.cols == CLASS_COUNT + 1);
  assert(boxes.rows == cand && boxes.cols == 4);
  assert(mask.rows == cand && mask.cols == 32);
  cv::Mat significant_score = (conf.colRange(1, CLASS_COUNT + 1) > conf_thresh);

  std::vector<RawDetectionBox> detections;
  detections.reserve(cand);
  for (int i = 0; i < cand; ++i) {
    cv::Mat row = significant_score.row(i);
    if (std::find(row.begin<uint8_t>(), row.end<uint8_t>(), 255) == row.end<uint8_t>()) continue;

    detections.push_back(RawDetectionBox{
        {
            boxes.at<float>(i, 0) * 138.f,
            boxes.at<float>(i, 1) * 138.f,
            boxes.at<float>(i, 2) * 138.f,
            boxes.at<float>(i, 3) * 138.f
        }, {}, cv::Mat(mask.row(i).t())});
    for (int j = 0; j < CLASS_COUNT; ++j) {
      detections.back().conf[j] = conf.at<float>(i, j + 1);
    }
//    std::cout << detections.back() << std::endl;
  }
  detections.shrink_to_fit();

  return detections;
}

std::vector<Detection> traditional_nms(const std::vector<RawDetectionBox>& detections, const cv::Mat& maskMat) {
  std::vector<Detection> filtered;
  filtered.reserve(detections.size() / 2);
  for (int classIndex = 0; classIndex < CLASS_COUNT; ++classIndex) {
    std::vector<bool> suppressed_within_class(detections.size(), false);
    std::vector<int> indices(detections.size());

    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [classIndex, &detections](int i1, int i2) {
      return detections[i1].conf[classIndex] > detections[i2].conf[classIndex];
    });
    auto first_less = std::upper_bound(indices.begin(), indices.end(), conf_thresh,
                                       [classIndex, &detections](float thresh, int i) {
                                         return detections[i].conf[classIndex] < thresh;
                                       });
//    std::cout << conf_thresh << ' ' << first_less - indices.begin() << '/' << indices.size() << std::endl;

    if (first_less == indices.begin()) continue;
    for (auto it = indices.begin(); it != first_less; ++it) {
      int ord_with_higher_conf = *it;
      if (suppressed_within_class[ord_with_higher_conf]) continue;
      const auto& det_with_higher_conf = detections[ord_with_higher_conf];
      const auto& box_with_higher_conf = det_with_higher_conf.box;

      filtered.push_back(Detection{box_with_higher_conf, classIndex + 1,
                                   det_with_higher_conf.conf[classIndex], det_with_higher_conf.mask});

      for (auto it2 = std::next(it); it2 != first_less; ++it2) {
        int ord_with_lower_conf = *it2;
        if (suppressed_within_class[ord_with_lower_conf]) continue;

        const auto& box_with_lower_conf = detections[ord_with_lower_conf].box;
        if (iou(box_with_higher_conf, box_with_lower_conf) > mns_thresh) {
          suppressed_within_class[ord_with_lower_conf] = true;
        }
      }
    }
  }

  std::sort(filtered.begin(), filtered.end(), [](const Detection& d1, const Detection& d2) {
    return d1.confidence > d2.confidence;
  });
  if (filtered.size() > max_num_detections) {
    filtered.erase(filtered.begin() + max_num_detections, filtered.end());
  }
  return filtered;
}

void postprocess(std::vector<Detection>& filtered_det, const cv::Mat& proto) {
  for (auto it = filtered_det.begin(); it != filtered_det.end(); ) {
    auto& det = *it;
    cv::Mat tmp;
    cv::exp(-proto * det.mask, tmp);
    cv::Mat sigmoid_mask = 1.f / (1.f + tmp);

    cv::Mat cropped_mask;
    {
      // crop
      int x, x2, y, y2;
      std::tie(x, x2) = sanitize_coordinates_range(det.box.x, det.box.x2, 1, 138);
      std::tie(y, y2) = sanitize_coordinates_range(det.box.y, det.box.y2, 1, 138);
      if (x2 <= 0 || x >= 138 || y2 <= 0 || y >= 138) {
        it = filtered_det.erase(it);
        continue;
      }
      cv::Mat box_mask = cv::Mat::zeros(138, 138, CV_32F);
      box_mask.colRange(x, x2).rowRange(y, y2) = 1.f;
      cropped_mask = box_mask.mul(sigmoid_mask.reshape(0, 138));
    }

    cv::Mat threshed_mask;
    {
      // interp
      cv::Mat tmp;
      cv::resize(cropped_mask, tmp, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
      threshed_mask = tmp > 0.5f;
    }

    det.mask = threshed_mask;
    std::tie(det.box.x, det.box.x2) = sanitize_coordinates_range(det.box.x / 138 * width, det.box.x2 / 138 * width, 0, width);
    std::tie(det.box.y, det.box.y2) = sanitize_coordinates_range(det.box.y / 138 * height, det.box.y2 / 138 * height, 0, height);

    ++it;
  }
}

cv::Mat draw_masks(const cv::Mat& img, const std::vector<Detection>& detections) {
  // filter masks to show
  int num_dets_to_consider = std::min(top_k, (int) detections.size());
  for (int i = 0; i < num_dets_to_consider; ++i) {
    if (detections[i].confidence < score_thresh) {
      num_dets_to_consider = i;
      break;
    }
  }

  if (num_dets_to_consider == 0) return img;
  std::vector<Detection> detections_to_show(detections.begin(), std::next(detections.begin(), num_dets_to_consider));

  // show
  cv::Mat masked_image;
  img.copyTo(masked_image);
  for (const auto& det: detections_to_show) {
    auto& color = COLORS[(det.label - 1) * 5 % COLORS_NUM];
    cv::Mat colored_mask = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    colored_mask.setTo(color, det.mask);
//    cv::imshow("masked image (incrementally)", masked_image);
//    cv::imshow("color mask", colored_mask);
//    cv::waitKey();
    cv::Mat new_masked = (masked_image + mask_alpha * (colored_mask - masked_image));
    new_masked.copyTo(masked_image, det.mask);
  }

  for (const auto& det: detections_to_show) {
    auto& color = COLORS[det.label * 5 % COLORS_NUM];
    cv::rectangle(masked_image,
                  {static_cast<int>(std::floor(det.box.x)), static_cast<int>(std::ceil(det.box.y))},
                  {static_cast<int>(std::floor(det.box.x2)), static_cast<int>(std::ceil(det.box.y2))},
                  color);
  }

  return masked_image;
}


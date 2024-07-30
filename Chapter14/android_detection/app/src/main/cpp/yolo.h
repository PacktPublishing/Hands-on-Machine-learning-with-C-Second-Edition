#pragma  once

#include <torch/script.h>
#include <torch/csrc/jit/mobile/import.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

#include <android/asset_manager.h>

struct YOLOResult {
    int class_index;
    std::string class_name;
    float score;
    cv::Rect rect;
};

class YOLO {
public:
    explicit YOLO(AAssetManager *asset_manager);

    ~YOLO() = default;

    YOLO(const YOLO &) = delete;

    YOLO &operator=(const YOLO &) = delete;

    YOLO(YOLO &&) = delete;

    YOLO &operator==(YOLO &&) = delete;

    std::vector<YOLOResult> detect(const cv::Mat &image);

private:
    void load_classes(std::istream &stream);

    void output2results(const torch::Tensor &output,
                        float img_scale_x,
                        float img_scale_y);

    std::vector<YOLOResult> non_max_suppression();

private:
    using Classes = std::map<size_t, std::string>;
    Classes classes_;
    torch::jit::mobile::Module model_;
    cv::Mat rgb_img_;
    std::vector<YOLOResult> results_;
};



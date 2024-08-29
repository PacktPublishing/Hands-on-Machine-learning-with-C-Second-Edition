#include "yolo.h"

#include "log.h"

#include <android/asset_manager_jni.h>

namespace {
    constexpr float threshold = 0.80f; // score above which a detection is generate
    constexpr int nms_limit = 5;

    class ReadAdapter : public caffe2::serialize::ReadAdapterInterface {
    public:
        explicit ReadAdapter(const std::vector<char> &buf) : buf_(&buf) {}

        [[nodiscard]] size_t size() const override {
            return buf_->size();
        }

        size_t read(uint64_t pos, void *buf, size_t n, [[maybe_unused]]const char *what)
        const override {
            std::copy_n(buf_->begin() + static_cast<ptrdiff_t>(pos), n,
                        reinterpret_cast<char *>(buf));
            return n;
        }

    private:
        const std::vector<char> *buf_;
    };

    std::vector<char> read_asset(AAssetManager *asset_manager, const std::string &name) {
        std::vector<char> buf;
        AAsset *asset = AAssetManager_open(asset_manager, name.c_str(), AASSET_MODE_UNKNOWN);
        if (asset != nullptr) {
            LOGI("Open asset %s OK", name.c_str());
            off_t buf_size = AAsset_getLength(asset);
            buf.resize(buf_size + 1, 0);
            auto num_read = AAsset_read(asset, buf.data(), buf_size);
            LOGI("Read asset %s OK", name.c_str());

            if (num_read == 0)
                buf.clear();
            AAsset_close(asset);
            LOGI("Close asset %s OK", name.c_str());
        }
        return buf;
    }

    template<typename CharT, typename TraitsT = std::char_traits<CharT> >
    struct VectorStreamBuf : public std::basic_streambuf<CharT, TraitsT> {
        explicit VectorStreamBuf(std::vector<CharT> &vec) {
            this->setg(vec.data(), vec.data(), vec.data() + vec.size());
        }
    };

    torch::Tensor mat2tensor(const cv::Mat &image) {
        ASSERT(image.channels() == 3, "Invalid image format");
        // The channel dimension is the last dimension in OpenCV
        torch::Tensor tensor_image = torch::from_blob(image.data,
                                                      {1, image.rows, image.cols, image.channels()},
                                                      at::kByte);
        // make float and normalize
        tensor_image = tensor_image.to(at::kFloat) / 255.;

        // Transpose the image for [channels, rows, columns] format of pytorch tensor
        tensor_image = torch::transpose(tensor_image, 1, 2);
        tensor_image = torch::transpose(tensor_image, 1, 3);
        return tensor_image;
    }

    float IOU(const cv::Rect &a, const cv::Rect &b) {
        if (a.empty() <= 0.0) return 0.0f;
        if (b.empty() <= 0.0) return 0.0f;

        auto min_x = std::max(a.x, b.x);
        auto min_y = std::max(a.y, b.y);
        auto max_x = std::min(a.x + a.width, b.x + b.width);
        auto max_y = std::min(a.y + a.height, b.y + b.height);
        auto area = std::max(max_y - min_y, 0) * std::max(max_x - min_x, 0);
        return static_cast<float>(area) / static_cast<float>(a.area() + b.area() - area);
    }
}

YOLO::YOLO(AAssetManager *asset_manager) {
    const std::string model_file_name = "yolov5s.torchscript";
    auto model_buf = read_asset(asset_manager, model_file_name);
    model_ = torch::jit::_load_for_mobile(std::make_unique<ReadAdapter>(model_buf));

    const std::string classes_file_name = "classes.txt";
    auto classes_buf = read_asset(asset_manager, classes_file_name);
    VectorStreamBuf<char> stream_buf(classes_buf);
    std::istream is(&stream_buf);
    load_classes(is);
}

void YOLO::load_classes(std::istream &stream) {
    LOGI("Init classes start OK");
    classes_.clear();
    if (stream) {
        std::string line;
        std::string id;
        std::string label;
        size_t idx = 0;
        while (std::getline(stream, line)) {
            auto pos = line.find_first_of(':');
            id = line.substr(0, pos);
            label = line.substr(pos + 1);
            classes_.insert({idx, label});
            ++idx;
        }
    }
    LOGI("Init classes finish OK");
}

std::vector<YOLOResult> YOLO::detect(const cv::Mat &image) {
    // yolov5 input size
    constexpr int input_width = 640;
    constexpr int input_height = 640;

    cv::cvtColor(image, rgb_img_, cv::COLOR_RGBA2RGB);
    cv::resize(rgb_img_, rgb_img_, cv::Size(input_width, input_height));

    auto img_scale_x = static_cast<float>( image.cols) / input_width;
    auto img_scale_y = static_cast<float>( image.rows) / input_height;

    auto input_tensor = mat2tensor(rgb_img_);
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input_tensor);

    auto output = model_.forward(inputs).toTuple()->elements()[0].toTensor().squeeze(0);
    output2results(output, img_scale_x, img_scale_y);
    return non_max_suppression();
}

void YOLO::output2results(const torch::Tensor &output,
                          float img_scale_x,
                          float img_scale_y) {
    auto outputs = output.accessor<float, 2>();
    auto output_row = output.size(0);
    auto output_column = output.size(1);
    results_.clear();
    for (int64_t i = 0; i < output_row; i++) {
        auto score = outputs[i][4];
        if (score > threshold) {
            float cx = outputs[i][0];
            float cy = outputs[i][1];
            float w = outputs[i][2];
            float h = outputs[i][3];

            int left = static_cast<int>(img_scale_x * (cx - w / 2));
            int top = static_cast<int>(img_scale_y * (cy - h / 2));
            int bw = static_cast<int>(img_scale_x * w);
            int bh = static_cast<int>(img_scale_y * h);

            float max = outputs[i][5];
            int cls = 0;
            for (int64_t j = 0; j < output_column - 5; j++) {
                if (outputs[i][5 + j] > max) {
                    max = outputs[i][5 + j];
                    cls = static_cast<int>(j);
                }
            }
            results_.push_back(
                    YOLOResult{
                            .class_index=cls,
                            .class_name=classes_[cls],
                            .score = score,
                            .rect=cv::Rect(left, top, bw, bh),
                    });

        }
    }
}

std::vector<YOLOResult> YOLO::non_max_suppression() {
    // do an sort on the confidence scores, from high to low.
    std::sort(results_.begin(), results_.end(), [](auto &r1, auto &r2) {
        return r1.score > r2.score;
    });

    std::vector<YOLOResult> selected;
    std::vector<bool> active(results_.size(), true);
    int num_active = static_cast<int>(active.size());

    bool done = false;
    for (size_t i = 0; i < results_.size() && !done; i++) {
        if (active[i]) {
            const auto &box_a = results_[i];
            selected.push_back(box_a);
            if (selected.size() >= nms_limit)
                break;

            for (size_t j = i + 1; j < results_.size(); j++) {
                if (active[j]) {
                    const auto &box_b = results_[j];
                    if (IOU(box_a.rect, box_b.rect) > threshold) {
                        active[j] = false;
                        num_active -= 1;
                        if (num_active <= 0) {
                            done = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    return selected;
}
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using Classes = std::map<size_t, std::string>;
Classes read_classes(const std::string& file_name) {
  Classes classes;
  std::ifstream file(file_name);
  if (file) {
    std::string line;
    std::string id;
    std::string label;
    std::string token;
    size_t idx = 1;
    while (std::getline(file, line)) {
      std::stringstream line_stream(line);
      size_t i = 0;
      while (std::getline(line_stream, token, ' ')) {
        switch (i) {
          case 0:
            id = token;
            break;
          case 1:
            label = token;
            break;
        }
        token.clear();
        ++i;
      }
      classes.insert({idx, label});
      ++idx;
    }
  }
  return classes;
}

void show_model_info(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;

  auto num_inputs = session.GetInputCount();
  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input name " << i << " : " << input_name << std::endl;
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto tensor_shape = tensor_info.GetShape();
    std::cout << "Input shape " << i << " : ";
    for (size_t j = 0; j < tensor_shape.size(); ++j)
      std::cout << tensor_shape[j] << " ";
    std::cout << std::endl;
  }

  auto num_outputs = session.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    std::cout << "Output name " << i << " : " << output_name << std::endl;
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto tensor_shape = tensor_info.GetShape();
    std::cout << "Output shape " << i << " : ";
    for (size_t j = 0; j < tensor_shape.size(); ++j)
      std::cout << tensor_shape[j] << " ";
    std::cout << std::endl;
  }
}

void show_output(const std::vector<float>& output, const Classes& classes) {
  std::vector<std::pair<float, size_t>> pairs;  // prob : class index
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i] > 0.01f) {  // threshold check
      pairs.push_back(
          std::make_pair(output[i], i + 1));  // 0 - background
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::reverse(pairs.begin(), pairs.end());
  pairs.resize(std::min(5UL, pairs.size()));
  for (auto& p : pairs) {
    std::cout << "Class " << p.second << " Label "
              << classes.at(p.second) << " Prob "
              << p.first << std::endl;
  }
}

void read_image(const std::string& file_name,
                int width,
                int height,
                std::vector<float>& image_data) {
  // load image
  auto image = cv::imread(file_name, cv::IMREAD_COLOR);

  if (image.empty()) {
    throw std::invalid_argument("Failed to load image");
  }

  if (image.cols != width || image.rows != height) {
    // scale image to fit
    cv::Size scaled(std::max(height * image.cols / image.rows, width),
                    std::max(height, width * image.rows / image.cols));
    cv::resize(image, image, scaled);

    // crop image to fit
    cv::Rect crop((image.cols - width) / 2, (image.rows - height) / 2, width,
                  height);
    image = image(crop);
  }

  image.convertTo(image, CV_32FC3);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);

  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> stddev = {0.229, 0.224, 0.225};

  size_t i = 0;
  for (auto& c : channels) {
    c = ((c / 255) - mean[i]) / stddev[i];
    ++i;
  }

  cv::vconcat(channels[0], channels[1], image);
  cv::vconcat(image, channels[2], image);
  assert(image.isContinuous());

  std::copy_n(reinterpret_cast<float*>(image.data), image.size().area(),
              image_data.begin());
}

int main(int argc, char** argv) {
  try {
    if (argc == 4) {
      Ort::Env env;
      Ort::Session session(env, argv[1], Ort::SessionOptions{nullptr});
      auto classes = read_classes(argv[2]);

      show_model_info(session);

      constexpr const int width = 224;
      constexpr const int height = 224;
      std::array<int64_t, 4> input_shape{1, 3, width, height};
      std::vector<float> input_image(3 * width * height);
      read_image(argv[3], width, height, input_image);

      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(),
                                                                input_shape.data(), input_shape.size());

      std::array<int64_t, 2> output_shape{1, 1000};
      std::vector<float> result(1000);
      Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, result.data(), result.size(),
                                                                 output_shape.data(), output_shape.size());

      const char* input_names[] = {"data"};
      const char* output_names[] = {"resnetv17_dense0_fwd"};

      Ort::RunOptions run_options;
      session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

      show_output(result, classes);

      return 0;

    } else {
      std::cout << "Usage: <ONNX model file> <classes file> <input image>";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what();
  } catch (...) {
    std::cerr << "Unknown exception!";
  }
  return 1;
}

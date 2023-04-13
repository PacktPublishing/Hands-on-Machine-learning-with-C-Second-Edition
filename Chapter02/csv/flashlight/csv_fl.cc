#include <csv.h>
#include <flashlight/fl/autograd/Functions.h>
#include <flashlight/fl/dataset/TensorDataset.h>
#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/TensorBase.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace fs = std::filesystem;

std::tuple<fl::Tensor, fl::Tensor> load_dataset(const std::string& file_path) {
  if (fs::exists(file_path)) {
    constexpr int columns_num = 5;
    io::CSVReader<columns_num> csv_reader(file_path);

    fl::Tensor x;
    fl::Tensor y;
    std::array<float, 4> features;
    std::string class_id;
    while (csv_reader.read_row(features[0], features[1], features[2], features[3], class_id)) {
      auto sample_x = fl::Tensor::fromBuffer({1, 4}, features.data(), fl::MemoryLocation::Host);
      x = x.isEmpty() ? sample_x : fl::concatenate({x, sample_x}, 0);
      fl::Tensor sample_y;
      if (class_id == "Iris-virginica")
        sample_y = fl::reshape(fl::fromScalar(2.f), {1, 1});
      else if (class_id == "Iris-versicolor")
        sample_y = fl::reshape(fl::fromScalar(1.f), {1, 1});
      else if (class_id == "Iris-setosa")
        sample_y = fl::reshape(fl::fromScalar(0.f), {1, 1});
      y = y.isEmpty() ? sample_y : fl::concatenate({y, sample_y}, 0);
    }
    return std::make_tuple(fl::transpose(x), fl::transpose(y));
  } else {
    throw std::runtime_error("Invalid dataset file path");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      fl::init();
      auto [x, y] = load_dataset(argv[1]);

      // min-max scaling
      auto x_min = fl::amin(x, {1});
      auto x_max = fl::amax(x, {1});
      auto x_min_max = (x - x_min) / (x_max - x_min);
      // std::cout << x_min_max << std::endl;

      // normalization(z-score)
      auto x_mean = fl::mean(x, {1});
      auto x_std = fl::std(x, {1});
      auto x_norm = (x - x_mean) / x_std;
      // std::cout << x_norm << std::endl;

      auto dataset = std::make_shared<fl::TensorDataset>(std::vector<fl::Tensor>{x_norm, y});
      for (auto& sample : fl::ShuffleDataset(dataset)) {
        std::cout << "X:\n"
                  << sample[0] << std::endl;
        std::cout << "Y:\n"
                  << sample[1] << std::endl;
      }
    } else {
      std::cerr << "Please specify path to the dataset\n";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }

  return 0;
}

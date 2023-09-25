#include <csv.h>
#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>
#include <plot.h>
#include <filesystem>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv", "dataset4.csv"};

const std::vector<std::string> colors{"red", "green", "blue", "cyan", "black"};

using DataType = float;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Classes = std::map<size_t, PointCoords>;

void PlotClasses(const Classes& classes,
                 const std::string& name,
                 const std::string& file_name) {
  plotcpp::Plot plt(true);
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
  for (auto& cls : classes) {
    std::stringstream params;
    params << "lc rgb '" << colors[cls.first] << "' pt 7";
    plt.AddDrawing(
        draw_state,
        plotcpp::Points(cls.second.first.begin(), cls.second.first.end(),
                        cls.second.second.begin(),
                        std::to_string(cls.first) + " cls", params.str()));
  }

  plt.EndDraw2D(draw_state);
  plt.Flush();
}

std::tuple<fl::Tensor, fl::Tensor, size_t> load_dataset(const std::string& file_path) {
  if (fs::exists(file_path)) {
    constexpr int columns_num = 4;
    io::CSVReader<columns_num> csv_reader(file_path);

    using features_t = std::array<float, 3>;
    features_t sample;
    std::vector<features_t> samples;
    int index{0};
    std::set<int> classes;
    while (csv_reader.read_row(index, sample[0], sample[1], sample[2])) {
      classes.insert(static_cast<int>(sample[2]));
      samples.push_back(sample);
    }

    // shuffle the dataset
    // std::random_device rd;
    // std::mt19937 generator(rd());
    // std::shuffle(samples.begin(), samples.end(), generator);

    // move data into tensors
    fl::Tensor x;
    fl::Tensor y;
    for (auto& cur_sample : samples) {
      auto sample_x = fl::Tensor::fromBuffer({1, 2}, &cur_sample[0], fl::MemoryLocation::Host);
      x = x.isEmpty() ? sample_x : fl::concatenate({x, sample_x}, /*axis=*/0);
      // classes should be 1 and -1
      fl::Tensor sample_y = fl::reshape(fl::fromScalar(cur_sample[2] < 1.0f ? 1.0f : -1.0f), {1, 1});
      y = y.isEmpty() ? sample_y : fl::concatenate({y, sample_y}, /*axis=*/0);
    }

    return std::make_tuple(fl::transpose(x), fl::transpose(y), classes.size());
  } else {
    throw std::runtime_error("Invalid dataset file path");
  }
}

fl::Tensor train_linear_classifier(const fl::Tensor& train_x, const fl::Tensor& train_y, float learning_rate) {
  // train system
  int num_epochs = 100;
  int batch_size = 8;
  // Define dataset
  std::vector<fl::Tensor> fields{train_x, train_y};
  auto dataset = std::make_shared<fl::TensorDataset>(fields);
  auto batch_dataset = std::make_shared<fl::BatchDataset>(dataset, batch_size);

  auto weights = fl::Variable(fl::rand({train_x.shape().dim(0), 1}), /*calcGrad=*/true);
  double error = 0;
  for (int e = 1; e <= num_epochs; ++e) {
    fl::Tensor epoch_error = fl::fromScalar(0);
    for (auto& batch : *batch_dataset) {
      auto x = fl::Variable(batch[0], /*calcGrad=*/false);
      auto y = fl::Variable(fl::reshape(batch[1], {1, batch[1].shape().dim(0)}), /*calcGrad=*/false);
      auto z = fl::matmul(fl::transpose(weights), x);

      auto loss = fl::sum(fl::log(1 + fl::exp(-1 * y * z)), /*axes=*/{1});

      // Compute gradients using backprop
      loss.backward();

      // Update the weights
      weights.tensor() -= learning_rate * weights.grad().tensor();

      // clear the gradients for next iteration
      weights.zeroGrad();

      epoch_error += loss.tensor();
    }
    epoch_error /= batch_dataset->size();
    error += epoch_error.scalar<float>();
    // std::cout << "Epoch: " << e << " learning_rate: " << learning_rate << " loss: " << epoch_error.scalar<float>() << std::endl;
  }
  error /= num_epochs;
  std::cout << "Training finished:"
            << " learning_rate: " << learning_rate << " loss: " << error << std::endl;
  return weights.tensor();
}

template <typename Classifier>
void apply_classifier(Classifier classifier, const fl::Tensor& test_x, const fl::Tensor& test_y, const std::string& name) {
  auto num_samples = test_x.shape().dim(1);
  Classes classes;
  DataType accuracy = 0;
  for (fl::Dim i = 0; i != num_samples; i++) {
    auto sample = test_x(fl::span, i);
    auto target = test_y(fl::span, i);
    auto class_idx = classifier(sample);
    if (static_cast<int>(target.scalar<float>()) == class_idx)
      ++accuracy;
    classes[class_idx].first.push_back(sample(0).scalar<float>());
    classes[class_idx].second.push_back(sample(1).scalar<float>());
  }

  accuracy /= num_samples;

  PlotClasses(classes, name + std::to_string(accuracy), name + "-fl.png");
}

fl::Tensor make_kernel_matrix(const fl::Tensor& x, const fl::Tensor& z, float gamma) {
  // ||x-z||^2 = ||x||^2 + ||z||^2 - 2 * z^T * y

  auto x_norm = fl::sum(fl::power(x, 2), /*axes=*/{-1});
  x_norm = fl::reshape(x_norm, {x_norm.dim(0), 1});

  auto z_norm = fl::sum(fl::power(z, 2), /*axes=*/{-1});
  z_norm = fl::reshape(z_norm, {1, z_norm.dim(0)});

  auto k = fl::exp(-gamma * (x_norm + z_norm - 2 * fl::matmul(fl::transpose(x), z)));
  return k;
}

constexpr float rbf_gamma = 100.f;
fl::Tensor train_kernel_classifier(const fl::Tensor& train_x, const fl::Tensor& train_y, float learning_rate) {
  auto kx = make_kernel_matrix(train_x, train_x, rbf_gamma);
  return train_linear_classifier(kx, train_y, learning_rate);
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    for (auto& dataset : data_names) {
      auto dataset_name = base_dir / dataset;
      if (fs::exists(dataset_name)) {
        auto [inputs, labels, num_classes] = load_dataset(dataset_name);
        auto num_samples = inputs.shape().dim(1);
        auto num_features = inputs.shape().dim(0);

        std::cout << dataset << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num classes: " << num_classes << std::endl;

        // split data set to the train and test parts
        fl::Dim test_num = 300;
        auto test_x = inputs(fl::span, fl::range(0, test_num));
        auto train_x = inputs(fl::span, fl::range(test_num, fl::end));
        auto test_y = labels(fl::span, fl::range(0, test_num));
        auto train_y = labels(fl::span, fl::range(test_num, fl::end));

        std::cout << "Logistic regression:\n";
        auto weights = train_linear_classifier(train_x, train_y, /*learning_rate=*/0.1f);
        apply_classifier([&](const fl::Tensor& sample) {
          constexpr float threshold = 0.5;
          auto p = fl::sigmoid(fl::matmul(fl::transpose(weights), sample));
          if (p.scalar<float>() > threshold)
            return 1;
          else
            return 0;
        },
                         test_x, test_y, "logistic-" + dataset);

        std::cout << "Kernel logistic regression:\n";
        auto kweights = train_kernel_classifier(train_x, train_y, /*learning_rate=*/0.1f);
        apply_classifier([&](const fl::Tensor& sample) {
          constexpr float threshold = 0.5;
          auto k_sample = make_kernel_matrix(fl::reshape(sample, {sample.dim(0), 1}), train_x, rbf_gamma);
          auto p = fl::sigmoid(fl::matmul(fl::transpose(kweights), fl::transpose(k_sample)));
          if (p.scalar<float>() > threshold)
            return 1;
          else
            return 0;
        },
                         test_x, test_y, "kernel-logistic-" + dataset);

      } else {
        std::cout << "Dataset file was not found:" << dataset_name << std::endl;
      }
    }
  }
  return 0;
}

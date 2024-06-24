#include <dlib/matrix.h>
#include <dlib/random_forest.h>
#include <plot.h>

#include <iostream>
#include <random>
#include <regex>

using DataType = double;
using SampleType = dlib::matrix<DataType, 0, 1>;
using Samples = std::vector<SampleType>;
using Labels = std::vector<DataType>;
using DataSamples = std::vector<DataType>;

void PlotResults(const DataSamples& test_features,
                 const Labels& test_labels,
                 const Labels& pred_labels,
                 const std::string& title,
                 const std::string& file_name) {
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput(file_name + ".png");
  plt.SetTitle(title);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  plt.Draw2D(
      plotcpp::Points(test_features.begin(), test_features.end(), test_labels.begin(),
                      "orig", "lc rgb 'black' pt 7"),
      plotcpp::Lines(test_features.begin(), test_features.end(), pred_labels.begin(),
                     "pred", "lc rgb 'red' lw 2"));
  plt.Flush();
}

std::pair<Samples, Labels> GenerateNoiseData(DataType start,
                                             DataType end,
                                             size_t n) {
  Samples x;
  x.resize(n);
  Labels y;
  y.resize(n);

  std::mt19937 re(3467);
  std::uniform_real_distribution<DataType> dist(start, end);
  std::normal_distribution<DataType> noise_dist;

  for (size_t i = 0; i < n; ++i) {
    auto x_val = dist(re);
    auto y_val = std::cos(M_PI * x_val) + (noise_dist(re) * 0.3);
    x[i] = SampleType({x_val});
    y[i] = y_val;
  }

  return {x, y};
}

std::pair<Samples, Labels> GenerateData(DataType start,
                                        DataType end,
                                        size_t n) {
  Samples x;
  x.resize(n);
  Labels y;
  y.resize(n);

  auto step = (end - start) / (n - 1);
  auto x_val = start;
  size_t i = 0;
  for (auto& x_item : x) {
    x_item = SampleType({x_val});
    auto y_val = std::cos(M_PI * x_val);
    y[i] = y_val;
    x_val += step;
    ++i;
  }

  return {x, y};
}

int main() {
  using namespace dlib;

  constexpr DataType start = -10;
  constexpr DataType end = 10;
  constexpr size_t num_samples = 1000;
  constexpr size_t num_trees = 1000;

  random_forest_regression_trainer<dense_feature_extractor> trainer;
  trainer.set_num_trees(num_trees);
  trainer.set_seed("random forest");

  auto [train_samples, train_lables] = GenerateNoiseData(start, end, num_samples);
  auto random_forest = trainer.train(train_samples, train_lables);

  DataSamples data_samples(num_samples);
  Labels pred_lables(num_samples);
  size_t i = 0;
  auto [test_samples, test_lables] = GenerateData(start, end, num_samples);
  for (const auto& sample : test_samples) {
    auto prediction = random_forest(sample);
    data_samples[i] = sample(0);
    pred_lables[i] = prediction;
    ++i;
  }
  PlotResults(data_samples, test_lables, pred_lables, "DLib Random Forest", "dlib-rf");

  return 0;
}

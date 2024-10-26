#include <plot.h>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm.hpp>
#include <mlpack/methods/logistic_regression.hpp>
#include <mlpack/methods/softmax_regression.hpp>

namespace fs = std::filesystem;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv"};

const std::vector<std::string> colors{"red", "green", "blue", "cyan", "black"};

using DataType = double;
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

template <typename ViewCol, typename ViewRow>
void SVMClassification(size_t num_classes,
                       const arma::mat& train_input,
                       const ViewRow& train_labels,
                       const ViewCol& test_input,
                       const ViewRow& test_labels,
                       const std::string& name) {
  using namespace mlpack;
  LinearSVM<> lsvm;
  lsvm.Train(train_input, train_labels, num_classes);

  arma::Row<size_t> predictions;
  lsvm.Classify(test_input, predictions);

  Classes classes;
  DataType accuracy = 0;
  for (size_t i = 0; i != test_labels.n_cols; i++) {
    auto vec = test_input.col(i);
    auto class_idx = predictions(i);
    if (test_labels(i) == class_idx)
      ++accuracy;
    classes[class_idx].first.push_back(vec(0));
    classes[class_idx].second.push_back(vec(1));
  }

  accuracy /= test_input.n_cols;

  PlotClasses(classes, "SVM " + std::to_string(accuracy),
              name + "-svm-mlpack.png");
}

template <typename ViewCol, typename ViewRow>
void LRClassification(const arma::mat& train_input,
                      const ViewRow& train_labels,
                      const ViewCol& test_input,
                      const ViewRow& test_labels,
                      const std::string& name) {
  using namespace mlpack;
  LogisticRegression<> lr;
  lr.Train(train_input, train_labels);

  arma::Row<size_t> predictions;
  lr.Classify(test_input, predictions);

  Classes classes;
  DataType accuracy = 0;
  for (size_t i = 0; i != test_labels.n_cols; i++) {
    auto vec = test_input.col(i);
    auto class_idx = predictions(i);
    if (test_labels(i) == class_idx)
      ++accuracy;
    classes[class_idx].first.push_back(vec(0));
    classes[class_idx].second.push_back(vec(1));
  }

  accuracy /= test_input.n_cols;

  PlotClasses(classes, "Logistic Regression " + std::to_string(accuracy),
              name + "-lr-mlpack.png");
}

template <typename ViewCol, typename ViewRow>
void SMRClassification(size_t num_classes,
                       const arma::mat& train_input,
                       const ViewRow& train_labels,
                       const ViewCol& test_input,
                       const ViewRow& test_labels,
                       const std::string& name) {
  using namespace mlpack;
  SoftmaxRegression smr(train_input.n_cols, num_classes);
  smr.Train(train_input, train_labels, num_classes);

  arma::Row<size_t> predictions;
  smr.Classify(test_input, predictions);

  Classes classes;
  DataType accuracy = 0;
  for (size_t i = 0; i != test_labels.n_cols; i++) {
    auto vec = test_input.col(i);
    auto class_idx = predictions(i);
    if (test_labels(i) == class_idx)
      ++accuracy;
    classes[class_idx].first.push_back(vec(0));
    classes[class_idx].second.push_back(vec(1));
  }

  accuracy /= test_input.n_cols;

  PlotClasses(classes, "SoftMax Regression " + std::to_string(accuracy),
              name + "-smr-mlpack.png");
}

int main(int argc, char** argv) {
  using namespace mlpack;
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    for (auto& dataset : data_names) {
      auto dataset_name = base_dir / dataset;
      if (fs::exists(dataset_name)) {
        arma::mat data;
        mlpack::data::DatasetInfo info;
        data::Load(dataset_name, data, info, /*fail with error*/ true);

        arma::Row<size_t> labels;
        labels = arma::conv_to<arma::Row<size_t>>::from(data.row(data.n_rows - 1));

        // remove labels row
        data.shed_row(data.n_rows - 1);
        // remove indices row
        data.shed_row(0);

        auto num_samples = data.n_cols;
        auto num_features = data.n_rows;
        std::size_t num_classes =
            std::set<double>(labels.begin(), labels.end()).size();

        std::cout << dataset << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num classes: " << num_classes << std::endl;

        // split data set to the train and test parts - make views
        size_t test_num = 300;
        auto test_input = data.head_cols(test_num);
        auto test_labels = labels.head_cols(test_num);
        arma::mat train_input = data.tail_cols(num_samples - test_num);
        auto train_labels = labels.tail_cols(num_samples - test_num);

        SVMClassification(num_classes, train_input, train_labels, test_input, test_labels, dataset);

        {
          arma::Row<size_t> two_class_train_labels = train_labels;
          two_class_train_labels(arma::find(two_class_train_labels > 0)).fill(1);
          arma::Row<size_t> two_class_test_labels = test_labels;
          two_class_test_labels(arma::find(two_class_test_labels > 0)).fill(1);
          LRClassification(train_input, two_class_train_labels, test_input, two_class_test_labels, dataset);
        }

        SMRClassification(num_classes, train_input, train_labels, test_input, test_labels, dataset);

      } else {
        std::cerr << "Failed to open dataset " << dataset_name << "\n";
      }
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}

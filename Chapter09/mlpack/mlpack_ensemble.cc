#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost.hpp>
#include <mlpack/methods/decision_tree.hpp>
#include <mlpack/methods/random_forest.hpp>
#include <regex>
#include "stacking.h"

namespace fs = std::filesystem;

void RFClassification(size_t num_classes,
                      const arma::mat& train_input,
                      const arma::Row<size_t>& train_labels,
                      const arma::mat& test_input,
                      const arma::Row<size_t>& test_labels) {
  using namespace mlpack;
  RandomForest<> rf;
  rf.Train(train_input,
           train_labels,
           num_classes,
           /*numTrees=*/100,
           /*minimumLeafSize=*/10,     // Minimum number of points in each tree's leaf nodes.
           /*minimumGainSplit=*/1e-7,  // Minimum gain for splitting a decision tree node.
           /*maximumDepth=*/10);

  Accuracy acc;
  auto acc_value = acc.Evaluate(rf, test_input, test_labels);
  std::cout << "Random Forest accuracy = " << acc_value << std::endl;
}

void DTClassification(size_t num_classes,
                      const arma::mat& train_input,
                      const arma::Row<size_t>& train_labels,
                      const arma::mat& test_input,
                      const arma::Row<size_t>& test_labels) {
  using namespace mlpack;
  DecisionTree<> dt;
  dt.Train(train_input,
           train_labels,
           num_classes,
           /*minimumLeafSize=*/10,     // Minimum number of points in each tree's leaf nodes.
           /*minimumGainSplit=*/1e-7,  // Minimum gain for splitting a decision tree node.
           /*maximumDepth=*/10);

  // arma::Row<size_t> predictions;
  // dt.Classify(test_input, predictions);

  Accuracy acc;
  auto acc_value = acc.Evaluate(dt, test_input, test_labels);
  std::cout << "Decision Tree accuracy = " << acc_value << std::endl;
}

void ABClassification(size_t num_classes,
                      const arma::mat& train_input,
                      const arma::Row<size_t>& train_labels,
                      const arma::mat& test_input,
                      const arma::Row<size_t>& test_labels) {
  using namespace mlpack;
  Perceptron<> p;
  AdaBoost<Perceptron<>> ab;
  ab.Train(train_input,
           train_labels,
           num_classes,
           p,
           /*iterations*/ 1000,
           /*tolerance*/ 1e-10);

  Accuracy acc;
  auto acc_value = acc.Evaluate(ab, test_input, test_labels);
  std::cout << "AdaBoost accuracy = " << acc_value << std::endl;
}

std::string ConvertToCSV(const std::string& dataset_name) {
  auto new_dataset_name = dataset_name + ".csv";
  if (fs::exists(dataset_name)) {
    std::ifstream file(dataset_name);
    std::ofstream out_file(new_dataset_name);
    std::string line;
    while (std::getline(file, line)) {
      std::regex re("[\\s,]+");
      std::sregex_token_iterator it(line.begin(), line.end(), re, -1);
      std::sregex_token_iterator reg_end;
      std::vector<std::string> tokens;
      for (int i = 0; it != reg_end; ++it, ++i) {
        if (i == 0) {  // skip
          continue;
        } else if (i == 1) {
          if (it->str() == "M")
            tokens.push_back("0");
          else
            tokens.push_back("1");
        } else {
          tokens.push_back(it->str());
        }
        tokens.push_back(", ");
      }
      tokens.resize(tokens.size() - 1);
      for (auto& token : tokens) {
        out_file << token;
      }
      out_file << "\n";
    }
  }
  return new_dataset_name;
}

int main(int argc, char** argv) {
  using namespace mlpack;
  if (argc > 1) {
    std::string dataset_name = fs::path(argv[1]);
    if (fs::exists(dataset_name)) {
      dataset_name = ConvertToCSV(dataset_name);
      arma::mat data;
      mlpack::data::DatasetInfo info;
      data::Load(dataset_name, data, info, /*fail with error*/ true);

      arma::Row<size_t> labels;
      labels = arma::conv_to<arma::Row<size_t>>::from(data.row(0));

      // remove labels row
      data.shed_row(0);

      auto num_samples = data.n_cols;
      auto num_features = data.n_rows;
      std::size_t num_classes =
          std::set<double>(labels.begin(), labels.end()).size();

      std::cout << "num samples: " << num_samples
                << "; num features: " << num_features
                << "; num classes: " << num_classes << std::endl;

      // split data set to the train and test parts - make views
      size_t train_num = 500;
      arma::mat train_input = data.head_cols(train_num);
      arma::Row<size_t> train_labels = labels.head_cols(train_num);
      arma::mat test_input = data.tail_cols(num_samples - train_num);
      arma::Row<size_t> test_labels = labels.tail_cols(num_samples - train_num);

      RFClassification(num_classes, train_input, train_labels, test_input, test_labels);
      DTClassification(num_classes, train_input, train_labels, test_input, test_labels);
      ABClassification(num_classes, train_input, train_labels, test_input, test_labels);
      StackingClassification(num_classes, train_input, train_labels, test_input, test_labels);

    } else {
      std::cerr << "Failed to open dataset " << dataset_name << "\n";
    }

  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}

#include "stacking.h"

#include <mlpack/methods/decision_tree.hpp>
#include <mlpack/methods/linear_svm.hpp>
#include <mlpack/methods/pca.hpp>
#include <mlpack/methods/softmax_regression.hpp>

struct KFoldDataSet {
  KFoldDataSet(const arma::mat& train_input,
               const arma::Row<size_t>& train_labels,
               size_t k)
      : k(k) {
    bin_size = train_input.n_cols / k;
    last_bin_size = train_input.n_cols - ((k - 1) * bin_size);

    inputs = arma::join_rows(train_input,
                             train_input.cols(0, train_input.n_cols - last_bin_size - 1));
    labels = arma::join_rows(train_labels,
                             train_labels.cols(0, train_labels.n_cols - last_bin_size - 1));
  }

  std::pair<arma::mat, arma::Row<size_t>> get_fold(const size_t i) {
    // If this is not the first fold, we have to handle it a little bit
    // differently, since the last fold may contain slightly more than 'binSize'
    // points.
    const size_t subset_size = (i != 0) ? last_bin_size + (k - 2) * bin_size : (k - 1) * bin_size;

    auto input_fold = arma::mat(inputs.colptr(bin_size * i), inputs.n_rows, subset_size, false, true);

    auto labels_fold = arma::Row<size_t>(labels.colptr(bin_size * i), subset_size, false, true);

    return {input_fold, labels_fold};
  }

  size_t k{0};
  size_t bin_size{0};
  size_t last_bin_size{0};
  arma::mat inputs;
  arma::Row<size_t> labels;
};

void StackingClassification(size_t num_classes,
                            const arma::mat& raw_train_input,
                            const arma::Row<size_t>& raw_train_labels,
                            const arma::mat& test_input,
                            const arma::Row<size_t>& test_labels) {
  using namespace mlpack;

  // Shuffle data
  arma::mat train_input;
  arma::Row<size_t> train_labels;
  ShuffleData(raw_train_input, raw_train_labels, train_input, train_labels);

  // Normalize data
  data::StandardScaler sample_scaler;
  sample_scaler.Fit(train_input);
  arma::mat scaled_train_input(train_input.n_rows, train_input.n_cols);
  sample_scaler.Transform(train_input, scaled_train_input);

  std::cout << "Data normalized, num dimensions " << scaled_train_input.n_rows << std::endl;

  // train weak models for predictions
  LinearSVM<> weak0;
  weak0.Train(scaled_train_input, train_labels, num_classes);

  SoftmaxRegression weak1(scaled_train_input.n_cols, num_classes);
  weak1.Train(scaled_train_input, train_labels, num_classes);

  DecisionTree<> weak2;
  weak2.Train(scaled_train_input, train_labels, num_classes);

  std::cout << "Weak models trained" << std::endl;

  // Generate meta dataset
  arma::mat meta_train_inputs;
  arma::Row<size_t> meta_train_labels;

  size_t k = 10;
  size_t half_len = scaled_train_input.n_cols / 2;
  KFoldDataSet meta_train(scaled_train_input.head_cols(half_len), train_labels.head_cols(half_len), k);
  KFoldDataSet meta_validation(scaled_train_input.tail_cols(half_len), train_labels.tail_cols(half_len), k);

  for (size_t i = 0; i < k; ++i) {
    auto [fold_train_inputs, fold_train_labels] = meta_train.get_fold(i);
    auto [fold_valid_inputs, fold_valid_labels] = meta_validation.get_fold(i);

    arma::Row<size_t> predictions;

    LinearSVM<> local_weak0;
    local_weak0.Train(fold_train_inputs, fold_train_labels, num_classes);
    local_weak0.Classify(fold_valid_inputs, predictions);
    meta_train_inputs = arma::join_rows(meta_train_inputs, arma::conv_to<arma::mat>::from(predictions));
    meta_train_labels = arma::join_rows(meta_train_labels, fold_valid_labels);

    SoftmaxRegression local_weak1(fold_train_inputs.n_cols, num_classes);
    local_weak1.Train(fold_train_inputs, fold_train_labels, num_classes);
    local_weak1.Classify(fold_valid_inputs, predictions);
    meta_train_inputs = arma::join_rows(meta_train_inputs, arma::conv_to<arma::mat>::from(predictions));
    meta_train_labels = arma::join_rows(meta_train_labels, fold_valid_labels);

    DecisionTree<> local_weak2;
    local_weak2.Train(fold_train_inputs, fold_train_labels, num_classes);
    local_weak2.Classify(fold_valid_inputs, predictions);
    meta_train_inputs = arma::join_rows(meta_train_inputs, arma::conv_to<arma::mat>::from(predictions));
    meta_train_labels = arma::join_rows(meta_train_labels, fold_valid_labels);
  }
  // shuffle meta dataset
  {
    arma::mat rand_train_input;
    arma::Row<size_t> rand_train_labels;
    ShuffleData(meta_train_inputs, meta_train_labels, rand_train_input, rand_train_labels);
    meta_train_inputs = rand_train_input;
    meta_train_labels = rand_train_labels;
  }
  std::cout << "Meta dataset was created" << std::endl;

  // train meta model
  DecisionTree<> meta_model;
  meta_model.Train(meta_train_inputs, meta_train_labels, num_classes);

  std::cout << "Meta algorithm trained" << std::endl;

  // evaluate ensemble
  arma::mat scaled_test_input(test_input.n_rows, test_input.n_cols);
  sample_scaler.Transform(test_input, scaled_test_input);

  arma::mat meta_eval_inputs;
  arma::Row<size_t> meta_eval_labes;
  arma::Row<size_t> predictions;

  weak0.Classify(scaled_test_input, predictions);
  meta_eval_inputs = arma::join_rows(meta_eval_inputs, arma::conv_to<arma::mat>::from(predictions));
  meta_eval_labes = arma::join_rows(meta_eval_labes, test_labels);

  weak1.Classify(scaled_test_input, predictions);
  meta_eval_inputs = arma::join_rows(meta_eval_inputs, arma::conv_to<arma::mat>::from(predictions));
  meta_eval_labes = arma::join_rows(meta_eval_labes, test_labels);

  weak2.Classify(scaled_test_input, predictions);
  meta_eval_inputs = arma::join_rows(meta_eval_inputs, arma::conv_to<arma::mat>::from(predictions));
  meta_eval_labes = arma::join_rows(meta_eval_labes, test_labels);

  Accuracy acc;
  auto acc_value = acc.Evaluate(meta_model, meta_eval_inputs, meta_eval_labes);
  std::cout << "Stacking ensemble accuracy = " << acc_value << std::endl;
}
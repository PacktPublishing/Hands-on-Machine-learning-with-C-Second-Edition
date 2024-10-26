#include <plot.h>
#include <algorithm>
#include <iostream>
#include <mlpack/core.hpp>

std::pair<arma::mat, arma::rowvec> GenerateData(size_t num_samples) {
  arma::mat samples = arma::randn<arma::mat>(1, num_samples);
  arma::rowvec labels = samples + arma::randn<arma::rowvec>(num_samples, arma::distr_param(1.0, 1.5));
  return {samples, labels};
}

int main(int /*argc*/, char** /*argv*/) {
  using namespace mlpack;

  size_t num_samples = 1000;
  auto [raw_samples, raw_labels] = GenerateData(num_samples);

  auto mm = std::minmax_element(
      raw_samples.begin(), raw_samples.end(),
      [](const auto& a, const auto& b) { return a < b; });
  std::pair<double, double> x_minmax{*mm.first, *mm.second};

  // Normalize data
  data::StandardScaler sample_scaler;
  sample_scaler.Fit(raw_samples);
  arma::mat samples(1, num_samples);
  sample_scaler.Transform(raw_samples, samples);

  data::StandardScaler label_scaler;
  label_scaler.Fit(raw_labels);
  arma::rowvec labels(num_samples);
  label_scaler.Transform(raw_labels, labels);

  // Randomize data
  arma::mat rand_samples(1, num_samples);
  arma::rowvec rand_labels(num_samples);
  ShuffleData(samples, labels, rand_samples, rand_labels);

  // Grid search for the best regularization parameter lambda
  // Using 80% of data for training and remaining 20% for assessing MSE.
  double validation_size = 0.2;
  HyperParameterTuner<LinearRegression<>, MSE, SimpleCV> parameters_tuner(validation_size,
                                                                        rand_samples, rand_labels);

  // Finding the best value for lambda from the values 0.0, 0.001, 0.01, 0.1,
  // and 1.0.
  arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  double best_lambda = 0;
  std::tie(best_lambda) = parameters_tuner.Optimize(lambdas);

  std::cout << "Best lambda: " << best_lambda << std::endl;

  // Train model
  // double lambda = 0.01;
  // LinearRegression<> linear_regression(rand_samples, rand_labels, lambda);

  // Use best model
  LinearRegression<>& linear_regression = parameters_tuner.BestModel();

  // Make new perdictions
  size_t num_new_samples = 50;
  arma::dvec new_samples_values = arma::linspace<arma::dvec>(x_minmax.first, x_minmax.second, num_new_samples);
  arma::mat new_samples(1, num_new_samples);
  new_samples.row(0) = arma::trans(new_samples_values);
  arma::mat norm_new_samples(1, num_new_samples);
  sample_scaler.Transform(new_samples, norm_new_samples);
  arma::rowvec predictions(num_new_samples);
  linear_regression.Predict(norm_new_samples, predictions);

  // Plot perdictions
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput("plot.png");
  plt.SetTitle("Linear regression");
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  std::vector<double> x_coords(samples.size());
  std::copy(samples.begin(), samples.end(), x_coords.begin());

  std::vector<double> x_pred_coords(new_samples.size());
  std::copy(new_samples.begin(), new_samples.end(), x_pred_coords.begin());

  plt.Draw2D(plotcpp::Points(x_coords.begin(), x_coords.end(),
                             labels.begin(), "orig", "lc rgb 'black' pt 7"),
             plotcpp::Lines(x_pred_coords.begin(), x_pred_coords.end(),
                            predictions.begin(), "pred", "lc rgb 'red' lw 2"));
  plt.Flush();

  return 0;
}

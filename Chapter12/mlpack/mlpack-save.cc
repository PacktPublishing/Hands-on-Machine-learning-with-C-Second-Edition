#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;

using ModelType = FFN<MeanSquaredError, ConstInitialization>;

ModelType make_model() {
  MeanSquaredError loss;
  ConstInitialization init(0.);
  ModelType model(loss, init);
  model.Add<Linear>(8);
  model.Add<ReLU>();
  model.Add<Linear>(16);
  model.Add<ReLU>();
  model.Add<Linear>(32);
  model.Add<ReLU>();
  model.Add<Linear>(1);
  return model;
}

int main() {
  size_t n = 10000;
  arma::mat x = arma::randn(n).t();
  arma::mat y = x * 0.3f + 0.4f;

  // Define optimizer
  ens::Adam optimizer;

  auto model = make_model();

  model.Train(x, y, optimizer, ens::ProgressBar());

  data::Save("model.bin", model.Parameters(), true);

  auto new_model = make_model();
  data::Load("model.bin", new_model.Parameters(), true);

  arma::mat predictions;
  new_model.Predict(x, predictions);

  auto mse = SquaredEuclideanDistance::Evaluate(predictions, y) / (y.n_elem);
  std::cout << "Final MSE: " << mse << std::endl;

  return 0;
}

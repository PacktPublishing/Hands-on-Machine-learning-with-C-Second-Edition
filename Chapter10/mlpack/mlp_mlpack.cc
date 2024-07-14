#include "../data/data.h"

#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;

int main() {
  
  size_t n = 10000;
  size_t seed = 45345;
  auto data = GenerateData(-1.5, 1.5, n, seed, false);

  arma::mat x = arma::mat(data.first).t();
  arma::mat y = arma::mat(data.second).t();

  // Normalize data
  arma::mat scaled_x;
  arma::mat scaled_y;

  data::MinMaxScaler x_scaler;
  x_scaler.Fit(x);
  x_scaler.Transform(x, scaled_x);

  data::MinMaxScaler y_scaler;
  y_scaler.Fit(y);
  y_scaler.Transform(y, scaled_y);

  // Define model
  MeanSquaredError loss;
  ConstInitialization init(0.);
  FFN<MeanSquaredError, ConstInitialization> model(loss, init);
  model.Add<Linear>(8);
  model.Add<ReLU>();
  model.Add<Linear>(16);
  model.Add<ReLU>();
  model.Add<Linear>(32);
  model.Add<ReLU>();
  model.Add<Linear>(1);

  // Define optimizer
  size_t epochs = 100;
  ens::MomentumSGD optimizer(/*stepSize=*/0.01,
                             /*batchSize= */ 64,
                             /*maxIterations= */ epochs * x.n_cols,
                             /*tolerance=*/1e-10,
                             /*shuffle=*/false);
                             

  ens::StoreBestCoordinates<arma::mat> best_params;

  model.Train(scaled_x, scaled_y, optimizer, ens::ProgressBar(), ens::EarlyStopAtMinLoss(20), best_params);

 // model.Parameters() = best_params.BestCoordinates();

  arma::mat predictions;
  model.Predict(scaled_x, predictions);

  auto mse = SquaredEuclideanDistance::Evaluate(predictions, scaled_y) / (scaled_y.n_elem);
  std::cout << "Final MSE: " << mse << std::endl;

  return 0;
}

#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>
#include <plot.h>
#include <iostream>

std::pair<fl::Tensor, fl::Tensor> generate_data(int num_samples) {
  auto samples = fl::randn({1, num_samples});
  auto labels = fl::cos(M_PI * samples) + (fl::randn({1, num_samples}) * 0.3);
  return {samples, labels};
}

fl::Tensor make_samples_polynomial(const fl::Tensor& samples, int polynomial_degree) {
  fl::Tensor polynomial_samples;
  for (int64_t sample_index = 0; sample_index < samples.shape().dim(1); ++sample_index) {
    auto sample = samples(fl::span, sample_index);
    auto sample_polynomial = fl::tile(sample, {polynomial_degree, 1});
    auto degrees = fl::iota({polynomial_degree, 1}) + 1;
    sample_polynomial = fl::power(sample_polynomial, degrees);
    polynomial_samples = polynomial_samples.isEmpty() ? sample_polynomial : fl::concatenate({polynomial_samples, sample_polynomial}, 1);
  }
  return polynomial_samples;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " polynomial_degree learning_rate batch_size" << std::endl;
    return 0;
  } else {
    // Hyper parameters

    // The best values selected manually
    // int polynomial_degree = 14;
    // double learning_rate = 0.045;
    // int batch_size = 32;

    int polynomial_degree = std::atoi(argv[1]);
    double learning_rate = std::atof(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int num_epochs = 300;

    // genreate training data
    int num_samples = 1000;
    auto [samples, labels] = generate_data(num_samples);

    // Apply polynomial regression transform
    auto poly_samples = make_samples_polynomial(samples, polynomial_degree);

    // Normalize data
    auto samples_mean = fl::mean(poly_samples, /*reduction axis*/ {1}, /*keep_dims*/ true);
    auto samples_std = fl::std(poly_samples, /*reduction axis*/ {1}, /*keep_dims*/ true);
    poly_samples = (poly_samples - samples_mean) / samples_std;

    // Define dataset
    std::vector<fl::Tensor> fields{poly_samples, labels};
    auto dataset = std::make_shared<fl::TensorDataset>(fields);
    auto batch_dataset = std::make_shared<fl::BatchDataset>(dataset, batch_size);

    // train system
    auto weight = fl::Variable(fl::rand({polynomial_degree, 1}), /*calcGrad*/ true);
    auto bias = fl::Variable(fl::full({1}, 0.0), /*calcGrad*/ true);

    float mse = 0;
    fl::MeanSquaredError mse_func;
    for (int e = 1; e <= num_epochs; ++e) {
      fl::Tensor error = fl::fromScalar(0);
      for (auto& batch : *batch_dataset) {
        auto input = fl::Variable(batch[0], /*calcGrad*/ false);

        auto local_batch_size = batch[0].shape().dim(1);
        auto predictions = fl::matmul(fl::transpose(weight), input) + fl::tile(bias, {1, local_batch_size});
        auto targets = fl::Variable(fl::reshape(batch[1], {1, local_batch_size}), /*calcGrad*/ false);

        // Mean Squared Error Loss
        auto loss = mse_func.forward(predictions, targets);

        // Compute gradients using backprop
        loss.backward();

        // Update the weight and bias
        weight.tensor() -= learning_rate * weight.grad().tensor();
        bias.tensor() -= learning_rate * bias.grad().tensor();

        // clear the gradients for next iteration
        weight.zeroGrad();
        bias.zeroGrad();
        mse_func.zeroGrad();

        error += loss.tensor();
      }

      // Mean Squared Error
      error /= batch_dataset->size();
      mse = error.scalar<float>();
      // std::cout << "Epoch: " << e << " learning_rate: " << learning_rate << " MSE: " << mse << std::endl;
    }

    // Plot perdictions
    int new_samples_num = 50;
    float start_range = -3.f;
    float end_range = 3.f;
    auto new_samples = fl::arange(start_range, end_range, (end_range - start_range) / new_samples_num);
    auto new_poly_samples = make_samples_polynomial(fl::reshape(new_samples, {1, new_samples_num}), polynomial_degree);
    new_poly_samples = (new_poly_samples - samples_mean) / samples_std;
    auto predictions = fl::matmul(fl::transpose(weight.tensor()), new_poly_samples) + bias.tensor();

    std::string plot_file_name = "plot_" +
                                 std::to_string(polynomial_degree) +
                                 "_" +
                                 std::to_string(learning_rate) +
                                 "_" +
                                 std::to_string(batch_size) +
                                 ".png";
    plotcpp::Plot plt;
    plt.SetTerminal("png");
    plt.SetOutput(plot_file_name);
    plt.SetTitle("Polynomial regression");
    plt.SetXLabel("x");
    plt.SetYLabel("y");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid");

    std::vector<float> x_coords = samples.toHostVector<float>();
    std::vector<float> y_coords = labels.toHostVector<float>();
    std::vector<float> x_pred_coords = new_samples.toHostVector<float>();
    std::vector<float> y_pred_coords = predictions.toHostVector<float>();

    plt.Draw2D(plotcpp::Points(x_coords.begin(), x_coords.end(),
                               y_coords.begin(), "orig", "lc rgb 'black' pt 7"),
               plotcpp::Lines(x_pred_coords.begin(), x_pred_coords.end(),
                              y_pred_coords.begin(), "pred", "lc rgb 'red' lw 2"));
    plt.Flush();

    std::cout << mse;
  }

  return 0;
}

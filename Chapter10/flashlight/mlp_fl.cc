#include "../data/data.h"

#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>
#include <iostream>

int main() {
  fl::init();

  size_t n = 10000;
  size_t seed = 45345;
  auto data = GenerateData(-1.5, 1.5, n, seed, false);

  fl::Tensor x = fl::Tensor::fromVector(data.first);
  fl::Tensor y = fl::Tensor::fromVector(data.second);

  // Normalize data
  auto x_mean = fl::mean(x, /*reduction axis*/ {0}, /*keep_dims*/ true);
  auto x_std = fl::std(x, /*reduction axis*/ {0}, /*keep_dims*/ true);
  x = (x - x_mean) / x_std;

  // Define dataset
  std::vector<fl::Tensor> fields{x, y};
  auto dataset = std::make_shared<fl::TensorDataset>(fields);
  fl::BatchDataset batch_dataset(dataset, /*batch_size=*/64);

  // Deifine model
  fl::Sequential model;
  model.add(fl::View({1, 1, 1, -1}));  // to process a batch
  model.add(fl::Linear(1, 8));
  model.add(fl::ReLU());
  model.add(fl::Linear(8, 16));
  model.add(fl::ReLU());
  model.add(fl::Linear(16, 32));
  model.add(fl::ReLU());
  model.add(fl::Linear(32, 1));

  // define MSE loss
  auto loss = fl::MeanSquaredError();

  // Define optimizer
  float learning_rate = 0.01;
  float momentum = 0.5;
  auto sgd = fl::SGDOptimizer(model.params(), learning_rate, momentum);

  // Define epoch average MSE meter
  fl::AverageValueMeter meter;

  const int epochs = 500;
  for (int epoch_i = 0; epoch_i < epochs; ++epoch_i) {
    meter.reset();
    for (auto& batch : batch_dataset) {
      sgd.zeroGrad();

      // Forward propagation
      auto predicted = model(fl::input(batch[0]));

      // Calculate loss
      auto local_batch_size = batch[0].shape().dim(0);
      auto target = fl::reshape(batch[1], {1, 1, 1, local_batch_size});
      auto loss_value = loss(predicted, fl::noGrad(target));

      // Backward propagation
      loss_value.backward();

      // Update parameters
      sgd.step();

      meter.add(loss_value.scalar<double>());
    }
    std::cout << "Epoch: " << epoch_i << " MSE: " << meter.value()[0]
              << std::endl;
  }

  auto predicted = model(fl::noGrad(x));
  auto target = fl::reshape(y, {1, 1, 1, x.shape().dim(0)});
  auto mse = loss(predicted, fl::noGrad(target));
  std::cout << "Final MSE: " << mse.scalar<double>() << std::endl;

  return 0;
}

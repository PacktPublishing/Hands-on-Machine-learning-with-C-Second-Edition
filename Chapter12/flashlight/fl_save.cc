#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>
#include <iostream>

int main() {
  fl::init();

  int64_t n = 10000;

  auto x = fl::randn({n});
  auto y = x * 0.3f + 0.4f;

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

  const int epochs = 5;
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

      meter.add(loss_value.scalar<float>());
    }
    std::cout << "Epoch: " << epoch_i << " MSE: " << meter.value()[0]
              << std::endl;
  }
  // fl::save("model.dat", model);
  fl::save("model_params.dat", model.params());

  fl::Sequential model_loaded;
  // fl::load("model.dat", model_loaded);

  model_loaded.add(fl::View({1, 1, 1, -1}));  // to process a batch
  model_loaded.add(fl::Linear(1, 8));
  model_loaded.add(fl::ReLU());
  model_loaded.add(fl::Linear(8, 16));
  model_loaded.add(fl::ReLU());
  model_loaded.add(fl::Linear(16, 32));
  model_loaded.add(fl::ReLU());
  model_loaded.add(fl::Linear(32, 1));
  std::vector<fl::Variable> params;
  fl::load("model_params.dat", params);
  for (int i = 0; i < static_cast<int>(params.size()); ++i) {
    model_loaded.setParams(params[i], i);
  }

  auto predicted = model_loaded(fl::noGrad(x));
  auto target = fl::reshape(y, {1, 1, 1, x.shape().dim(0)});
  auto mse = loss(predicted, fl::noGrad(target));
  std::cout << "Final MSE: " << mse.scalar<float>() << std::endl;

  return 0;
}

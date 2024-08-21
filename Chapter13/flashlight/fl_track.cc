#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>

#include "mlflow.h"

#include <iostream>

std::shared_ptr<fl::BatchDataset> make_dataset(int64_t n, int batch_size) {
  // generate some linear dependency
  auto x = fl::randn({n});
  auto y = x * 0.3f + 0.4f;

  // and add some small noise
  auto noise = fl::randn({n}) * 0.1f;
  y += noise; 

  std::vector<fl::Tensor> fields{x, y};
  auto dataset = std::make_shared<fl::TensorDataset>(fields);
  return std::make_shared<fl::BatchDataset>(dataset, batch_size);
}

int main() {
  fl::init();

  MLFlow mlflow;
  mlflow.set_experiment("Linear regression");

  // set run parameters
  int batch_size = 64;
  float learning_rate = 0.0001;
  float momentum = 0.01;
  const int epochs = 100;
  
  // start a run
  mlflow.start_run();

  auto train_dataset = make_dataset(/*n=*/10000, batch_size);
  auto test_dataset = make_dataset(/*n=*/1000, batch_size);

  // Define a model
  fl::Sequential model;
  model.add(fl::View({1, 1, 1, -1}));  // to process a batch
  model.add(fl::Linear(1, 1));

  // define MSE loss
  auto loss = fl::MeanSquaredError();

  // Define optimizer
  auto sgd = fl::SGDOptimizer(model.params(), learning_rate, momentum);

  // Define epoch average MSE meter
  fl::AverageValueMeter meter;

  // launch the training cycle
  for (int epoch_i = 0; epoch_i < epochs; ++epoch_i) {
    meter.reset();
    model.train();
    for (auto& batch : *train_dataset) {
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

    auto avr_loss_value = meter.value()[0];
    mlflow.log_metric("train loss", avr_loss_value, epoch_i);
    std::cout << "Epoch: " << epoch_i << "\n\ttrain MSE: " << avr_loss_value << std::endl;

    // every 10th epoch calculate test metric
    if (epoch_i % 10 == 0) {
      fl::AverageValueMeter test_meter;
      model.eval();
      for (auto& batch : *test_dataset) {
        auto predicted = model(fl::input(batch[0]));
        // Calculate loss
        auto local_batch_size = batch[0].shape().dim(0);
        auto target = fl::reshape(batch[1], {1, 1, 1, local_batch_size});
        auto loss_value = loss(predicted, fl::noGrad(target));
        test_meter.add(loss_value.scalar<float>());
      }
      auto avr_loss_value = test_meter.value()[0];
      mlflow.log_metric("test loss", avr_loss_value, epoch_i);
      std::cout << "\t test MSE: " << avr_loss_value << std::endl;
    }
  }
  mlflow.end_run();
  mlflow.log_param("epochs", epochs);
  mlflow.log_param("batch_size", batch_size);
  mlflow.log_param("learning_rate", learning_rate);
  mlflow.log_param("momentum", momentum);

  std::cout << "Learned model params:\n";
  for (auto& param : model.params()) {
    std::cout << param.host<float>()[0] << std::endl;
  }
  return 0;
}

#include "imdbdataset.h"
#include "model.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

struct Stack : public torch::data::transforms::Collation<ImdbExample> {
  ImdbExample apply_batch(std::vector<ImdbExample> examples) override {
    std::vector<torch::Tensor> input_ids;
    std::vector<torch::Tensor> attention_masks;
    std::vector<torch::Tensor> labels;
    input_ids.reserve(examples.size());
    attention_masks.reserve(examples.size());
    labels.reserve(examples.size());
    for (auto& example : examples) {
      input_ids.push_back(std::move(example.data.first));
      attention_masks.push_back(std::move(example.data.second));
      labels.push_back(std::move(example.target));
    }
    return {{torch::stack(input_ids), torch::stack(attention_masks)}, torch::stack(labels)};
  }
};

int main(int argc, char** argv) {
  if (argc == 4) {
    auto dataset_path = fs::path(argv[1]);
    if (!fs::exists(dataset_path)) {
      std::cerr << "Incorrect dataset path!\n";
    }
    auto vocab_path = fs::path(argv[2]);
    if (!fs::exists(vocab_path)) {
      std::cerr << "Incorrect vocabulary path!\n";
    }
    auto model_path = fs::path(argv[3]);
    if (!fs::exists(model_path)) {
      std::cerr << "Incorrect model path!\n";
    }

    torch::DeviceType device = torch::cuda::is_available()
                                   ? torch::DeviceType::CUDA
                                   : torch::DeviceType::CPU;

    auto tokenizer = std::make_shared<Tokenizer>(vocab_path);
    ImdbDataset train_dataset(dataset_path / "train", tokenizer);

    int batch_size = 8;
    auto train_loader = torch::data::make_data_loader(
        train_dataset.map(Stack()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(8));  // random sampler is default

    Model model(model_path);
    model->to(device);

    torch::optim::AdamW optimizer(model->parameters(),
                                  torch::optim::AdamWOptions(1e-5));

    int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch) {
      model->train();

      int batch_index = 0;
      for (auto& batch : (*train_loader)) {
        optimizer.zero_grad();

        auto batch_label = batch.target.to(device);
        auto batch_input_ids = batch.data.first.squeeze(1).to(device);
        auto batch_attention_mask = batch.data.second.squeeze(1).to(device);

        auto output = model(batch_input_ids, batch_attention_mask);

        torch::Tensor loss =
            torch::cross_entropy_loss(output, batch_label);

        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();

        // Output the loss and accuracy every 10 batches.
        if (++batch_index % 10 == 0) {
          auto predictions = output.argmax(/*dim=*/1);
          auto acc = (predictions == batch_label).sum().item<float>() / batch_size;

          std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << " | Acc: " << acc << std::endl;
        }
      }
    }
    return 0;
  }

  std::cerr << "Please specify dataset folder, vocablary file path, and model file path\n";
  return 1;
}

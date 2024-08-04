#pragma once

#include <torch/script.h>
#include <torch/torch.h>

class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl() = delete;
  ModelImpl(const std::string& bert_model_path);

  torch::Tensor forward(at::Tensor input_ids, at::Tensor attention_masks);

 private:
  torch::jit::script::Module bert_;
  torch::nn::Dropout dropout_;
  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
};

TORCH_MODULE(Model);

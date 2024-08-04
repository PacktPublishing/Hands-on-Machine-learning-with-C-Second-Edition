#include "model.h"

ModelImpl::ModelImpl(const std::string& bert_model_path)
    : dropout_(register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions().p(0.2)))),
      fc1_(register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(768, 512)))),
      fc2_(register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(512, 2)))) {
  bert_ = torch::jit::load(bert_model_path);
}

torch::Tensor ModelImpl::forward(at::Tensor input_ids, at::Tensor attention_masks) {
  std::vector<torch::jit::IValue> inputs = {input_ids, attention_masks};
  auto bert_output = bert_.forward(inputs);
  // Pooled output is the embedding of the [CLS] token (from Sequence output),
  // further processed by a Linear layer and a Tanh activation function.
  // The Linear layer weights are trained from the next sentence prediction
  // (classification) objective during pretraining.
  auto pooler_output = bert_output.toTuple()->elements()[1].toTensor();
  auto x = fc1_(pooler_output);
  x = torch::nn::functional::relu(x);
  x = dropout_(x);
  x = fc2_(x);
  x = torch::softmax(x, /*dim=*/1);
  return x;
}


#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

class Tokenizer {
 public:
  Tokenizer(const std::string& vocab_file_path, int max_len = 128);

  std::pair<torch::Tensor, torch::Tensor> tokenize(const std::string text);

 private:
  std::unordered_map<std::string, int> vocab_;
  int max_len_{0};
};

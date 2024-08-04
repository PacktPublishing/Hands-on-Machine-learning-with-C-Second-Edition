#include "imdbdataset.h"
#include <cassert>
#include <fstream>

ImdbDataset::ImdbDataset(const std::string& dataset_path,
                         std::shared_ptr<Tokenizer> tokenizer)
    : reader_(dataset_path), tokenizer_(std::move(tokenizer)) {}

ImdbExample ImdbDataset::get(size_t index) {
  torch::Tensor target;
  const std::string* review{nullptr};
  if (index < reader_.get_pos_size()) {
    review = &reader_.get_pos(index);
    target = torch::tensor(1, torch::dtype(torch::kLong));
  } else {
    review = &reader_.get_neg(index - reader_.get_pos_size());
    target = torch::tensor(0, torch::dtype(torch::kLong));
  }
  // encode text
  auto tokenizer_out = tokenizer_->tokenize(*review);

  return {tokenizer_out, target.squeeze()};
}

torch::optional<size_t> ImdbDataset::size() const {
  return reader_.get_pos_size() + reader_.get_neg_size();
}

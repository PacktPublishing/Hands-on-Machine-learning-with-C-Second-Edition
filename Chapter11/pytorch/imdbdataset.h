#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include "imdbreader.h"
#include "tokenizer.h"

#include <torch/torch.h>

#include <string>

using ImdbData = std::pair<torch::Tensor, torch::Tensor>;
using ImdbExample = torch::data::Example<ImdbData, torch::Tensor>;

class ImdbDataset : public torch::data::Dataset<ImdbDataset, ImdbExample> {
 public:
  ImdbDataset(const std::string& dataset_path,
              std::shared_ptr<Tokenizer> tokenizer);

  // torch::data::Dataset implementation
  ImdbExample get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  ImdbReader reader_;
  std::shared_ptr<Tokenizer> tokenizer_;
};

#endif  // MNISTDATASET_H

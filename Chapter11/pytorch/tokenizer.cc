#include "tokenizer.h"

#include <fstream>

Tokenizer::Tokenizer(const std::string& vocab_file_path, int max_len)
    : max_len_{max_len} {
  auto file = std::ifstream(vocab_file_path);
  std::string line;
  while (std::getline(file, line)) {
    auto sep_pos = line.find_first_of(' ');
    auto token = line.substr(0, sep_pos);
    auto id = std::stoi(line.substr(sep_pos + 1));
    vocab_.insert({token, id});
  }
}

std::pair<torch::Tensor, torch::Tensor> Tokenizer::tokenize(const std::string text) {
  std::string pad_token = "[PAD]";
  std::string start_token = "[CLS]";
  std::string end_token = "[SEP]";
  auto pad_token_id = vocab_[pad_token];
  auto start_token_id = vocab_[start_token];
  auto end_token_id = vocab_[end_token];

  std::vector<int> input_ids(max_len_, pad_token_id);
  std::vector<int> attention_mask(max_len_, 0);
  input_ids[0] = start_token_id;
  attention_mask[0] = 1;

  std::string word;
  std::istringstream ss(text);

  int input_id = 1;
  while (getline(ss, word, ' ')) {
    size_t start = 0;
    while (start < word.size()) {
      size_t end = word.size();
      std::string token;
      bool has_token = false;
      while (start < end) {
        auto token = word.substr(start, end - start);
        if (start > 0)
          token = "##" + token;
        auto token_iter = vocab_.find(token);
        if (token_iter != vocab_.end()) {
          attention_mask[input_id] = 1;
          input_ids[input_id] = token_iter->second;
          ++input_id;
          has_token = true;
          break;
        }
        end--;
      }
      if (input_id == max_len_ - 1) {
        break;
      }
      if (!has_token) {
        break;
      }
      start = end;
    }
    if (input_id == max_len_ - 1) {
      break;
    }
  }
  attention_mask[input_id] = 1;
  input_ids[input_id] = end_token_id;

  auto input_ids_tensor = torch::tensor(input_ids).unsqueeze(0);
  auto attention_masks_tensor = torch::tensor(attention_mask).unsqueeze(0);
  return std::make_pair(input_ids_tensor, attention_masks_tensor);
}
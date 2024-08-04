#pragma once

#include <string>
#include <vector>

class ImdbReader {
 public:
  ImdbReader(const std::string& root_path);

  size_t get_pos_size() const;
  size_t get_neg_size() const;

  const std::string& get_pos(size_t index) const;
  const std::string& get_neg(size_t index) const;

 private:
  using Reviews = std::vector<std::string>;

  void read_directory(const std::string& path, Reviews& reviews);

 private:
  Reviews pos_samples_;
  Reviews neg_samples_;
  size_t max_size_{0};
};

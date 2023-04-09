#include "reviewsreader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>

using json = nlohmann::json;

enum class HandlerState {
  None,
  Global,
  PapersArray,
  Paper,
  ReviewArray,
  Review
};

struct ReviewsHandler : public json::json_sax_t {
  ReviewsHandler(Papers* papers)
      : papers_(papers) {}

  bool null() override {
    return true;
  }

  bool boolean(bool) override {
    return true;
  }

  bool number_integer(number_integer_t) override {
    return true;
  }

  bool number_float(number_float_t, const string_t&) override {
    return true;
  }

  bool binary(json::binary_t&) override {
    return true;
  }

  bool parse_error(std::size_t, const std::string&, const json::exception& ex) override {
    error_ = ex.what();
    return false;
  }

  bool
  number_unsigned(number_unsigned_t u) override {
    bool res{true};
    try {
      if (state_ == HandlerState::Paper && key_ == "id") {
        paper_.id = u;
      } else if (state_ == HandlerState::Review && key_ == "id") {
        review_.id = u;
      } else {
        res = false;
      }
    } catch (...) {
      res = false;
    }
    key_.clear();
    return res;
  }

  bool string(string_t& str) override {
    bool res{true};
    try {
      if (state_ == HandlerState::Paper && key_ == "preliminary_decision") {
        paper_.preliminary_decision = str;
      } else if (state_ == HandlerState::Review && key_ == "confidence") {
        review_.confidence = str;
      } else if (state_ == HandlerState::Review && key_ == "evaluation") {
        review_.evaluation = str;
      } else if (state_ == HandlerState::Review && key_ == "lan") {
        review_.language = str;
      } else if (state_ == HandlerState::Review && key_ == "orientation") {
        review_.orientation = str;
      } else if (state_ == HandlerState::Review && key_ == "remarks") {
        review_.remarks = str;
      } else if (state_ == HandlerState::Review && key_ == "text") {
        review_.text = str;
      } else if (state_ == HandlerState::Review && key_ == "timespan") {
        review_.timespan = str;
      } else {
        res = false;
      }
    } catch (...) {
      res = false;
    }
    key_.clear();
    return res;
  }

  bool key(string_t& str) override {
    key_ = str;
    return true;
  }

  bool start_object(std::size_t) override {
    if (state_ == HandlerState::None && key_.empty()) {
      state_ = HandlerState::Global;
    } else if (state_ == HandlerState::PapersArray && key_.empty()) {
      state_ = HandlerState::Paper;
    } else if (state_ == HandlerState::ReviewArray && key_.empty()) {
      state_ = HandlerState::Review;
    } else {
      return false;
    }
    return true;
  }

  bool end_object() override {
    if (state_ == HandlerState::Global) {
      state_ = HandlerState::None;
    } else if (state_ == HandlerState::Paper) {
      state_ = HandlerState::PapersArray;
      papers_->push_back(paper_);
      paper_ = Paper();
    } else if (state_ == HandlerState::Review) {
      state_ = HandlerState::ReviewArray;
      paper_.reviews.push_back(review_);
    } else {
      return false;
    }
    return true;
  }

  bool start_array(std::size_t) override {
    if (state_ == HandlerState::Global && key_ == "paper") {
      state_ = HandlerState::PapersArray;
      key_.clear();
    } else if (state_ == HandlerState::Paper && key_ == "review") {
      state_ = HandlerState::ReviewArray;
      key_.clear();
    } else {
      return false;
    }
    return true;
  }

  bool end_array() override {
    if (state_ == HandlerState::ReviewArray) {
      state_ = HandlerState::Paper;
    } else if (state_ == HandlerState::PapersArray) {
      state_ = HandlerState::Global;
    } else {
      return false;
    }
    return true;
  }

  Paper paper_;
  Review review_;
  std::string key_;
  Papers* papers_{nullptr};
  HandlerState state_{HandlerState::None};
  std::string error_;
};

Papers ReadPapersReviews(const std::string& filename) {
  std::ifstream file(filename);
  if (file) {
    Papers papers;
    ReviewsHandler handler(&papers);
    bool result = json::sax_parse(file, &handler);

    if (!result) {
      throw std::runtime_error(handler.error_);
    }
    return papers;
  } else {
    throw std::invalid_argument("File can't be opened " + filename);
  }
}

#include "mlflow.h"

#include <chrono>
#include <nlohmann/json.hpp>

namespace {
bool check_result(const httplib::Result& res, int code) {
  if (!res) {
    throw std::runtime_error("REST error: " + httplib::to_string(res.error()));
  }
  return res->status == code;
};

void handle_result(const httplib::Result& res) {
  if (check_result(res, 200))
    return;
  std::ostringstream oss;
  oss << "Request error status: " << res->status << " " << httplib::status_message(res->status);
  oss << ", message: " << std::endl
      << res->body;
  throw std::runtime_error(oss.str());
};
}  // namespace

MLFlow::MLFlow()
    : http_client_("http://127.0.0.1:5000") {}

MLFlow::MLFlow(const std::string& host, size_t port)
    : http_client_(host, port) {
}

void MLFlow::set_experiment(const std::string& name) {
  auto res = http_client_.Get("/api/2.0/mlflow/experiments/get-by-name?experiment_name=" + name);
  if (check_result(res, 404)) {
    // create new one
    nlohmann::json request;
    request["name"] = name;
    res = http_client_.Post("/api/2.0/mlflow/experiments/create", request.dump(), "application/json");
    handle_result(res);
    // remember id
    auto json = nlohmann::json::parse(res->body);
    experiment_id_ = json["experiment_id"].get<std::string>();
  } else if (check_result(res, 200)) {
    // remember id
    auto json = nlohmann::json::parse(res->body);
    experiment_id_ = json["experiment"]["experiment_id"].get<std::string>();
  } else {
    handle_result(res);
  }
}

void MLFlow::start_run() {
  nlohmann::json request;
  request["experiment_id"] = experiment_id_;
  request["start_time"] = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  auto res = http_client_.Post("/api/2.0/mlflow/runs/create", request.dump(), "application/json");
  handle_result(res);
  auto json = nlohmann::json::parse(res->body);
  run_id_ = json["run"]["info"]["run_id"];
}

void MLFlow::end_run() {
  nlohmann::json request;
  request["run_id"] = run_id_;
  request["status"] = "FINISHED";
  request["end_time"] = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  auto res = http_client_.Post("/api/2.0/mlflow/runs/update", request.dump(), "application/json");
  handle_result(res);
}

void MLFlow::log_metric(const std::string& name, float value, size_t epoch) {
  nlohmann::json request;
  request["run_id"] = run_id_;
  request["key"] = name;
  request["value"] = value;
  request["step"] = epoch;
  request["timestamp"] = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  auto res =
      http_client_.Post("/api/2.0/mlflow/runs/log-metric", request.dump(), "application/json");
  handle_result(res);
}

void MLFlow::log_param(const std::string& name, const std::string& value) {
  nlohmann::json request;
  request["run_id"] = run_id_;
  request["key"] = name;
  request["value"] = value;

  auto res =
      http_client_.Post("/api/2.0/mlflow/runs/log-parameter", request.dump(), "application/json");
  handle_result(res);
}


#pragma once

#include <httplib.h>

#include <string>

// The mlflow REST API use implementation https://mlflow.org/docs/latest/rest-api.html
// 1. start a server with command: mlflow server --backend-store-uri file:///some_directory/mlruns

class MLFlow {
 public:
  MLFlow();
  MLFlow(const std::string& host, size_t port);

  ~MLFlow() = default;
  MLFlow(const MLFlow&) = delete;
  MLFlow& operator=(const MLFlow&) = delete;
  MLFlow(MLFlow&&) = delete;
  MLFlow& operator=(MLFlow&&) = delete;

  void set_experiment(const std::string& name);
  void start_run();
  void end_run();
  void log_metric(const std::string& name, float value, size_t epoch);
  void log_param(const std::string& name, const std::string& value);

  template <typename T>
  void log_param(const std::string& name, T value) {
    log_param(name, std::to_string(value));
  }

 private:
  httplib::Client http_client_;
  std::string experiment_id_;
  std::string run_id_;
};
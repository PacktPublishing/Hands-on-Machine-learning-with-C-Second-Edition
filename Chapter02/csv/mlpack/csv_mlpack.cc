#include <filesystem>
#include <fstream>
#include <iostream>
#include <mlpack/core.hpp>
#include <regex>

namespace fs = std::filesystem;

using namespace mlpack;

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      if (fs::exists(argv[1])) {
        arma::mat dataset;
        mlpack::data::DatasetInfo info;
        data::Load(argv[1], dataset, info, /*fail with error*/ true);
        std::cout << "Number of dimensions: " << info.Dimensionality() << std::endl;
        std::cout << "Number of classes: " << info.NumMappings(4) << std::endl;

        arma::Row<size_t> labels;
        labels = arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
        dataset.shed_row(dataset.n_rows - 1);

        data::MinMaxScaler min_max_scaler;
        min_max_scaler.Fit(dataset);

        arma::mat scaled_dataset;
        min_max_scaler.Transform(dataset, scaled_dataset);

        std::cout << scaled_dataset << std::endl;

        min_max_scaler.InverseTransform(scaled_dataset, dataset);

        data::StandardScaler standard_scaler;
        standard_scaler.Fit(dataset);

        standard_scaler.Transform(dataset, scaled_dataset);

        std::cout << scaled_dataset << std::endl;

        standard_scaler.InverseTransform(scaled_dataset, dataset);
      } else {
        std::cerr << "Invalid file path " << argv[1] << "\n";
      }
    } else {
      std::cerr << "Please specify path to the dataset\n";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }

  return 0;
}

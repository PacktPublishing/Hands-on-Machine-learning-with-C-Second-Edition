#include <plot.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/gmm.hpp>
#include <mlpack/methods/linear_svm.hpp>

using namespace mlpack;
namespace fs = std::filesystem;

const std::vector<std::string> colors{"black", "red", "blue", "green",
                                      "cyan", "yellow", "brown", "magenta"};

using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

void PlotClusters(const Clusters& clusters,
                  const std::string& name,
                  const std::string& file_name) {
  plotcpp::Plot plt(true);
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.GnuplotCommand("set size square");
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
  for (auto& cluster : clusters) {
    std::stringstream params;
    params << "lc rgb '" << colors[cluster.first] << "' pt 7";
    plt.AddDrawing(draw_state,
                   plotcpp::Points(
                       cluster.second.first.begin(), cluster.second.first.end(),
                       cluster.second.second.begin(),
                       std::to_string(cluster.first) + " cls", params.str()));
  }

  plt.EndDraw2D(draw_state);
  plt.Flush();
}

void MultivariateGaussianDist(const arma::mat& normal,
                              const arma::mat& test,
                              const std::string& file_name) {
  GMM gmm(/*gaussians*/ 1, /*dimensionality*/ 2);
  KMeans<> kmeans;
  size_t max_iterations = 250;
  double tolerance = 1e-10;
  EMFit<KMeans<>, NoConstraint> em(max_iterations, tolerance, kmeans);
  gmm.Train(normal, /*trials*/ 3, /*use_existing_model*/ false, em);

  // change this parameter to see descision boundary
  double prob_threshold = 0.001;

  Clusters plot_clusters;

  auto detect = [&](const arma::mat& samples) {
    for (size_t c = 0; c < samples.n_cols; ++c) {
      auto sample = samples.col(c);
      double x = sample.at(0, 0);
      double y = sample.at(1, 0);
      auto p = gmm.Probability(sample);
      if (p >= prob_threshold) {
        plot_clusters[0].first.push_back(x);
        plot_clusters[0].second.push_back(y);
      } else {
        plot_clusters[1].first.push_back(x);
        plot_clusters[1].second.push_back(y);
      }
    }
  };
  detect(normal);
  detect(test);
  PlotClusters(plot_clusters, "Multivariate Gaussian Distribution", file_name);
}

using Dataset = std::pair<arma::mat, arma::mat>;

Dataset LoadDataset(const fs::path& file_path) {
  arma::mat dataset;
  mlpack::data::DatasetInfo info;
  // by default dataset will be loaded trasposed
  data::Load(file_path, dataset, info, /*fail with error*/ true);

  // split the data assuming that columns are samples and rows are features
  long n_normal = 50;
  arma::mat normal = dataset.cols(0, n_normal - 1);
  arma::mat test = dataset.cols(n_normal, dataset.n_cols - 1);

  return {normal, test};
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);

    std::string data_name_multi{"multivar.csv"};
    std::string data_name_uni{"univar.csv"};

    auto dataset_multi = LoadDataset(base_dir / data_name_multi);
    auto dataset_uni = LoadDataset(base_dir / data_name_uni);

    MultivariateGaussianDist(dataset_multi.first, dataset_multi.second,
                             "mlpack-multi-var.png");
    // OneClassSvm(dataset_multi.first, dataset_multi.second, "dlib-ocsvm.png");

  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }
  return 0;
}

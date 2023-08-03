#include <plot.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan.hpp>
#include <mlpack/methods/gmm.hpp>
#include <mlpack/methods/kmeans.hpp>
#include <mlpack/methods/mean_shift.hpp>

using namespace mlpack;
namespace fs = std::filesystem;

const std::vector<std::string> dataset_names{"dataset0.csv", "dataset1.csv",
                                             "dataset2.csv", "dataset3.csv",
                                             "dataset4.csv", "dataset5.csv"};

const std::vector<std::string> colors{"black", "red", "blue", "green",
                                      "cyan", "yellow", "brown", "magenta"};

using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

void PlotClusters(const Clusters& clusters,
                  const std::string& name,
                  const std::string& file_name) {
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
  for (auto& cluster : clusters) {
    std::stringstream params;
    auto color_index = cluster.first % colors.size();
    params << "lc rgb '" << colors[color_index] << "' pt 7";
    plt.AddDrawing(draw_state,
                   plotcpp::Points(
                       cluster.second.first.begin(), cluster.second.first.end(),
                       cluster.second.second.begin(),
                       std::to_string(color_index) + " cls", params.str()));
  }

  plt.EndDraw2D(draw_state);
  plt.Flush();
}

void DoKMeansClustering(const arma::mat& inputs,
                        size_t num_clusters,
                        const std::string& name) {
  arma::Row<size_t> assignments;
  KMeans<> kmeans;
  kmeans.Cluster(inputs, num_clusters, assignments);

  Clusters plot_clusters;
  for (size_t i = 0; i != inputs.n_cols; ++i) {
    auto cluser_idx = assignments[i];
    plot_clusters[cluser_idx].first.push_back(inputs.at(0, i));
    plot_clusters[cluser_idx].second.push_back(inputs.at(1, i));
  }

  PlotClusters(plot_clusters, "K-Means", name + "-kmeans.png");
}

void DoDBScanClustering(const arma::mat& inputs,
                        const std::string& name) {
  arma::Row<size_t> assignments;

  DBSCAN<> dbscan(/*epsilon*/ 0.1, /*min_points*/ 15);
  dbscan.Cluster(inputs, assignments);

  Clusters plot_clusters;
  for (size_t i = 0; i != inputs.n_cols; ++i) {
    auto cluser_idx = assignments[i];
    if (cluser_idx != SIZE_MAX) {
      plot_clusters[cluser_idx].first.push_back(inputs.at(0, i));
      plot_clusters[cluser_idx].second.push_back(inputs.at(1, i));
    }
  }

  PlotClusters(plot_clusters, "DBDScan", name + "-dbscan.png");
}

void DoMeanShiftClustering(const arma::mat& inputs,
                           const std::string& name) {
  arma::Row<size_t> assignments;
  arma::mat centroids;

  MeanShift<> mean_shift;
  auto radius = mean_shift.EstimateRadius(inputs);
  mean_shift.Radius(radius);
  mean_shift.Cluster(inputs, assignments, centroids);

  Clusters plot_clusters;
  for (size_t i = 0; i != inputs.n_cols; ++i) {
    auto cluser_idx = assignments[i];
    plot_clusters[cluser_idx].first.push_back(inputs.at(0, i));
    plot_clusters[cluser_idx].second.push_back(inputs.at(1, i));
  }

  PlotClusters(plot_clusters, "MeanShift", name + "-mean-shift.png");
}

void DoGMMClustering(const arma::mat& inputs,
                     size_t num_clusters,
                     const std::string& name) {
  GMM gmm(num_clusters, /*dimensionality*/ 2);
  KMeans<> kmeans;
  size_t max_iterations = 250;
  double tolerance = 1e-10;
  EMFit<KMeans<>, NoConstraint> em(max_iterations, tolerance, kmeans);
  gmm.Train(inputs, /*trials*/ 3, /*use_existing_model*/ false, em);

  arma::Row<size_t> assignments;
  gmm.Classify(inputs, assignments);

  Clusters plot_clusters;
  for (size_t i = 0; i != inputs.n_cols; ++i) {
    auto cluser_idx = assignments[i];
    plot_clusters[cluser_idx].first.push_back(inputs.at(0, i));
    plot_clusters[cluser_idx].second.push_back(inputs.at(1, i));
  }

  PlotClusters(plot_clusters, "GMM", name + "-gmm.png");
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    for (auto& dataset_name : dataset_names) {
      auto dataset_full_name = base_dir / dataset_name;
      if (fs::exists(dataset_full_name)) {
        arma::mat dataset;
        mlpack::data::DatasetInfo info;
        // by default dataset will be loaded trasposed
        data::Load(dataset_full_name, dataset, info, /*fail with error*/ true);

        arma::Row<size_t> labels;
        labels = arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
        // remove label row
        dataset.shed_row(dataset.n_rows - 1);
        // remove index row
        dataset.shed_row(0);

        auto num_samples = dataset.n_cols;
        auto num_features = dataset.n_rows;
        std::size_t num_clusters =
            std::set<double>(labels.begin(), labels.end()).size();
        if (num_clusters < 2)
          num_clusters = 3;

        std::cout << dataset_name << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num clusters: " << num_clusters << std::endl;

        DoKMeansClustering(dataset, num_clusters, dataset_name);
        DoDBScanClustering(dataset, dataset_name);
        DoMeanShiftClustering(dataset, dataset_name);
        DoGMMClustering(dataset, num_clusters, dataset_name);

      } else {
        std::cerr << "Dataset file " << dataset_name << " missed\n";
      }
    }
    return 0;
  }
}

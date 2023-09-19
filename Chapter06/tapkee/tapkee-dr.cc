// http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html

#include <plot.h>
#include <tapkee/tapkee.hpp>
#include "util.h"

#include <filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

using DataType = double;
using index_t = int64_t;

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan"};

using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using PointCoords3d = std::tuple<Coords, Coords, Coords>;
using Clusters = std::unordered_map<index_t, PointCoords>;
using Clusters3d = std::unordered_map<index_t, PointCoords3d>;

const std::string data_file_name{"swissroll.dat"};
const std::string labels_file_name{"swissroll_labels.dat"};

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

void Plot3DData(const tapkee::DenseMatrix& features,
                const tapkee::DenseMatrix& lables) {
  Clusters3d clusters;

  for (index_t i = 0; i < features.cols(); ++i) {
    auto vector = features.col(i);
    auto label = static_cast<int>(lables(i));
    std::get<0>(clusters[label]).push_back(vector[0]);
    std::get<1>(clusters[label]).push_back(vector[1]);
    std::get<2>(clusters[label]).push_back(vector[2]);
  }

  plotcpp::Plot plt(true);
  plt.SetTerminal("png");
  plt.SetOutput("3d_data.png");
  plt.SetTitle("Swissroll data");
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.GnuplotCommand("set size square");
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw3D<Coords::const_iterator>();
  for (auto& cluster : clusters) {
    std::stringstream params;
    params << "lc rgb '" << colors[cluster.first] << "' pt 7";
    plt.AddDrawing(draw_state,
                   plotcpp::Points3D(std::get<0>(cluster.second).begin(),
                                     std::get<0>(cluster.second).end(),
                                     std::get<1>(cluster.second).begin(),
                                     std::get<2>(cluster.second).begin(),
                                     std::to_string(cluster.first) + " cls",
                                     params.str()));
  }

  plt.EndDraw3D(draw_state);
  plt.Flush();
}

struct gaussian_kernel_callback {
  gaussian_kernel_callback(const tapkee::DenseMatrix& matrix, tapkee::ScalarType gamma)
      : feature_matrix(matrix), gamma(gamma){};
  inline tapkee::ScalarType kernel(tapkee::IndexType a, tapkee::IndexType b) const {
    auto distance = (feature_matrix.col(a) - feature_matrix.col(b)).norm();
    return exp(-(distance * distance) * gamma);
  }
  inline tapkee::ScalarType operator()(tapkee::IndexType a, tapkee::IndexType b) const {
    return kernel(a, b);
  }
  const tapkee::DenseMatrix& feature_matrix;
  tapkee::ScalarType gamma{1};
};

void Reduction(tapkee::ParametersSet parameters,
               bool with_kernel,
               const tapkee::DenseMatrix& features,
               const tapkee::DenseMatrix& lables,
               const std::string& img_file) {
  using namespace tapkee;
  // eigen_kernel_callback kcb(features);
  gaussian_kernel_callback kcb(features, 2.0);
  eigen_distance_callback dcb(features);
  eigen_features_callback fcb(features);

  auto n = features.cols();
  std::vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;

  TapkeeOutput result;
  if (with_kernel) {
    result = initialize().withParameters(parameters).withKernel(kcb).withFeatures(fcb).withDistance(dcb).embedRange(indices.begin(), indices.end());
  } else {
    result = initialize().withParameters(parameters).withFeatures(fcb).withDistance(dcb).embedRange(indices.begin(), indices.end());
  }

  Clusters clusters;
  for (index_t i = 0; i < result.embedding.rows(); ++i) {
    auto new_vector = result.embedding.row(i);
    auto label = static_cast<int>(lables(i));
    clusters[label].first.push_back(new_vector[0]);
    clusters[label].second.push_back(new_vector[1]);
  }

  PlotClusters(clusters, get_method_name(parameters[method]), img_file);
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto data_dir = fs::path(argv[1]);
    auto data_file_path = data_dir / data_file_name;
    auto lables_file_path = data_dir / labels_file_name;
    if (fs::exists(data_file_path) && fs::exists(lables_file_path)) {
      char delimiter = ' ';
      tapkee::DenseMatrix input_data = read_data(data_file_path.string(), delimiter);
      if (input_data.size() == 0) {
        std::cerr << "Failed to read input data\n";
        exit(1);
      }
      tapkee::DenseMatrix labels_data = read_data(lables_file_path.string(), delimiter);
      if (input_data.size() == 0) {
        std::cerr << "Failed to read label data\n";
        exit(1);
      }

      input_data.transposeInPlace();
      labels_data.transposeInPlace();

      Plot3DData(input_data, labels_data);

      int target_dim = 2;
      using namespace tapkee;

      Reduction((method = FactorAnalysis,
                 target_dimension = target_dim,
                 fa_epsilon = 1e-5,
                 max_iteration = 100),
                false, input_data, labels_data, "fa-tapkee.png");

      Reduction((method = tDistributedStochasticNeighborEmbedding,
                 target_dimension = target_dim,
                 sne_perplexity = 30),
                false, input_data, labels_data, "tsne-tapkee.png");

      Reduction((method = PCA,
                 target_dimension = target_dim),
                false, input_data, labels_data, "pca-tapkee.png");

      Reduction((method = KernelPCA,
                 target_dimension = target_dim),
                true, input_data, labels_data, "kernel-pca-tapkee.png");

      Reduction((method = Isomap,
                 target_dimension = target_dim,
                 num_neighbors = 100),
                false, input_data, labels_data, "isomap-tapkee.png");

      Reduction((method = MultidimensionalScaling,
                 target_dimension = target_dim),
                false, input_data, labels_data, "mds-tapkee.png");

    } else {
      std::cerr << "Dataset file " << data_file_path << " missed\n";
    }

  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}

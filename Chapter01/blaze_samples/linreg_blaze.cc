
#include <blaze/Math.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "blaze/math/StorageOrder.h"
#include "blaze/math/dense/DynamicVector.h"

typedef blaze::DynamicMatrix<float, blaze::columnMajor> Matrix;
typedef blaze::DynamicVector<float, blaze::columnVector> Vector;

std::pair<Vector, Matrix> GenerateData(size_t n) {
  std::vector<float> x_data(n);
  std::iota(x_data.begin(), x_data.end(), 0);
  std::vector<float> y_data(n);
  std::iota(y_data.begin(), y_data.end(), 0);

  // mutate data
  std::random_device rd;
  std::mt19937 re(rd());
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

  for (auto& x : x_data) {
    x += dist(re);  // add noise
  }

  for (auto& y : y_data) {
    y += dist(re);  // add noise
  }

  // Make result
  Vector x(n, x_data.data());
  Matrix y(n, 1UL, y_data.data());

  return {x, y};
}

int main() {
  size_t n = 1000;
  // generate training data
  Vector x0 = blaze::uniform<blaze::columnVector>(n, 1.f);
  Vector x1;
  Matrix y;
  std::tie(x1, y) = GenerateData(n);

  // setup line coeficients y = b(4) + k(0.3)*x
  y *= 0.3f;
  y += 4.f;

  // combine X matrix
  Matrix x(n, 2UL);
  blaze::column<0UL>(x) = x0;
  blaze::column<1UL>(x) = x1;

  // solve normal equation
  // calculate X^T*X
  auto xtx = blaze::trans(x) * x;

  // calculate the inverse of X^T*X
  auto inv_xtx = blaze::inv(xtx);

  // calculate X^T*y
  auto xty = blaze::trans(x) * y;

  // calculate the coefficients of the linear regression
  Matrix beta = inv_xtx * xty;

  std::cout << "Estimated line coefficients: \n"
            << beta << "\n";

  // predict
  Matrix new_x = {{1, 1},
                  {1, 2},
                  {1, 3},
                  {1, 4},
                  {1, 5}};

  auto line_coeffs = blaze::expand(blaze::row<0UL>(blaze::trans(beta)), new_x.rows());
  auto new_y_norm = new_x % line_coeffs;
  std::cout << "Predicted(norm) values : \n"
            << new_y_norm << std::endl;

  return 0;
};

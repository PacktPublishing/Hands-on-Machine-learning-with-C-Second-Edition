#include <blaze/Math.h>
#include <iostream>
#include "blaze/math/TransposeFlag.h"
#include "blaze/math/dense/UniformVector.h"
#include "blaze/math/expressions/Forward.h"
#include "blaze/math/sparse/IdentityMatrix.h"
#include "blaze/util/Random.h"

typedef blaze::StaticMatrix<float, 3UL, 3UL, blaze::columnMajor> MyMatrix33f;
typedef blaze::StaticVector<float, 3UL> MyVector3f;
typedef blaze::DynamicMatrix<double> MyMatrix;

int main() {
  {
    // declaration
    MyMatrix33f a;
    MyVector3f v;
    MyMatrix m(10, 15);

    // initialization
    a = blaze::zero<float>(3UL, 3UL);
    std::cout << "Zero matrix:\n"
              << a << std::endl;

    a = blaze::IdentityMatrix<float>(3UL);
    std::cout << "Identity matrix:\n"
              << a << std::endl;

    blaze::Rand<float> rnd;
    v = blaze::generate(3UL, [&](size_t) { return rnd.generate(); });
    std::cout << "Random vector:\n"
              << v << std::endl;

    a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::cout << "Initializer list matrix:\n"
              << a << std::endl;

    a = blaze::uniform(3UL, 3UL, 3.f);
    std::cout << "Uniformly initialized matrix:\n"
              << a << std::endl;

    a(0, 0) = 7;
    std::cout << "Matrix with changed element[0][0]:\n"
              << a << std::endl;

    std::array<int, 4> data = {1, 2, 3, 4};
    blaze::CustomVector<int, blaze::unaligned, blaze::unpadded, blaze::rowMajor> v2(data.data(), data.size());
    data[1] = 5;
    std::cout << "Vector mapped to array:\n"
              << v2 << std::endl;

    std::vector<float> mdata = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded, blaze::rowMajor> a2(mdata.data(), 3UL, 3UL);
    std::cout << "Matrix mapped to array:\n"
              << a2 << std::endl;
  }
  // arithmetic
  {
    blaze::StaticMatrix<float, 2UL, 2UL> a = {{1, 2}, {3, 4}};
    auto b = a;

    // element wise operations
    blaze::StaticMatrix<float, 2UL, 2UL> result = a % b;
    std::cout << "element wise a * b :\n"
              << result << std::endl;

    a = b * 4;
    std::cout << "element wise a = b * 4 :\n"
              << a << std::endl;

    // matrix operations
    result = a + b;
    std::cout << "matrices a + b :\n"
              << result << std::endl;

    a += b;
    std::cout << "matrices a += b :\n"
              << a << std::endl;

    result = a * b;
    std::cout << "matrices a * b :\n"
              << result << std::endl;
  }

  // patial access
  {
    blaze::StaticMatrix<float, 4UL, 4UL> m = {{1, 2, 3, 4},
                                              {5, 6, 7, 8},
                                              {9, 10, 11, 12},
                                              {13, 14, 15, 16}};
    std::cout << "4x4 matrix :\n"
              << m << std::endl;

    blaze::StaticMatrix<float, 2UL, 2UL> b =
        blaze::submatrix<1UL, 1UL, 2UL, 2UL>(m);  // coping the middle part of matrix
    std::cout << "Middle of 4x4 matrix :\n"
              << b << std::endl;

    blaze::submatrix<1UL, 1UL, 2UL, 2UL>(m) *= 0;  // change values in original matrix
    std::cout << "Modified middle of 4x4 matrix :\n"
              << m << std::endl;

    blaze::row<1UL>(m) += 3;
    std::cout << "Modified row of 4x4 matrix :\n"
              << m << std::endl;

    blaze::column<2UL>(m) /= 4;
    std::cout << "Modified col of 4x4 matrix :\n"
              << m << std::endl;
  }

  // broadcasting (There is no implicit brodcasting in Blaze )
  {
    blaze::DynamicMatrix<float, blaze::rowVector> mat = blaze::uniform(4UL, 4UL, 2);
    std::cout << "Uniform 4x4 matrix :\n"
              << mat << std::endl;

    blaze::DynamicVector<float, blaze::rowVector> vec = {1, 2, 3, 4};
    auto ex_vec = blaze::expand(vec, 4UL);  // no allocation, gives a proxy object
    std::cout << "expanded vector :\n"
              << ex_vec << std::endl;

    mat += ex_vec;
    std::cout << "Sum broadcasted over rows :\n"
              << mat << std::endl;
  }
  return 0;
};

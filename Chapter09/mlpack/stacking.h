#pragma once

#include <mlpack/core.hpp>

void StackingClassification(size_t num_classes,
                            const arma::mat& train_input,
                            const arma::Row<size_t>& train_labels,
                            const arma::mat& test_input,
                            const arma::Row<size_t>& test_labels);
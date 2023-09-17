#pragma once
#include <string>
#include <tapkee/tapkee.hpp>

tapkee::DenseMatrix read_data(const std::string& file_name, char delimiter);

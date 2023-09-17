#include <fstream>
#include "util.h"

// The code was taken from the CLI util of the Tapkee library

tapkee::DenseMatrix read_data(const std::string& file_name, char delimiter) {
  using namespace std;
  string str;
  vector<vector<tapkee::ScalarType>> input_data;
  ifstream ifs(file_name);
  while (ifs) {
    getline(ifs, str);

    istringstream ss(str);
    if (str.size()) {
      vector<tapkee::ScalarType> row;
      while (ss) {
        string value_string;
        if (!getline(ss, value_string, delimiter))
          break;
        istringstream value_stream(value_string);
        tapkee::ScalarType value;
        if (value_stream >> value)
          row.push_back(value);
      }
      input_data.push_back(row);
    }
  }

  if (!input_data.empty()) {
    tapkee::DenseMatrix fm(input_data.size(), input_data[0].size());
    for (int i = 0; i < fm.rows(); i++) {
      if (static_cast<tapkee::DenseMatrix::Index>(input_data[i].size()) != fm.cols()) {
        stringstream ss;
        ss << "Wrong data at line " << i;
        throw std::runtime_error(ss.str());
      }
      for (int j = 0; j < fm.cols(); j++)
        fm(i, j) = input_data[i][j];
    }
    return fm;
  } else {
    return tapkee::DenseMatrix(0, 0);
  }
}

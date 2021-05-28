#pragma once

#include <Eigen/Dense>

#include <fstream>
#include <sstream>

namespace nn::util {

template<std::size_t nBatch>
std::tuple<Eigen::Matrix<float, nBatch, 784>, Eigen::Matrix<float, nBatch, 1>> getMNISTImageBatch(std::ifstream& data) {

	std::vector<float> values;
	std::vector<float> labels;
	for (auto i(0ul); i < nBatch; ++i) {
		std::string line;
		std::getline(data, line);
		std::stringstream stream(line);
		std::string item;
		std::getline(stream, item, ',');
		labels.push_back(std::stof(item));
		for (; std::getline(stream, item, ',');) {
			values.push_back(std::stof(item));
		}
	}

	if (values.size() != nBatch * 784 or labels.size() != nBatch) {
		std::cout << "Wrong number of values: " << values.size() << std::endl;
		return std::make_tuple(Eigen::Matrix<float, nBatch, 784>::Zero(), Eigen::Matrix<float, nBatch, 1>::Zero());
	}

	return std::make_tuple(Eigen::Map<Eigen::Matrix<float, nBatch, 784>>(values.data(), nBatch, 784),
	                       Eigen::Map<Eigen::Matrix<float, nBatch, 1>>(labels.data(), nBatch, 1));
}

} // namespace nn::util

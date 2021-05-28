#pragma once

#include <type_traits>
#include <Eigen/Dense>

namespace nn::activation {

class Sigmoid {
public:
	template<typename X>
	auto static activate(X const& x) {
		return 1.f / (1.f + Eigen::exp(-x.array()));
	}

	template<typename X>
	auto static derivative(X const& x) {
		return x.array() * (1.f - x.array());
	}
};

} // namespace nn::activation

#pragma once

#include <type_traits>
#include <Eigen/Dense>

namespace net::activation {

class Activation {
public:
	//template<typename Layer>
	//typename Layer::Output operator()(typename Layer::Input const& inputVector,
	//                                  typename Layer::Weights const& weightMatrix,
	//                                  typename Layer::Scalar const bias) const;

	//template<typename InputLayer, typename WeightMatrix, typename Bias>
	//auto operator()(InputLayer const& input,
	//                WeightMatrix const& weights,
	//                float const bias) const override {
};

class Sigmoid : public Activation {
public:
	template<typename InputLayer, typename WeightMatrix>
	auto static activate(InputLayer const& input,
	                     WeightMatrix const& weights,
	                     float const bias) {
		return bias + 1.f / (1.f + Eigen::exp(-(weights * input).array()));
	}

	template<typename OutputLayer>
	auto static derivative(OutputLayer const& output) {
		return output.array() * (1.f - output).array();
	}
};

} // namespace net::activation

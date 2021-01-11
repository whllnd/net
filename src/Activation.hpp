#pragma once

#include <type_traits>
#include <Eigen/Dense>

namespace net {
namespace activation {

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
	auto operator()(InputLayer const& input,
	                WeightMatrix const& weights,
	                float const bias) const {
		return bias + 1.f / (1.f + Eigen::exp(-(weights * input).array()));
	}

};

} // namespace activation
} // namespace net

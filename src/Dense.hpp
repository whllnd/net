#pragma once

#include <type_traits>
#include <Eigen/Dense>

#include "Activation.hpp"
#include "Layer.hpp"

namespace net::layer {

template<std::size_t nInput, std::size_t nOutput, typename Activation>
class Dense : public Layer {
	using Scalar = float;
	using Input = typename Eigen::Matrix<Scalar, nInput, 1>;
	using Output = typename Eigen::Matrix<Scalar, nOutput, 1>;
	using Weights = typename Eigen::Matrix<Scalar, nOutput, nInput>;

public:
	Dense(Scalar const bias=0.f)
		: mBias{bias}
		, mWeights{Weights::Random()}
		, mActivation{Output::Zero()}
	{}

	// @todo: std::enable_if to distinguish simple forwarding from learning
	auto fwd(Input input) {
		mActivation = static_cast<Output>(Activation::activate(input, mWeights, mBias));
		return mActivation;
	}

	auto backprop(Output output) {
		return output;
	}

	// In case of output layer
	// @todo: std::enable_if to distinguish output layer from hidden layers
	auto errorForOutputLayer(Output const& expectation) {
		return (expectation - mActivation) * Activation::derivate(mActivation);
	}

	auto errorForHiddenLayer(Output const& output)
	{
		return 1;
	}

	//template<typename Error>
	//auto backprop(Error&& error)

	Weights const& getWeights() const { return mWeights; }

private:
	Scalar mBias;
	Weights mWeights;
	Output mActivation;
};

} // namespace net::layer

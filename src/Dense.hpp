#pragma once

#include <type_traits>
#include <Eigen/Dense>

#include "Activation.hpp"
#include "Layer.hpp"

namespace net {
namespace layer {

template<std::size_t nInput, std::size_t nOutput, typename Activation>
class Dense : public Layer {
	using Scalar = float;
	using Input = typename Eigen::Matrix<Scalar, nInput, 1>;
	using Output = typename Eigen::Matrix<Scalar, nOutput, 1>;
	using Weights = typename Eigen::Matrix<Scalar, nOutput, nInput>;

public:
	Dense(Scalar const bias=0.f)
		: mBias{bias}
	{
		mWeights = Weights::Random();
	}

	//Dense(Dense&& dense)
	//	: mActivation{/*std::move(*/dense.mActivation/*)*/}
	//	, mBias{dense.mBias}
	//	, mWeights{std::move(dense.mWeights)}
	//{}

	template<typename InputLayer>
	auto fwd(InputLayer input) const {
		return Output{mActivation(input, mWeights, mBias)};
	}

	Weights const& getWeights() const { return mWeights; }

private:
	Activation mActivation;
	Scalar mBias;
	Weights mWeights;
};

} // namespace layer
} // namespace net

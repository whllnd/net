#pragma once

#include <type_traits>
#include <Eigen/Dense>

#include "Activation.hpp"
#include "Layer.hpp"

namespace nn::layer {

template<std::size_t nInput, std::size_t nOutput, std::size_t nBatch=1, typename Activation=activation::Sigmoid>
class Dense {
public:
	using Scalar = float;
	using Input = typename Eigen::Matrix<Scalar, nInput, nBatch>;
	using Output = typename Eigen::Matrix<Scalar, nOutput, nBatch>;
	using Weights = typename Eigen::Matrix<Scalar, nOutput, nInput>;
	using Activate = Activation; // @todo: More elegant solution ...

	//Dense(Output const& bias=Output::Zero())
	//	: mWeights{Weights::Random()} // @todo: Make it cool
	//	, mBias{bias}
	//{}
	Dense(Eigen::MatrixXf const& bias=Eigen::MatrixXf::Zero(nOutput, nBatch))
		: mWeights{Eigen::MatrixXf::Random(nOutput, nInput)}
		, mBias{bias}
	{}

	//auto fwd(Input const& input) {
	Eigen::MatrixXf fwd(Eigen::MatrixXf const& input) {
		return Activation::activate(mBias.array() + (mWeights * input).array());
	}

	//Output const& bias() const { return mBias; }
	//Output& bias() { return mBias; }
	Eigen::MatrixXf const& bias() const { return mBias; }
	Eigen::MatrixXf& bias() { return mBias; }

	//Weights const& weights() const { return mWeights; }
	//Weights& weights() { return mWeights; }
	Eigen::MatrixXf const& weights() const { return mWeights; }
	Eigen::MatrixXf& weights() { return mWeights; }

private:
	//Weights mWeights;
	//Output mBias;
	Eigen::MatrixXf mWeights;
	Eigen::MatrixXf mBias;
};

} // namespace nn::layer

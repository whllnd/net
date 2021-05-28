#pragma once

#include <type_traits>

namespace nn {

// Net itself based on std::tuple-esque recursion -----------------------------
template<typename... Layers>
class Net{};

template<typename Layer, typename... Layers>
class Net<Layer, Layers...> : public Net<Layers...> {
public:
	Net(Layer layer, Layers... layers)
		: Net<Layers...>(layers...)
		, mLayer{layer}
	{}

	Layer const& layer() const { return mLayer; }
	Layer& layer() { return mLayer; }

private:
	Layer mLayer;
};

// Enable automatic template deduction (I have no idea why this works ...)
template<typename... Layers>
Net(Layers... layers) -> Net<Layers...>;

// Layer access ---------------------------------------------------------------
template<std::size_t, typename>
struct ElemType;

template<typename Layer, typename... Layers>
struct ElemType<0, Net<Layer, Layers...>> {
	typedef Layer type;
};

template<std::size_t i, typename Layer, typename... Layers>
struct ElemType<i, Net<Layer, Layers...>> {
	typedef typename ElemType<i - 1, Net<Layers...>>::type type;
};

template<std::size_t i, typename Layer, typename... Layers>
auto get(Net<Layer, Layers...>& net) {
	if constexpr (i) {
		Net<Layers...>& base(net);
		return get<i - 1>(base);
	} else {
		return net.layer();
	}
}

// Train ----------------------------------------------------------------------
template<typename Net, typename X, typename Y>
void train(Net& net, X const& x, Y const& y, std::size_t const nIter) {
	for (std::size_t i(0); i < nIter; ++i) {
		std::cout << "---" << std::endl;
		std::cout << "labels: " << y.transpose() << std::endl;
		train(net, x.transpose(), y.transpose());
	}
}

template<typename X, typename Y, typename Layer, typename... Layers>
auto train(Net<Layer, Layers...>& net, X const& x, Y const& y) {
	// Layer specific types
	//using Input = typename Layer::Input;
	//using Output = typename Layer::Output;
	using Activation = typename Layer::Activate;

	// Forward pass
	auto& layer(net.layer());
	//Output const input(layer.bias().array() + (layer.weights() * x).array());
	//Output const activation(Activation::activate(input));
	Eigen::MatrixXf const input(layer.bias().array() + (layer.weights() * x).array());
	Eigen::MatrixXf const activation(Activation::activate(input));

	// Backpropagation
	//Output delta;
	Eigen::MatrixXf delta;
	if constexpr (sizeof...(Layers)) {
		Net<Layers...>& next(net);
		delta = train(next, activation, y).array() * Activation::derivative(activation).array();
	} else {
		delta = (activation - y).array() * Activation::derivative(activation).array();
		std::cout << "output: " << activation << std::endl;
		std::cout << "delta:  " << delta.array().sum() << std::endl;
	}

	//Input wd(layer.weights().transpose() * delta);
	Eigen::MatrixXf wd(layer.weights().transpose() * delta);
	layer.weights() = layer.weights().array() - 1.f * (delta * x.transpose()).array();
	return wd;
}

// Simple forward pass --------------------------------------------------------
template<typename Input, typename Layer, typename... Layers>
auto fwd(Net<Layer, Layers...>& net, Input const& input) {
	Net<Layers...>& next(net);
	if constexpr (sizeof...(Layers)) {
		return fwd(next, net.layer().fwd(input));
	}
	return net.layer().fwd(input);
}

} // namespace nn

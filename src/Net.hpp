#pragma once

#include "Layer.hpp"

#include <type_traits>

namespace net {

enum class LayerType {
	Hidden,
	Output
};

// @todo: Get rid of code duplicates
template<typename... Layers>
class Net{};

template<typename Layer>
class Net<Layer> : public Net<> {
public:
	Net(Layer layer)
		: Net<>()
		, mLayer{layer}
	{}

	Layer const& layer() const { return mLayer; }
	Layer& layer() { return mLayer; }
	LayerType type() const { return mType; }

private:
	Layer mLayer;
	LayerType mType{LayerType::Output};
};

template<typename Layer, typename... Layers>
class Net<Layer, Layers...> : public Net<Layers...> {
public:
	Net(Layer layer, Layers... layers)
		: Net<Layers...>(layers...)
		, mLayer{layer}
	{}

	Layer const& layer() const { return mLayer; }
	Layer& layer() { return mLayer; }
	LayerType type() const { return mType; }

private:
	Layer mLayer;
	LayerType mType{LayerType::Hidden};
};

// Enable automatic template deduction (I have no idea why this works ...)
template<typename... Layers>
Net(Layers... layers) -> Net<Layers...>;

// Layer access
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

template<std::size_t i, typename... Layers>
typename std::enable_if<0 == i, typename ElemType<i, Net<Layers...>>::type&>::type get(Net<Layers...>& net) {
	return net.layer(); //mLayer;
}

template<std::size_t i, typename Layer, typename... Layers>
typename std::enable_if<0 != i, typename ElemType<i, Net<Layer, Layers...>>::type&>::type get(Net<Layer, Layers...>& net) {
	Net<Layers...>& base(net);
	return get<i - 1>(base);
}

// Forward pass
// @todo: Add const qualifiers once I figured out how to distinguish
// simple forwarding from learning process
template<typename Input, typename Layer>
auto fwd(Net<Layer>& net, Input const& input) {
	std::cout << "fwd<>    Type: " << static_cast<int>(net.type()) << std::endl;
	return net.layer().fwd(input);
}

template<typename Input, typename Layer, typename... Layers>
auto fwd(Net<Layer, Layers...>& net, Input const& input) {
	std::cout << "fwd<...> Type: " << static_cast<int>(net.type()) << std::endl;
	Net<Layers...>& next(net);
	return fwd(next, net.layer().fwd(input));
}

// Backpropagation
template<typename Output, typename Layer>
auto backprop(Net<Layer>& net, Output const& output) {
	std::cout << "backprop<>    Type: " << static_cast<int>(net.type()) << std::endl;
	return net.layer().backprop(output);
}

template<typename Output, typename Layer, typename... Layers>
auto backprop(Net<Layer, Layers...>& net, Output const& output) {
	std::cout << "backprop<...> Type: " << static_cast<int>(net.type()) << std::endl;
	Net<Layers...>& next(net);
	return net.layer().backprop(backprop(next, output));
}

} // namespace net

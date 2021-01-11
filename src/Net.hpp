#pragma once

#include "Layer.hpp"

#include <type_traits>

namespace net {

template<typename... Layers>
class Net{};

template<typename Layer, typename... Layers>
class Net<Layer, Layers...> : public Net<Layers...> {
public:
	//Net(Layer&& layer, Layers... layers)
	//	: Net<Layers...>(layers...)
	//	, mLayer{std::move(layer)}
	//{}

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

// Layer access
template<std::size_t, typename>
struct LayerType;

template<typename Layer, typename... Layers>
struct LayerType<0, Net<Layer, Layers...>> {
	typedef Layer type;
};

template<std::size_t i, typename Layer, typename... Layers>
struct LayerType<i, Net<Layer, Layers...>> {
	typedef typename LayerType<i - 1, Net<Layers...>>::type type;
};

template<std::size_t i, typename... Layers>
typename std::enable_if<0 == i, typename LayerType<i, Net<Layers...>>::type&>::type get(Net<Layers...>& net) {
	return net.layer(); //mLayer;
}

template<std::size_t i, typename Layer, typename... Layers>
typename std::enable_if<0 != i, typename LayerType<i, Net<Layer, Layers...>>::type&>::type get(Net<Layer, Layers...>& net) {
	Net<Layers...>& base(net);
	return get<i - 1>(base);
}

// Forward pass
template<typename Input, typename Layer>
auto fwd(Net<Layer> const& net, Input const& input) {
	return net.layer().fwd(input);
}

template<typename Input, typename Layer, typename... Layers>
auto fwd(Net<Layer, Layers...> const& net, Input const& input) {
	Net<Layers...> const& next(net);
	return fwd(next, net.layer().fwd(input));
}

} // namespace net

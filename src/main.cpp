#include <iostream>
#include "Dense.hpp"
#include "Net.hpp"

int main() {

	net::Net network(net::layer::Dense<3, 4, net::activation::Sigmoid>{},
	                 net::layer::Dense<4, 6, net::activation::Sigmoid>{},
	                 net::layer::Dense<6, 3, net::activation::Sigmoid>{});

	Eigen::Vector3f input;
	input << 1.f, 2.f, 3.f;

	auto output(net::fwd(network, input));
	std::cout << "output:" << std::endl;
	std::cout << output << std::endl;

	//auto backprop(net::backprop(network, output));
	//std::cout << "backprop:" << std::endl;
	//std::cout << backprop << std::endl;

	//auto l0(net::get<0>(network));
	//auto l1(net::get<1>(network));

	//auto o0(l0.forwardPass(input));
	//auto o1(l1.forwardPass(o0));
	//std::cout << o0 << std::endl;

	//auto layer(net::get<0>(network));
	//auto output(layer.forwardPass(input));
	//std::cout << "output:" << std::endl;
	//std::cout << output << std::endl;
	//std::cout << network.numLayers() << std::endl;

	//auto l1(net::get<0>(network));
	//auto l2(net::get<1>(network));
	//auto l3(net::get<2>(network));
	//std::cout << l1.forwardPass(input) << std::endl;

	//auto out(net::forwardPass<3>(network, input));
	//std::cout << out << std::endl;

	//Input input;
	//input << 1, 2, 3;

	//Output output = dense.forwardPass(input);
	//std::cout << output << std::endl;

	//net::Net<typename net::layer::Dense<Input, Output, typename net::activation::Sigmoid<Input, Output>>, 2> net{d1, d2};
	//Layer<4>(net::layer::type::Dense)};
	//
	//std::cout << net.getLayers().size() << std::endl;

	//net::Net<Dense<Input, Output>, Dense<Output, Input>

	//auto output(net.pass(input));
}

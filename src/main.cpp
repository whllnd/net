#include <iostream>
#include "Dense.hpp"
#include "Net.hpp"
#include "Util.hpp"

int main() {

	//std::size_t constexpr batch(2);
	//std::size_t constexpr nIter(5);
	//nn::Net net(nn::layer::Dense<3, 4, batch>{},
	//            nn::layer::Dense<4, 4, batch>{},
	//            nn::layer::Dense<4, 1, batch>{});

	//Eigen::Matrix<float, batch, 3> input;
	//input << 1.f, 2.f, 3.f,
	//         2.f, 1.f, 3.f;

	//Eigen::Matrix<float, batch, 1> labels;
	//labels << 1.f,
	//          2.f;

	//nn::train(net, input, labels, nIter);

	std::size_t constexpr batch(25); // @todo: Make it sane
	std::size_t constexpr nIter(1);
	nn::Net net(nn::layer::Dense<784, 256, batch>{},
	            nn::layer::Dense<256, 128, batch>{},
	            nn::layer::Dense<128, 32, batch>{},
	            nn::layer::Dense<32, 1, batch>{});

	for (auto i(0); i < 50; ++i) {
		std::ifstream file("/home/per/Code/net/data/mnist_test.csv");
		for (auto i(0); i < 200; ++i) {
			auto [data, labels](nn::util::getMNISTImageBatch<batch>(file));
			data = data.array() / 255.f;
			labels = labels.array() / 9.f;
			nn::train(net, data, labels, nIter);
		}
	}
}

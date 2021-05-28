#pragma once

#include <Eigen/Dense>
#include <memory>

namespace nn::layer {

class Layer {
public:
	//template<typename Input, typename Output>
	//Output forwardPass(Input const& v) { return  {}; }
	virtual Eigen::VectorXf forwardPass(Eigen::VectorXf const& v) const {
		return {};
	}

	//template<typename InputVector, typename OutputVector>
	//OutputVector backPropagate(Vector const& v) = 0;
};

//class Layer {
//public:
//	template <typename T>
//	Layer(T const& layer)
//		: mLayer(std::make_shared<Model<T>>(std::move(layer)))
//	{}
//
//	Eigen::VectorXf forwardPass(Eigen::VectorXf const& input) const {
//		return mLayer->forwardPass(input);
//	}
//
//	struct Concept {
//		virtual ~Concept() {}
//		virtual Eigen::VectorXf forwardPass(Eigen::VectorXf const& input) const = 0;
//	};
//
//	template< typename T >
//	struct Model : Concept {
//		Model(const T& t) : object(t) {}
//		Eigen::VectorXf forwardPass(Eigen::VectorXf const& input) const override {
//			return object.forwardPass(input);
//		}
//	private:
//		T object;
//	};
//
//	std::shared_ptr<Concept const> mLayer;
//};

} // namespace nn::layer

#pragma once
#include <math.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#include "util.h"

enum ActivationFunctionType{Tanh=0, Sigmoid};

class ActivationFunction{
	private:
		ActivationFunctionType type;
	public:
		ActivationFunction(ActivationFunctionType a=Tanh):
			type(a){

		}

		VectorXd operator()(VectorXd x){
			for(uint i=0;i<x.size();i++){
				switch(type){
					case Tanh:
						x(i) = tanh(x(i));
						break;
					case Sigmoid:
						x(i) = sigmoid(x(i));
						break;
					default:
						HALT("Function type is not supported");
						break;
				}
			}
			return x;
		}

		VectorXd ds(VectorXd x){
			for(uint i=0;i<x.size();i++){
				switch(type){
					case Tanh:
						x(i) = 1.0-tanh(x(i))*tanh(x(i));
						break;
					case Sigmoid:
						x(i) = sigmoid(x(i))*(1.0-sigmoid(x(i)));
						break;
					default:
						HALT("Function type is not supported");
						break;
				}
			}
			return x;
		}
	private:

		double sigmoid(double x){
			return 1.0/(1.0+exp(-x));
		}

};


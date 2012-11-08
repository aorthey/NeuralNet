#pragma once
#include <math.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#include "util.h"

enum ActivationFunctionType{Tanh=0, Sigmoid};

class ActivationFunction{
	public:
		ActivationFunction(ActivationFunctionType a):
			type(a){

		}
		VectorXd operator()(VectorXd x){
			for(uint i=0;i<x.size();i++){
				x(i) = (*this)((double)x(i));
			}
			return x;
		}
		double operator()(double x){
			switch(type){
				case Tanh:
					return tanh(x);
					break;
				case Sigmoid:
					return sigmoid(x);
					break;
				default:
					HALT("Function type is not supported");
					break;
			}
		}
		VectorXd ds(VectorXd x){
			for(uint i=0;i<x.size();i++){
				x(i) = this->ds((double)x(i));
			}
			return x;
		}
		double ds(double x){
			switch(type){
				case Tanh:
					return 1.0-tanh(x)*tanh(x);
					break;
				case Sigmoid:
					return sigmoid(x)*(1.0-sigmoid(x));
					break;
				default:
					HALT("Function type is not supported");
					break;
			}
		}
	private:

		double sigmoid(double x){
			return 1.0/(1.0+exp(-x));
		}

	private:
		ActivationFunctionType type;
};


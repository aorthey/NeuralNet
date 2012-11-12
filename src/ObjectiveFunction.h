#pragma once
#include <Eigen/Dense>
#include "ActivationFunction.h"
#include "NeuralNetworkState.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;


struct ObjectiveFunction{
	public:
		double operator()(NeuralNetworkState &state, VectorXd label_y){

			//residual between output and target
			VectorXd residual = state.output() - label_y;
			
			//weight decay and squared loss
			double e = residual.squaredNorm();

			return e;
		}
		VectorXd ds(NeuralNetworkState &state, VectorXd label_y, ActivationFunction &a){
			//squared loss derivative
			VectorXd x = state.output();

			//dE/dtheta * dtheta/ds

			double dE_dt = (x-label_y).norm();
				
			VectorXd ones = VectorXd::Constant(x.size(),1);
			VectorXd dE_ds = dE_dt * x *(ones-x);

			return dE_ds;

		}
};


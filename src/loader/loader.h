#pragma once
#include <vector>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace loader{
	class Loader{

	public:
		virtual void get_training_data(std::vector<VectorXd> &samples)=0;

	};
};

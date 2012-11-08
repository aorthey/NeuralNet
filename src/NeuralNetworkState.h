#pragma once
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

//current state, including weights, biases and derivations of the activation
//function in each layer
struct NeuralNetworkState{
	public:
		std::vector<MatrixXd> W;
		std::vector<VectorXd> b;
		std::vector<VectorXd> x;
		std::vector<VectorXd> x_ds;

		std::vector<VectorXd> delta;

		double error;
		double error_ds;

		unsigned int L_hidden_layers;
		unsigned int N_input_neurons;
		unsigned int N_hidden_neurons_per_layer;
		unsigned int N_output_neurons;

	public:
		uint get_L_hidden_layers(){ return L_hidden_layers; }
		uint get_N_output_neurons(){ return N_output_neurons; }
		VectorXd output(){ return x.at(L_hidden_layers+1); }
		void setOutputDelta(VectorXd d){ delta.at(L_hidden_layers+1) = d;}

	NeuralNetworkState(uint N_input_neurons, uint N_output_neurons, uint L_hidden_layers, uint N_hidden_neurons_per_layer): 
		N_input_neurons(N_input_neurons),
		N_output_neurons(N_output_neurons), 
		L_hidden_layers(L_hidden_layers),
		N_hidden_neurons_per_layer(N_hidden_neurons_per_layer){
		
			assert(L_hidden_layers>=1);
			assert(N_hidden_neurons_per_layer>=1);

			W.at(0)= MatrixXd::Random(N_hidden_neurons_per_layer, N_input_neurons);
			b.at(0)= VectorXd::Random(N_hidden_neurons_per_layer);
			for(uint l=1;l<L_hidden_layers;l++){
				W.at(l)= MatrixXd::Random(N_hidden_neurons_per_layer, N_hidden_neurons_per_layer);
				b.at(l)= VectorXd::Random(N_hidden_neurons_per_layer);
			}
			W.at(L_hidden_layers)= MatrixXd::Random(N_output_neurons, N_hidden_neurons_per_layer);
			b.at(L_hidden_layers)= VectorXd::Random(N_output_neurons);
		
		};

};

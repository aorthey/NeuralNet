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
		VectorXd output(){ return x.back(); }
		VectorXd output_xds(){ return x_ds.back(); }
		VectorXd output_delta(){ return delta.back(); }
		void setOutputDelta(VectorXd d){ delta.at(L_hidden_layers) = d;}

	NeuralNetworkState(uint N_input_neurons, uint N_output_neurons, uint L_hidden_layers, uint N_hidden_neurons_per_layer): 
		N_input_neurons(N_input_neurons),
		N_output_neurons(N_output_neurons), 
		L_hidden_layers(L_hidden_layers),
		N_hidden_neurons_per_layer(N_hidden_neurons_per_layer){
		
			assert(L_hidden_layers>=1);
			assert(N_hidden_neurons_per_layer>=1);

			W.push_back( MatrixXd::Random(N_hidden_neurons_per_layer, N_input_neurons) );
			b.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );
			delta.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );

			x.push_back( VectorXd::Random(N_input_neurons) );
			x_ds.push_back( VectorXd::Random(N_input_neurons) );
			for(uint l=1;l<L_hidden_layers+1;l++){
				x.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );
				x_ds.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );
			}

			for(uint l=1;l<L_hidden_layers;l++){
				W.push_back( MatrixXd::Random(N_hidden_neurons_per_layer, N_hidden_neurons_per_layer) );
				b.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );
				delta.push_back( VectorXd::Random(N_hidden_neurons_per_layer) );
			}
			W.push_back( MatrixXd::Random(N_output_neurons, N_hidden_neurons_per_layer) );
			b.push_back( VectorXd::Random(N_output_neurons));
			delta.push_back( VectorXd::Random(N_output_neurons) );

			x.push_back( VectorXd::Random(N_output_neurons) );
			x_ds.push_back( VectorXd::Random(N_output_neurons) );

			PRINT("Neural Network -- "<<N_input_neurons << " input units -- " 
					<< N_output_neurons << " output units -- " << L_hidden_layers << " hidden layer");
		
		};


	void print(){
		for(uint i=0;i<x.size();i++){
			PRINT(" layer "<<i<<": " << x.at(i).size() << " units");
		}
	}

};

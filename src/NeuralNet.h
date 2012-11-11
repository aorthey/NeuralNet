#pragma once
#include <string>
#include <vector>
#include <assert.h>
#include <Eigen/Dense>
#include "ActivationFunction.h"
#include "ObjectiveFunction.h"
#include "NeuralNetworkState.h"
#include "NeuralNetworkDisplayUnit.h"
#include "loader/loader.h"
#include "loader/mnist.h"
using std::string;
using Eigen::MatrixXd;
using Eigen::VectorXd;


//monitoring the optimization process by checking the gradient
struct NeuralNetworkDebugUnit{
	void check_gradient_computation(){

	}

};

//doing optimization on weights, based on SGD,B-SGD
struct NeuralNetworkComputationUnit{

	ActivationFunction *theta;
	ObjectiveFunction *loss;

	void clean(){
		delete theta;
		delete loss;
	}

	void init(){
		theta = new ActivationFunction(Tanh);
		loss = new ObjectiveFunction();
	}

	void forward_propagation(NeuralNetworkState &state, VectorXd &input, VectorXd label_y){
		uint L = state.get_L_hidden_layers();

		state.x.at(0) = input;

		for(uint i=0;i<L+1;i++){
			VectorXd s_i = state.W.at(i)*state.x.at(i) + state.b.at(i);
			state.x.at(i+1) = (*theta)(s_i);
			state.x_ds.at(i+1) = theta->ds(s_i);
		}

		state.error = (*loss)(state, label_y);

		//delta(L) = dE/ds
		VectorXd error_ds = loss->ds(state, label_y, *theta);
		state.setOutputDelta(error_ds);
	
	}

	void batch_stochastic_gradient_descent(std::vector<VectorXd> &training_samples, std::vector<VectorXd> &label, uint batch_size){


	}

	void back_propagation(NeuralNetworkState &state){
		double tau = 0.01;
		uint L = state.L_hidden_layers;

		//assume, that delta.at(L+1) is set by the forward propagation
		for(uint l=L;l>=0;l--){
			state.delta.at(l) = state.x_ds.at(l).asDiagonal() * (state.W.at(l+1) * state.delta.at(l+1));
			MatrixXd dW = state.delta.at(l)*state.x.at(l-1);

			state.W.at(l) = state.W.at(l) - tau * dW;
		}

	}


};

//visualizing features in the net, showing error rates
struct NeuralNetworkOutputUnit{
	void visualize_unit(uint l_hidden_layer, uint j_hidden_unit){

	}
	void visualize_sample(uint i_sample){


	}
	void error_score_on_data_set(){

	}

};

struct NeuralNetworkSerializationUnit{

	//serialize network state to filename
	void save(string filename){

	}

	//build network state from previous saved state
	void load(string filename){

	}
};

struct NeuralNetwork{
	private:
	NeuralNetworkState *state;
	NeuralNetworkComputationUnit *computing;
	NeuralNetworkDisplayUnit *output;
	std::vector< VectorXd > training_samples;
	std::vector< VectorXd > training_labels;

	public:
	NeuralNetwork(uint N_input_neurons, uint N_output_neurons, uint L_hidden_layers, uint N_hidden_neurons_per_layer){
		state = new NeuralNetworkState(N_input_neurons, N_output_neurons, L_hidden_layers, N_hidden_neurons_per_layer);
		output = new NeuralNetworkDisplayUnit();
	}

	// get training training_samples from filename
	// filename must contain one line for each sample
	//
	//  x_01, x_02,..., x_0N, y_0
	//  x_11, x_12,..., x_1N, y:1
	//
	// whereby x_ij represents dimension j of sample i, N the number of input
	// dimensions and y the label or target of the sample
	//
	void load_training_samples(){

		loader::Loader *l = new loader::MnistLoader();

		l->get_training_data(training_samples);
		l->get_training_labels(training_labels);

		PRINT("loaded " << training_samples.size() << " training_samples");
		//loader::get_training_data();
		//oo
		//
		//
		for(uint i=0;i<10;i++){
		PRINT("label " << training_labels.at(i));
		output->visualize_sample(training_samples.at(i), 28,28);
		}

	}
	void load_test_training_samples(string filename){

	}

	void train(){

	}

	void predict(){

	}

};

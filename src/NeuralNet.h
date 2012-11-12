#pragma once
#include <string>
#include <vector>
#include <assert.h>
#include <Eigen/Dense>
#include "ObjectiveFunction.h"
#include "NeuralNetworkState.h"
#include "NeuralNetworkDisplayUnit.h"
#include "NeuralNetworkComputationUnit.h"
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

	uint L_hidden_layers;
	uint N_hidden_neurons_per_layer;

	public:
	NeuralNetwork(uint L_hidden_layers, uint N_hidden_neurons_per_layer):
		L_hidden_layers(L_hidden_layers),
		N_hidden_neurons_per_layer(N_hidden_neurons_per_layer)
	{
		assert(L_hidden_layers >= 1);
		assert(N_hidden_neurons_per_layer >= 1);

		output = new NeuralNetworkDisplayUnit();
		state = NULL; //is constructed, when we load the training samples
		computing = new NeuralNetworkComputationUnit();
	}

	void load_training_samples(){

		loader::Loader *l = new loader::MnistLoader();

		l->get_training_data(training_samples);
		l->get_training_labels(training_labels);

		uint N_input_neurons = training_samples.at(0).size();
		uint N_output_neurons = training_labels.at(0).size();
		state = new NeuralNetworkState(N_input_neurons, N_output_neurons, L_hidden_layers, N_hidden_neurons_per_layer);


	}

	void visualize(){
		for(uint i=0;i<10;i++){
			PRINT("label " << training_labels.at(i));
			output->visualize_sample(training_samples.at(i), 28,28);
		}

	}

	void train(){
		uint batch_size = 1; //SGD
		computing->stochastic_gradient_descent(*state, training_samples, training_labels);
	}
	void load_test_training_samples(string filename){

	}

	void predict(){

	}

};

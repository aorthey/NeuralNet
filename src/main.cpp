#include <iostream>
#include "util.h"
#include "NeuralNetwork.h"

int main()
{
	using std::cout;
	using std::endl;

	NeuralNetwork nn(2,10);

	nn.load_training_samples();
	nn.train();

	//nn->save("storage.dat");
	//nn->load("storage.dat");


	return 0;
}



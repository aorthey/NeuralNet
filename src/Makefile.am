CFLAGS = --pedantic -Wall -std=c99 -O2
AM_CPPFLAGS=`pkg-config eigen3 --cflags`
AM_LDFLAGS=`pkg-config eigen3 --cflags`
bin_PROGRAMS = neuralnet
neuralnet_SOURCES = NeuralNetworkState.h ActivationFunction.h main.cpp NeuralNet.h ObjectiveFunction.h util.h loader/loader.h loader/mnist.h

run:
	./$(bin_PROGRAMS)

.PHONY: run 

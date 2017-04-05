/*
 * main.cpp
 *
 *  Created on: 01.04.2017
 *      Author: kamil
 */

#include <iostream>
#include <armadillo>
#include "NeuralNetwork.hpp"

using namespace std;
using namespace arma;

int main() {

	/*
	 * TODO
	 * check:
	 * accumulateGradient
	 * gradientDescent
	 *
	 *
	 */

	NeuralNetwork nn = NeuralNetwork(3);
	nn.setLayersSizes({2, 4, 3});
	nn.trainingData.push_back(mat("0; 0"));
	nn.trainingData.push_back(mat("1; 0"));
	nn.trainingData.push_back(mat("0; 1"));
	nn.trainingData.push_back(mat("1; 1"));

	nn.trainingValues.push_back(mat("0; 0; 0"));
	nn.trainingValues.push_back(mat("1; 1; 0"));
	nn.trainingValues.push_back(mat("1; 1; 0"));
	nn.trainingValues.push_back(mat("0; 1; 1"));

	nn.randomInitialize();
	nn.propagateAllTrainingData();

	nn.gradientDescent(2000);


	if(true) {
		nn.x = mat("0; 0");
		nn.propagate();
		cout<<nn.getOutput()<<endl;

		nn.x = mat("1; 0");
		nn.propagate();
		cout<<nn.getOutput()<<endl;

		nn.x = mat("0; 1");
		nn.propagate();
		cout<<nn.getOutput()<<endl;

		nn.x = mat("1; 1");
		nn.propagate();
		cout<<nn.getOutput()<<endl;
	}


	return 0;
}


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

	NeuralNetwork nn = NeuralNetwork(3);
	nn.setLayersSizes({2, 4, 3});

	nn.theta[0] = mat(3, 4);
	nn.theta[0](0, 0)= 20;
	nn.theta[0](1, 0)= -20;
	nn.theta[0](2, 0)= -10;

	nn.theta[0](0, 1)= -20;
	nn.theta[0](1, 1)= 20;
	nn.theta[0](2, 1)= -10;

	nn.theta[0](0, 2)= 20;
	nn.theta[0](1, 2)= 0;
	nn.theta[0](2, 2)= -10;

	nn.theta[0](0, 3)= 0;
	nn.theta[0](1, 3)= 20;
	nn.theta[0](2, 3)= -10;

	nn.theta[1] = mat(5, 3);
	nn.theta[1](0, 0) = 20;
	nn.theta[1](1, 0) = 20;
	nn.theta[1](2, 0) = 0;
	nn.theta[1](3, 0) = 0;
	nn.theta[1](4, 0) = -10;

	nn.theta[1](0, 1) = 0;
	nn.theta[1](1, 1) = 0;
	nn.theta[1](2, 1) = 20;
	nn.theta[1](3, 1) = 20;
	nn.theta[1](4, 1) = -10;

	nn.theta[1](0, 2) = 0;
	nn.theta[1](1, 2) = 0;
	nn.theta[1](2, 2) = 20;
	nn.theta[1](3, 2) = 20;
	nn.theta[1](4, 2) = -30;

	nn.x = mat(2, 1);
	nn.x[0] = 0;
	nn.x[1] = 1;

	nn.y = mat(3, 1);
	nn.y[0] = 1;
	nn.y[1] = 1;
	nn.y[2] = 0;

	nn.propagate();
	nn.backPropagate();
	cout<<nn.getOutput()<<endl;

	for(int i=0; i<nn.layers-1; i++) {
		cout<<nn.delta[i]<<endl;
	}

	return 0;
}


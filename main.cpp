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

	NeuralNetwork nn = NeuralNetwork(2);
	nn.setLayersSizes({2, 2});

	nn.theta[0] = mat(3, 2);
	nn.theta[0](0, 0)= 30;
	nn.theta[0](1, 0)= 30;
	nn.theta[0](2, 0)= -20;

	nn.theta[0](0, 1)= 20;
	nn.theta[0](1, 1)= 20;
	nn.theta[0](2, 1)= -30;

	nn.a[0] = mat(2, 1);
	nn.a[0][0] = 1;
	nn.a[0][1] = 0;
	nn.addBiasUnit(nn.a[0]);

	nn.propagate();
	cout<<nn.getOutput()<<endl;
	return 0;
}


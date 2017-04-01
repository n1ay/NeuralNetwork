/*
 * NeuralNetwork.cpp
 *
 *  Created on: 01.04.2017
 *      Author: kamil
 */

#include "NeuralNetwork.hpp"
#include <cmath>
#include <omp.h>

NeuralNetwork::NeuralNetwork(int layers): layers(layers) {
	this->layersSizes = std::vector<int>(layers);

	a = std::vector<arma::mat>(layers);
	z = std::vector<arma::mat>(layers-1);
	theta = std::vector<arma::mat>(layers-1);
}

NeuralNetwork::~NeuralNetwork() {

}

void NeuralNetwork::setLayersSizes(std::initializer_list<int> list) {
	if(list.size() != layers) {
		std::cerr<<"Inconvenience with previous initialization"<<std::endl;
		exit(-1);
	}

	layersSizes = list;
}

void NeuralNetwork::addBiasUnit(arma::mat& vector) {
	int rows = vector.n_rows;
	vector.reshape(rows+1, 1);
	vector[rows] = 1;
}

void NeuralNetwork::propagate() {
	for(int i=0; i<layers-1; i++) {
		z[i] = theta[i].t()*a[i];
		a[i+1] = sigmoid(z[i]);
		if(i+1 < layers-1)
			addBiasUnit(a[i+1]);
	}
}

double inline NeuralNetwork::sigmoid(const double number) {
	return 1.0/(1.0 + pow(M_E, -1.0*number));
}

arma::mat inline NeuralNetwork::sigmoid(const arma::mat& matrix) {
	int rows = matrix.n_rows;
	int cols = matrix.n_cols;
	arma::mat result = arma::mat(rows, cols);

	#pragma omp parallel for
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++)
			result(i, j) = sigmoid(matrix(i, j));
	}
	return result;
}

arma::mat NeuralNetwork::getOutput() {
	return a[layers-1];
}

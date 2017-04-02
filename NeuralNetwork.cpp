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
	delta = std::vector<arma::mat>(layers-1);
	gradient = std::vector<arma::mat>(layers-1);
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
	vector.resize(rows+1, 1);
	vector[rows] = 1;
}

void NeuralNetwork::propagate() {
	a[0] = x;
	addBiasUnit(a[0]);
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

	if(rows > cols) {
		#pragma omp parallel for
		for(int i=0; i<rows; i++)
			for(int j=0; j<cols; j++)
				result(i, j) = sigmoid(matrix(i, j));
	}
	else {
		#pragma omp parallel for
		for(int j=0; j<cols; j++)
			for(int i=0; i< rows; i++)
				result(i, j) = sigmoid(matrix(i, j));
	}
	return result;
}

double inline NeuralNetwork::sigmoidGradient(double number) {
	return sigmoid(number)*(1-sigmoid(number));
}

arma::mat inline NeuralNetwork::sigmoidGradient(const arma::mat& matrix) {
	return sigmoid(matrix)*(1-sigmoid(matrix));
}

arma::mat NeuralNetwork::getOutput() {
	return a[layers-1];
}

void NeuralNetwork::backPropagateError() {
	delta[layers-2] = y - a[layers-1];
	for(int i=layers-3; i>=0; i--) {
		arma::mat th = theta[i+1];
		th.resize(th.n_rows-1, th.n_cols);
		delta[i] = th*delta[i+1];
		delta[i] %= z[i];
	}
}

void NeuralNetwork::randomInitialize() {

	int previousLayerSize;
	int nextLayerSize;

	#pragma omp parallel for
	for(int i=0; i<layers-1; i++) {
		previousLayerSize = layersSizes[i];
		nextLayerSize = layersSizes[i+1];
		arma::arma_rng::set_seed_random();
		theta[i] = arma::randu<arma::mat>(previousLayerSize+1, nextLayerSize);
		theta[i]*=(init*2);
		theta[i]-=arma::mat(theta[i].n_rows, theta[i].n_cols).fill(init);
	}
}

double NeuralNetwork::costFunction() {
	double cost = 0;

	//regularization for cost function
	#pragma omp parallel
	for(int i=0; i<layers-1; i++) {
		arma::mat th = theta[i];
		th.resize(theta[i].n_rows-1, theta[i].n_cols);
		arma::square(th);
		cost += arma::accu(th);
	}
	cost *= (lambda/2);

	#pragma omp parallel for
	for(int i=0; i<trainingData.size(); i++) {
		for(int j=0; j<layersSizes[layers-1]; i++) {
			cost += (-trainingValues[i][j]*log(trainingOutput[i][j]) - (1-trainingValues[i][j])*log(1-trainingOutput[i][j]));
		}
	}
	cost /= trainingData.size();
	return cost;
}

void NeuralNetwork::accumulateGradient() {

	#pragma omp parallel for
	for(int i=0; i<layers-1; i++) {
		gradient[i] = a[i+1]*delta[i];
		arma::mat th = theta[i];
		for(int j=0; j<th.n_cols; i++)
			th(th.n_rows-1, j) = 0;
		th*=lambda;
		gradient[i] = gradient[i] + th;
		gradient[i] = gradient[i]/trainingData.size();
	}
}

void NeuralNetwork::propagateAllTrainingData() {
	for(int i=0; i<trainingData.size(); i++) {
		x = trainingData[i];
		propagate();
		trainingValues[i] = y;
	}
}

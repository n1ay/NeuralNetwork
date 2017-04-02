/*
 * NeuralNetwork.hpp
 *
 *  Created on: 01.04.2017
 *      Author: kamil
 */

#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include <vector>
#include <armadillo>

struct NeuralNetwork {

	int layers;
	std::vector<int> layersSizes;

	std::vector<arma::mat> a;
	std::vector<arma::mat> z;
	std::vector<arma::mat> theta;
	std::vector<arma::mat> delta;
	arma::mat x;
	arma::mat y;

	NeuralNetwork(int layers);
	virtual ~NeuralNetwork();

	void setLayersSizes(std::initializer_list<int> list);
	void addBiasUnit(arma::mat& vector);
	void propagate();
	void backPropagate();
	arma::mat getOutput();

	double inline static sigmoid(double number);
	arma::mat inline static sigmoid(const arma::mat& matrix);
};

#endif /* NEURALNETWORK_HPP_ */

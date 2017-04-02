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

/*
 * 1) create NN class
 * 2) setLayersSizes
 * 3) randomInitialize
 * 4) set TrainingData & TrainingValues
 * 5) propagateAllTrainingData
 * 6) ...
 *
 */

struct NeuralNetwork {

	double init = 1;
	double epsilon = 1;
	double lambda = 1;

	int layers;
	std::vector<int> layersSizes;

	std::vector<arma::mat> a;
	std::vector<arma::mat> z;
	std::vector<arma::mat> theta;
	std::vector<arma::mat> delta;
	std::vector<arma::mat> gradient;

	std::vector<arma::mat> trainingData;
	std::vector<arma::mat> trainingValues;
	std::vector<arma::mat> trainingOutput;
	arma::mat x;
	arma::mat y;

	NeuralNetwork(int layers);
	virtual ~NeuralNetwork();

	void setLayersSizes(std::initializer_list<int> list);
	void addBiasUnit(arma::mat& vector);
	void propagate();
	void propagateAllTrainingData();
	void backPropagateError();
	void accumulateGradient();
	arma::mat getOutput();
	void randomInitialize();
	double costFunction();

	double inline static sigmoid(double number);
	double inline static sigmoidGradient(double number);
	arma::mat inline static sigmoid(const arma::mat& matrix);
	arma::mat inline static sigmoidGradient(const arma::mat& matrix);
};

#endif /* NEURALNETWORK_HPP_ */

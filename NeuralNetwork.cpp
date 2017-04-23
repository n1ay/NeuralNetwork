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

	trainingData = std::vector<arma::mat>();
	trainingValues = std::vector<arma::mat>();
	trainingOutput = std::vector<arma::mat>();

	testData = std::vector<arma::mat>();
	testValues = std::vector<arma::mat>();
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
	int rows = matrix.n_rows;
		int cols = matrix.n_cols;
		arma::mat result = arma::mat(rows, cols);

		if(rows > cols) {
			#pragma omp parallel for
			for(int i=0; i<rows; i++)
				for(int j=0; j<cols; j++)
					result(i, j) = sigmoidGradient(matrix(i, j));
		}
		else {
			#pragma omp parallel for
			for(int j=0; j<cols; j++)
				for(int i=0; i< rows; i++)
					result(i, j) = sigmoidGradient(matrix(i, j));
		}
		return result;
}

arma::mat NeuralNetwork::getOutput() {
	return a[layers-1];
}

void NeuralNetwork::backPropagateError(arma::mat x, arma::mat y) {
	this -> x = x;
	propagate();
	delta[layers-2] = a[layers-1] - y;
	for(int i=layers-3; i>=0; i--) {
		arma::mat th = theta[i+1];
		th.resize(th.n_rows-1, th.n_cols);
		delta[i] = th*delta[i+1];
		delta[i] = delta[i] % sigmoidGradient(z[i]);
	}
}

void NeuralNetwork::randomInitialize() {

	int previousLayerSize;
	int nextLayerSize;

	double init = sqrt(6)/sqrt(layersSizes[0]+layersSizes[layers-1]);

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
	propagateAllTrainingData();

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
		for(int j=0; j<layersSizes[layers-1]; j++) {
			cost += singleCost(trainingValues[i][j], trainingOutput[i][j]);
		}
	}
	cost /= trainingData.size();
	return cost;
}

void NeuralNetwork::accumulateGradient() {
	for(int i=0; i<layers-1; i++) {
		gradient[i].copy_size(theta[i]);
		gradient[i].fill(0);
	}
	for(int k=0; k<trainingOutput.size(); k++) {
		backPropagateError(trainingData[k], trainingValues[k]);
		#pragma omp parallel for
		for(int i=0; i<layers-1; i++) {
			gradient[i] += a[i]*delta[i].t();
			arma::mat th = theta[i];
			for(int j=0; j<th.n_cols; j++)
				th(th.n_rows-1, j) = 0;
			th*=lambda;
			gradient[i] += th;
		}
	}
	for(int i=0; i<gradient.size(); i++)
		gradient[i] /= trainingData.size();
}

double NeuralNetwork::checkGradient(int layer, int row, int col, bool accumulate) {
	if(accumulate)
		accumulateGradient();
	double gradient = this->gradient[layer](row, col);
	std::vector<arma::mat> thetaPlus = theta;
	std::vector<arma::mat> thetaMinus = theta;
	std::vector<arma::mat> thetaTmp = theta;
	thetaPlus[layer](row, col) += epsilon;
	thetaMinus[layer](row, col) -= epsilon;
	theta = thetaPlus;
	double costPlus = costFunction();
	theta = thetaMinus;
	double costMinus = costFunction();
	theta = thetaTmp;
	return (costPlus - costMinus)/(2*epsilon);
}

void NeuralNetwork::checkGradientAll() {
	accumulateGradient();
	std::vector<arma::mat> gradientDefinition = gradient;
	for(int i=0; i<gradient.size(); i++)
		for(int j=0; j<gradient[i].n_rows; j++)
			for(int k=0; k<gradient[i].n_cols; k++)
				gradientDefinition[i](j, k) = checkGradient(i, j, k);

	for(int i=0; i<gradient.size(); i++) {
		std::cout<<gradient[i]<<std::endl;
		std::cout<<gradientDefinition[i]<<std::endl;
	}
	gradient = gradientDefinition;
}

void NeuralNetwork::propagateAllTrainingData() {
	trainingOutput.clear();
	for(int i=0; i<trainingData.size(); i++) {
		x = trainingData[i];
		propagate();
		trainingOutput.push_back(getOutput());
	}
}

void NeuralNetwork::gradientDescent(int maxIter, bool quiet) {
	double prevprevCost = INFINITY;
	double prevCost = INFINITY;
	double cost = costFunction();
	for(int i=1; i<maxIter; i++) {
		if ((prevCost - cost < epsilonStep) && (prevprevCost - prevCost < epsilonStep)) {
			if(!quiet)
				std::cout<<"Minimum found. Iteration: "<<i<<std::endl;
			return;
		}
		else {
			if(!quiet)
				std::cout<<"Cost: "<<cost<<std::endl;
			prevprevCost = prevCost;
			prevCost = cost;
			//An additional check if this function is bug-free
			//checkGradientAll();
			accumulateGradient();
			#pragma omp parallel for
			for(int i=0; i<gradient.size(); i++)
				theta[i] = theta[i] - gradient[i]*step;
			cost = costFunction();
		}
	}
}

void NeuralNetwork::printLayers(std::vector<arma::mat> list) {
	for(auto iter: list)
		std::cout<<iter<<std::endl;
}

double inline NeuralNetwork::singleCost(double y, double hypothesis) {
	return -y*log(hypothesis) - (1-y)*log(1-hypothesis);
}

void NeuralNetwork::test() {
	if(getResult == nullptr) {
		std::cerr<<"You have to implement function to get result"<<std::endl;
		exit(-1);
	}

	int errors = 0;
		for(int i=0; i<trainingData.size(); i++) {
			x = trainingData[i];
			propagate();
			arma::mat res = getResult(getOutput() - trainingValues[i]);
			for(int j=0; j<res.n_rows; j++) {
				if(res[j]!=0) {
					errors++;
					break;
				}
			}
		}
		std::cout<<"Data training: "<<(trainingData.size()-errors)<<"/"<<trainingData.size()<<" ("<<(trainingData.size()-errors)*100.0/trainingData.size()<<"%)"<<std::endl;

	errors = 0;
	for(int i=0; i<testData.size(); i++) {
		x = testData[i];
		propagate();
		arma::mat res = getResult(getOutput() - testValues[i]);
		for(int j=0; j<res.n_rows; j++) {
			if(res[j]!=0) {
				errors++;
				break;
			}
		}
	}
	std::cout<<"Data test: "<<(testData.size()-errors)<<"/"<<testData.size()<<" ("<<(testData.size()-errors)*100.0/testData.size()<<"%)"<<std::endl;
}

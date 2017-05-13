#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include <vector>
#include <armadillo>
#include <tuple>
#include <iostream>

//some useful typedefs
template <typename T> struct NeuralNetwork;
using NN = NeuralNetwork<double>;
using NNf = NeuralNetwork<float>;
using matf = arma::Mat<float>;

//add -D_USE_CV_ to g++ to use this; it needs opencv to work
//this is a set of useful function for image loading and compressing
#ifdef _USE_CV_
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//compressionLevel (1-9); 9 is the maximum compression level, which is default
std::vector<uchar> compressPNG(cv::Mat image, int compressionLevel=9);

//compresses all images from given vector and puts 0 at the end to make every row same length
std::tuple<std::vector<std::vector<uchar> >, std::vector<int> > compressImgSet(std::vector<cv::Mat> imgs);

//decompresses single image that was lenghtened to fit the set
cv::Mat decompressResizeImg(std::vector<uchar> data, int length);

//decompresses and resizes all images from vector
std::vector<cv::Mat> decompressImgSet(std::vector<std::vector<uchar> > data, std::vector<int> dataLength);

void loadImages(std::string inputImgsFilename, std::vector<cv::Mat>& inputImgs, std::string outputImgsFilename, std::vector<cv::Mat>& outputImgs, int loadFlag=CV_LOAD_IMAGE_UNCHANGED);

#endif

/*
 * How to run
 *
 * 1) create NN class
 * 2) setLayersSizes
 * 3) randomInitialize
 * 4) set TrainingData & TrainingValues
 * 5) propagateAllTrainingData
 * 6) set
 * 7) gradientDescent
 * 8) define getResult function & test
 */

template <typename T = float>
struct NeuralNetwork {

	static const int SIGMOID = 0;
	static const int TANH = 1;

	//if cost decrease is lower than this value then training algorithm stops
	T minCostDecrease = 10e-6;

	//value used for gradient checking and calculating f'
	T epsilon = 10e-4;

	//learning rate
	T learningRate = 0.3;

	//learning rate will always be at least this value
	T learningRateMin = 0;

	//learning rate won't be higher than this value, but when set to 0 it won't check it
	T learningRateMax = 0;

	/*
	 * Learning rate decreases every iteration in accordance with given equation
	 * learningRate -= learningRate*learningRateDecrasePercent/100;
	 * learningRate -= learningRateDecraseValue;
	 */
	T learningRateDecrasePercent = 0;

	/*
	 * Learning rate decreases every iteration in accordance with given equation
	 * learningRate -= learningRate*learningRateDecrasePercent/100;
	 * learningRate -= learningRateDecraseValue;
	 */
	T learningRateDecraseValue = 0;

	//regularization parameter
	T regularizationParameter = 0;

	//save theta parameters every Nth iteration
	int saveN = 0;

	int layers;
	std::vector<int> layersSizes;

	std::vector<arma::Mat<T> > a;
	std::vector<arma::Mat<T> > z;
	std::vector<arma::Mat<T> > theta;
	std::vector<arma::Mat<T> > delta;
	std::vector<arma::Mat<T> > gradient;

	std::vector<arma::Mat<T> > trainingData;
	std::vector<arma::Mat<T> > trainingValues;
	std::vector<arma::Mat<T> > trainingOutput;
	std::vector<arma::Mat<T> > testData;
	std::vector<arma::Mat<T> > testValues;
	arma::Mat<T> x;
	int activationFunction = SIGMOID;

	NeuralNetwork(std::initializer_list<int> list);
	virtual ~NeuralNetwork();

	T inline static sigmoid(T number);
	T inline static tanh(T number);
	T inline static sigmoidGradient(T number);
	T inline static tanhGradient(T number);
	template <typename B> arma::Mat<T> mapMatrix (const arma::Mat<T>& matrix, B binaryOp) {
		int rows = matrix.n_rows;
		int cols = matrix.n_cols;
		arma::Mat<T> result = arma::Mat<T>(rows, cols);

		if(rows > cols) {
			#pragma omp parallel for
			for(int i=0; i<rows; i++)
				for(int j=0; j<cols; j++)
					result(i, j) = binaryOp(matrix(i, j), *this);
		}
		else {
			#pragma omp parallel for
			for(int j=0; j<cols; j++)
				for(int i=0; i< rows; i++)
					result(i, j) = binaryOp(matrix(i, j), *this);
		}
		return result;
	}

	arma::Mat<T> (*getResult)(arma::Mat<T> result) = nullptr;
	T inline activationFun(T number);
	T inline activationGradient(T number);

	void addBiasUnit(arma::Mat<T>& vector);
	void propagate();
	void propagateAllTrainingData();
	void backPropagateError(arma::Mat<T> y, arma::Mat<T> hypothesis);
	void accumulateGradient();
	arma::Mat<T> getOutput();
	void randomInitialize(bool setBiasToZeros = false);
	T costFunction();
	void gradientDescent(int maxIter = 1000, bool quiet=true);
	void printLayers(std::vector<arma::Mat<T> > list);
	void printLayers(std::vector<int> list);
	T inline singleCost(T y, T hypothesis);
	T checkGradient(int layer, int row, int col, bool accumulate = false);
	void checkGradientAll();
	void test();
	void loadParams();
	void saveParams();
};

template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::initializer_list<int> list) {
	layers = list.size();
	layersSizes = list;

	a = std::vector<arma::Mat<T> >(layers);
	z = std::vector<arma::Mat<T> >(layers-1);
	theta = std::vector<arma::Mat<T> >(layers-1);
	delta = std::vector<arma::Mat<T> >(layers-1);
	gradient = std::vector<arma::Mat<T> >(layers-1);

	trainingData = std::vector<arma::Mat<T> >();
	trainingValues = std::vector<arma::Mat<T> >();
	trainingOutput = std::vector<arma::Mat<T> >();

	testData = std::vector<arma::Mat<T> >();
	testValues = std::vector<arma::Mat<T> >();
}

template <typename T>
NeuralNetwork<T>::~NeuralNetwork() {

}

template <typename T>
void NeuralNetwork<T>::addBiasUnit(arma::Mat<T>& vector) {
	int rows = vector.n_rows;
	vector.resize(rows+1, 1);
	vector[rows] = 1;
}

template <typename T>
void NeuralNetwork<T>::propagate() {
	a[0] = x;
		addBiasUnit(a[0]);
	for(int i=0; i<layers-1; i++) {
		z[i] = theta[i].t()*a[i];
		a[i+1] = mapMatrix(z[i], [](double num, NeuralNetwork net){return net.activationFun(num);});
		if(i+1 < layers-1)
			addBiasUnit(a[i+1]);
	}
}

template <typename T>
void NeuralNetwork<T>::loadParams() {
	std::vector<std::ifstream> stream (theta.size());
	for(int i=0; i<stream.size(); i++) {
		stream[i] = std::ifstream("th"+std::to_string(i), std::ios::in);
		if(!stream[i].good()) {
			std::cerr<<"Can't open file th"+std::to_string(i)<<" to read"<<std::endl;
			exit(-1);
		}
		theta[i].load(stream[i]);
		stream[i].close();
	}
}

template <typename T>
void NeuralNetwork<T>::saveParams() {
	std::vector<std::ofstream> stream (theta.size());
	for(int i=0; i<stream.size(); i++) {
		stream[i] = std::ofstream("th"+std::to_string(i), std::ios::out | std::ios::trunc);
		if(!stream[i].good()) {
			std::cerr<<"Can't open file th"+std::to_string(i)<<" to write"<<std::endl;
			exit(-1);
		}
		theta[i].save(stream[i]);
		stream[i].close();
	}
}

template <typename T>
T inline NeuralNetwork<T>::sigmoid(const T number) {
	return 1.0/(1.0 + pow(M_E, -1.0*number));
}

template <typename T>
T inline NeuralNetwork<T>::tanh(const T number) {
	return 2.0/(1 + pow(M_E, -number*2)) - 1;
}

template <typename T>
T inline NeuralNetwork<T>::sigmoidGradient(T number) {
	return sigmoid(number)*(1-sigmoid(number));
}

template <typename T>
T inline NeuralNetwork<T>::tanhGradient(const T number) {
	return (1-tanh(number)*tanh(number));
}

template <typename T>
T inline NeuralNetwork<T>::activationFun(const T number) {
	if(activationFunction == SIGMOID)
		return sigmoid(number);
	else if(activationFunction == TANH)
		return tanh(number);
}

template <typename T>
T inline NeuralNetwork<T>::activationGradient(const T number) {
	if(activationFunction == SIGMOID)
		return sigmoidGradient(number);
	else if(activationFunction == TANH)
		return tanhGradient(number);
}

template <typename T>
arma::Mat<T> NeuralNetwork<T>::getOutput() {
	return a[layers-1];
}

template <typename T>
void NeuralNetwork<T>::backPropagateError(arma::Mat<T> x, arma::Mat<T> y) {
	this -> x = x;
	propagate();
	delta[layers-2] = a[layers-1] - y;
	for(int i=layers-3; i>=0; i--) {
		arma::Mat<T> th = theta[i+1];
		th.resize(th.n_rows-1, th.n_cols);
		delta[i] = th*delta[i+1];
		delta[i] = delta[i] % mapMatrix(z[i], [](double num, NeuralNetwork net){return net.activationGradient(num);});
	}
}

template <typename T>
void NeuralNetwork<T>::randomInitialize(bool setBiasParamsToZeros) {

	int previousLayerSize;
	int nextLayerSize;

	T init = sqrt(6)/sqrt(layersSizes[0]+layersSizes[layers-1]);

	#pragma omp parallel for
	for(int i=0; i<layers-1; i++) {
		previousLayerSize = layersSizes[i];
		nextLayerSize = layersSizes[i+1];
		arma::arma_rng::set_seed_random();
		theta[i] = arma::randu<arma::Mat<T> >(previousLayerSize+1, nextLayerSize);
		theta[i]*=(init*2);
		theta[i]-=arma::Mat<T>(theta[i].n_rows, theta[i].n_cols).fill(init);
	}
	if(setBiasParamsToZeros) {
		for(int i=0; i<layers-1; i++)
			for(int j=0; j<theta[i].n_cols; j++)
				theta[i](theta[i].n_rows-1, j) = 0;
	}
}

template <typename T>
T NeuralNetwork<T>::costFunction() {
	T cost = 0;
	propagateAllTrainingData();

	//regularization for cost function
	#pragma omp parallel
	for(int i=0; i<layers-1; i++) {
		arma::Mat<T> th = theta[i];
		th.resize(theta[i].n_rows-1, theta[i].n_cols);
		arma::square(th);
		cost += arma::accu(th);
	}
	cost *= (regularizationParameter/2);

	#pragma omp parallel for
	for(int i=0; i<trainingData.size(); i++) {
		for(int j=0; j<layersSizes[layers-1]; j++) {
			cost += singleCost(trainingValues[i][j], trainingOutput[i][j]);
		}
	}
	cost /= trainingData.size();
	return cost;
}

template <typename T>
void NeuralNetwork<T>::accumulateGradient() {
	for(int i=0; i<layers-1; i++) {
		gradient[i].copy_size(theta[i]);
		gradient[i].fill(0);
	}
	for(int k=0; k<trainingOutput.size(); k++) {
		backPropagateError(trainingData[k], trainingValues[k]);
		#pragma omp parallel for
		for(int i=0; i<layers-1; i++) {
			gradient[i] += a[i]*delta[i].t();
			arma::Mat<T> th = theta[i];
			for(int j=0; j<th.n_cols; j++)
				th(th.n_rows-1, j) = 0;
			th*=regularizationParameter;
			gradient[i] += th;
		}
	}
	for(int i=0; i<gradient.size(); i++)
		gradient[i] /= trainingData.size();
}

template <typename T>
T NeuralNetwork<T>::checkGradient(int layer, int row, int col, bool accumulate) {
	if(accumulate)
		accumulateGradient();
	T gradient = this->gradient[layer](row, col);
	std::vector<arma::Mat<T> > thetaPlus = theta;
	std::vector<arma::Mat<T> > thetaMinus = theta;
	std::vector<arma::Mat<T> > thetaTmp = theta;
	thetaPlus[layer](row, col) += epsilon;
	thetaMinus[layer](row, col) -= epsilon;
	theta = thetaPlus;
	T costPlus = costFunction();
	theta = thetaMinus;
	T costMinus = costFunction();
	theta = thetaTmp;
	return (costPlus - costMinus)/(2*epsilon);
}

template <typename T>
void NeuralNetwork<T>::checkGradientAll() {
	accumulateGradient();
	std::vector<arma::Mat<T> > gradientDefinition = gradient;
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

template <typename T>
void NeuralNetwork<T>::propagateAllTrainingData() {
	trainingOutput.clear();
	for(int i=0; i<trainingData.size(); i++) {
		x = trainingData[i];
		propagate();
		trainingOutput.push_back(getOutput());
	}
}

template <typename T>
void NeuralNetwork<T>::gradientDescent(int maxIter, bool quiet) {
	T prevprevCost = INT_MAX;
	T prevCost = INT_MAX - 1;
	T cost = costFunction();
	for(int i=1; i<maxIter; i++) {
		if ((prevCost - cost < minCostDecrease) && (prevprevCost - prevCost < minCostDecrease)) {
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
				theta[i] = theta[i] - gradient[i]*learningRate;
			cost = costFunction();
			learningRate -= learningRate*learningRateDecrasePercent/100;
			learningRate -= learningRateDecraseValue;
			if(learningRate < learningRateMin)
				learningRate = learningRateMin;
			if((learningRateMax > 0) && (learningRate > learningRateMax))
				learningRate = learningRateMax;
			if((saveN > 0) && (i%saveN == 0)) {
				saveParams();
			}
		}
	}
}

template <typename T>
void NeuralNetwork<T>::printLayers(std::vector<arma::Mat<T> > list) {
	for(auto& iter: list)
		std::cout<<iter<<std::endl;
}

template <typename T>
void NeuralNetwork<T>::printLayers(std::vector<int> list) {
	for(auto& iter: list)
		std::cout<<iter<<std::endl;
}

template <typename T>
T inline NeuralNetwork<T>::singleCost(T y, T hypothesis) {
	return -y*log(hypothesis) - (1-y)*log(1-hypothesis);
}

template <typename T>
void NeuralNetwork<T>::test() {
	if(getResult == nullptr) {
		std::cerr<<"You have to implement function to get result"<<std::endl;
		exit(-1);
	}

	int errors = 0;
		for(int i=0; i<trainingData.size(); i++) {
			x = trainingData[i];
			propagate();
			arma::Mat<T> res = getResult(getOutput() - trainingValues[i]);
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
		arma::Mat<T> res = getResult(getOutput() - testValues[i]);
		for(int j=0; j<res.n_rows; j++) {
			if(res[j]!=0) {
				errors++;
				break;
			}
		}
	}
	std::cout<<"Data test: "<<(testData.size()-errors)<<"/"<<testData.size()<<" ("<<(testData.size()-errors)*100.0/testData.size()<<"%)"<<std::endl;
}

template <typename T, typename V>
arma::Mat<T> matFromVec(std::vector<V> vec, bool normalize) {
	arma::Mat<T> result(vec.size(), 1);
	for(int i=0; i<vec.size(); i++) {
		if(normalize)
			result(i, 0) = vec[i]/255;
		else
			result(i, 0) = vec[i];
	}
	return result;
}

template <typename T, typename V>
std::vector<T> vecFromMat(arma::Mat<V> mat, bool normalize) {
	std::vector<T> result(mat.n_rows);
	for(int i=0; i<mat.n_rows; i++) {
		if(normalize)
			result[i] = mat(i, 0)/255;
		else
			result[i] = mat(i, 0);
	}
	return result;
}

#endif /* NEURALNETWORK_HPP_ */

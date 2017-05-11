#include "NeuralNetwork.hpp"
#include <cmath>
#include <omp.h>

#ifdef _USE_CV_

void loadImages(std::string inputImgsFilename, std::vector<cv::Mat>& inputImgs, std::string outputImgsFilename, std::vector<cv::Mat>& outputImgs, int loadFlag) {
	std::ifstream imagesInputStream(inputImgsFilename, std::ios::in), imagesOutputStream(outputImgsFilename, std::ios::in);
	if(!imagesInputStream.good()) {
		std::cerr<<"Can't open file with input image names"<<std::endl;
		exit(-1);
	}
	else if(!imagesOutputStream.good()) {
		std::cerr<<"Can't open file with output image names"<<std::endl;
		exit(-1);
	}
	std::string imageName;

	while(std::getline(imagesInputStream, imageName))
		inputImgs.push_back(cv::imread(imageName, loadFlag));

	while(std::getline(imagesOutputStream, imageName))
		outputImgs.push_back(cv::imread(imageName, loadFlag));

	for(int i=0; i<inputImgs.size(); i++) {
		if(!inputImgs[i].data) {
			std::cerr<<"Can't read input image number "<<i<<std::endl;
			exit(-1);
		}
		if(!outputImgs[i].data) {
			std::cerr<<"Can't read output image number "<<i<<std::endl;
			exit(-1);
		}
	}
	imagesInputStream.close();
	imagesOutputStream.close();
}

std::vector<uchar> compressPNG(cv::Mat image, int compressionLevel) {
	std::vector<uchar> compressedData(image.rows*image.cols);

	std::vector<int> compressionParams;
	compressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(9);
	cv::imencode(".png", image, compressedData, compressionParams);

	compressedData.shrink_to_fit();
	return compressedData;

}

std::tuple<std::vector<std::vector<uchar> >, std::vector<int> > compressImgSet(std::vector<cv::Mat> imgs) {
	std::vector<std::vector<uchar> > data;
	std::vector<int> dataLength;

	int maxLength = 0;
	for(int i=0; i<imgs.size(); i++) {
		data.push_back(compressPNG(imgs[i], 9));
		dataLength.push_back(data[i].size());
		if(dataLength[i] > maxLength)
			maxLength = dataLength[i];
	}
	for(auto& vec: data)
		vec.resize(maxLength);

	return make_tuple(data, dataLength);
}

cv::Mat decompressResizeImg(std::vector<uchar> data, int length) {
	data.resize(length);
	return imdecode(cv::Mat(data), CV_LOAD_IMAGE_UNCHANGED);
}

std::vector<cv::Mat> decompressImgSet(std::vector<std::vector<uchar> > data, std::vector<int> dataLength) {
	std::vector<cv::Mat> result;
	for(int i=0; i<data.size(); i++) {
		result.push_back(decompressResizeImg(data[i], dataLength[i]));
	}
	return result;
}
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Color.hpp"

struct ColorLab convertRGB2Lab(struct Color color) {
	cv::Mat matRGB(1,1,CV_8UC3);

	matRGB.data[2] = color.r;
	matRGB.data[1] = color.g;
	matRGB.data[0] = color.b;

	cv::Mat matLab(1,1,CV_8UC3);
	cv::cvtColor(matRGB, matLab, CV_BGR2Lab);

	struct ColorLab result;
	result.L = matLab.data[0];
	result.a = matLab.data[1];
	result.b = matLab.data[2];

	return result;
}

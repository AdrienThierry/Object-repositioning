#ifndef FOREGROUND_EXTRACTION_H_   /* Include guard */
#define FOREGROUND_EXTRACTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "BoundingBox.hpp"
#include "Point.hpp"

cv::Mat* computeProbsBackground(cv::Mat *image);
cv::Mat* computeProbsForeground(cv::Mat *image, struct BoundingBox bb);

// Convert a CV_64FC1 probs matrix into a CV_8UC3 displayable matrix
cv::Mat* convertProbsToCV_Mat(cv::Mat *probs);

#endif

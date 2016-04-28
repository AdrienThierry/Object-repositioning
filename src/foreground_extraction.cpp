#include <opencv2/core/core.hpp>

#include "foreground_extraction.hpp"

GraphType* createGraph(int rows, int cols) {
	GraphType *g = new GraphType(/*estimated # of nodes*/ rows*cols, /*estimated # of edges*/ rows*cols);

	for (int i = 0 ; i < rows*cols ; i++) {
		g -> add_node();
	}

	return g; 
}

void assignDataTerm(GraphType *g, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols) {
	for (int i = 0 ; i < rows ; i++) {
		for (int j = 0 ; j < cols ; j++) {
			struct Point currentPixel;
			currentPixel.x = j;
			currentPixel.y = i;

			double foregroundWeight, backgroundWeight;

			// Possibly foreground
			if (isInsideBB(currentPixel, bb)) {
				// It's normal that background and foreground are inverted (see https://www.youtube.com/watch?v=lYQQ88nzxAM)
				foregroundWeight = std::max(GMMBackground->weightedLL.at(i).at(j), MAX_DATA_TERM);
				backgroundWeight = std::max(GMMForeground->weightedLL.at(i).at(j), MAX_DATA_TERM);
			}
			// Outside bounding box => positively background
			else {
				foregroundWeight = 0.0;
				backgroundWeight = MAX_DATA_TERM;
			}
			g -> add_tweights( i*cols+j,   /* capacities */  foregroundWeight, backgroundWeight );
		}
	}
}

void assignSmoothnessTerm(IplImage** leftRightImage, IplImage** topBottomImage, GraphType *g, IplImage *image, int rows, int cols) {
	cv::Mat imageMat(image);
	
	std::vector<std::vector<float> > leftRightWeights;
	std::vector<std::vector<float> > topBottomWeights;

	// Left-right weights
	for (int i = 0 ; i < rows ; i++) {
		std::vector<float> row;

		for (int j = 0 ; j < cols-1 ; j++) {

			float weight = 0.0;
			struct Color color1, color2;
			struct ColorLab color1Lab, color2Lab;

			color1.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 2];
			color1.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 1];
			color1.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 0];
			
			color2.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 2];
			color2.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 1];
			color2.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 0];

			color1Lab = convertRGB2Lab(color1);
			color2Lab = convertRGB2Lab(color2);

			float dist = distanceLab(color1Lab, color2Lab);

			weight = LAMBDA * exp(-1.0 * dist * dist / (2.0 * (float)SIGMA * (float)SIGMA));

			g -> add_edge( i*cols+j, i*cols+(j+1), /* capacities */ weight , weight );

			row.push_back(weight);
		}

		leftRightWeights.push_back(row);
	}

	// Top-bottom weights
	for (int i = 0 ; i < rows-1 ; i++) {

			std::vector<float> row;

		for (int j = 0 ; j < cols ; j++) {

			float weight = 0.0;
			struct Color color1, color2;
			struct ColorLab color1Lab, color2Lab;

			color1.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 2];
			color1.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 1];
			color1.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 0];
			
			color2.r = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 2];
			color2.g = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 1];
			color2.b = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 0];

			color1Lab = convertRGB2Lab(color1);
			color2Lab = convertRGB2Lab(color2);

			float dist = distanceLab(color1Lab, color2Lab);

			weight = LAMBDA * exp(-1.0 * dist * dist / (2.0 * (float)SIGMA * (float)SIGMA));

			g -> add_edge( i*cols+j, (i+1)*cols+j, /* capacities */ weight , weight );

			row.push_back(weight);
		}

		topBottomWeights.push_back(row);
	}

	// Normalize weights to show them as an image
	float maxLeftRight = leftRightWeights.at(0).at(0);
	float minLeftRight = leftRightWeights.at(0).at(0);
	float maxTopBottom = leftRightWeights.at(0).at(0);
	float minTopBottom = leftRightWeights.at(0).at(0);

	for (unsigned int i = 0 ; i < leftRightWeights.size() ; i++) {
		for (unsigned int j = 0 ; j < leftRightWeights.at(i).size() ; j++) {
			if (leftRightWeights.at(i).at(j) > maxLeftRight)
				maxLeftRight = leftRightWeights.at(i).at(j);
			if (leftRightWeights.at(i).at(j) < minLeftRight)
				minLeftRight = leftRightWeights.at(i).at(j);
			
		}
	}

	for (unsigned int i = 0 ; i < topBottomWeights.size() ; i++) {
		for (unsigned int j = 0 ; j < topBottomWeights.at(i).size() ; j++) {
			if (topBottomWeights.at(i).at(j) > maxTopBottom)
				maxTopBottom = topBottomWeights.at(i).at(j);
			if (topBottomWeights.at(i).at(j) < minTopBottom)
				minTopBottom = topBottomWeights.at(i).at(j);
			
		}
	}

	for (unsigned int i = 0 ; i < leftRightWeights.size() ; i++) {
		for (unsigned int j = 0 ; j < leftRightWeights.at(i).size() ; j++) {
			leftRightWeights.at(i).at(j) = (leftRightWeights.at(i).at(j) - minLeftRight) / (maxLeftRight - minLeftRight);
		}
	}

	for (unsigned int i = 0 ; i < topBottomWeights.size() ; i++) {
		for (unsigned int j = 0 ; j < topBottomWeights.at(i).size() ; j++) {
			topBottomWeights.at(i).at(j) = (topBottomWeights.at(i).at(j) - minLeftRight) / (maxLeftRight - minLeftRight);
		}
	}

	// Create images to show weights
	cv::Mat tmpLR = cv::Mat::zeros(leftRightWeights.size(), leftRightWeights.at(0).size(), CV_8UC3);
	cv::Mat tmpTB = cv::Mat::zeros(topBottomWeights.size(), topBottomWeights.at(0).size(), CV_8UC3);
	
	for (int i = 0 ; i < tmpLR.rows ; i++) {
		for (int j = 0 ; j < tmpLR.cols ; j++) {
			int color = 255 * leftRightWeights.at(i).at(j);

			tmpLR.data[tmpLR.step[0]*i + tmpLR.step[1]*j + 0] = color;
			tmpLR.data[tmpLR.step[0]*i + tmpLR.step[1]*j + 1] = color;
			tmpLR.data[tmpLR.step[0]*i + tmpLR.step[1]*j + 2] = color;
		}
	}

	for (int i = 0 ; i < tmpTB.rows ; i++) {
		for (int j = 0 ; j < tmpTB.cols ; j++) {
			int color = 255 * topBottomWeights.at(i).at(j);

			tmpTB.data[tmpTB.step[0]*i + tmpTB.step[1]*j + 0] = color;
			tmpTB.data[tmpTB.step[0]*i + tmpTB.step[1]*j + 1] = color;
			tmpTB.data[tmpTB.step[0]*i + tmpTB.step[1]*j + 2] = color;
		}
	}

	cvReleaseImage(leftRightImage);
	cvReleaseImage(topBottomImage);
	*leftRightImage = cvCreateImage(cvSize(tmpLR.cols,tmpLR.rows),8,3);
	*topBottomImage = cvCreateImage(cvSize(tmpTB.cols,tmpTB.rows),8,3);
	IplImage ipltemp = tmpLR;
	cvCopy(&ipltemp,*leftRightImage);
	ipltemp = tmpTB;
	cvCopy(&ipltemp,*topBottomImage);
}

struct Foreground extractForeground(IplImage *input, IplImage **result, IplImage **leftRightImage, IplImage **topBottomImage, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);
	cv::Mat inputMat(input);

	struct Foreground foregroundResult;

	// Create graph
	GraphType *g = createGraph(rows, cols);

	// Assign weights
	assignDataTerm(g, GMMForeground, GMMBackground, bb, rows, cols);
	assignSmoothnessTerm(leftRightImage, topBottomImage, g, input, rows, cols);

	// Perform graph cut
	g -> maxflow();

	// Create resulting image and foreground mask
	for (int i = 0 ; i < rows ; i++) {
		std::vector<bool> row;
		for (int j = 0 ; j < cols ; j++) {
			
			// If foreground
			if (g->what_segment(i*cols+j) == GraphType::SOURCE) {
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 0];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 1];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 2];

				row.push_back(true);
			}

			// If background
			else {
				row.push_back(false);			
			}
		}
		foregroundResult.mask.push_back(row);
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);

	freeGraph(g);

	return foregroundResult;
}

void computeForegroundBB(struct Foreground *foreground) {
	unsigned int Xmax = 0, Ymax = 0;
	unsigned int Xmin = foreground->mask.at(0).size();
	unsigned int Ymin = foreground->mask.size();

	// Compute bounding box that fits foreground mask the most
	for (unsigned int i = 0 ; i < foreground->mask.size() ; i++) {
		for (unsigned int j = 0 ; j < foreground->mask.at(0).size() ; j++) {
			if (foreground->mask.at(i).at(j) == true) {
				if (j > Xmax)
					Xmax = j;
				if (j < Xmin)
					Xmin = j;
				if (i > Ymax)
					Ymax = i;
				if (i < Ymin)
					Ymin = i;
			}
		}
	}

	Point p1, p2, p3, p4;
	p1.x = Xmin;
	p1.y = Ymin;
	p2.x = Xmax;
	p2.y = Ymin;
	p3.x = Xmax;
	p3.y = Ymax;
	p4.x = Xmin;
	p4.y = Ymax;

	foreground->bb.points[0] = p1;
	foreground->bb.points[1] = p2;
	foreground->bb.points[2] = p3;
	foreground->bb.points[3] = p4;

	foreground->bb.ends[0] = Xmin;
	foreground->bb.ends[1] = Xmax;
	foreground->bb.ends[2] = Ymin;
	foreground->bb.ends[3] = Ymax;
}

void computeForegroundImage(IplImage *input, IplImage **result, struct Foreground *foreground) {
	int xMin = foreground->bb.ends[0];
	int xMax = foreground->bb.ends[1];
	int yMin = foreground->bb.ends[2];
	int yMax = foreground->bb.ends[3];

	int rows = yMax - yMin;
	int cols = xMax - xMin;

	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC4);
	cv::Mat inputMat(input);

	for (int i = yMin ; i < yMax ; i++) {
		for (int j = xMin ; j < xMax ; j++) {
			if (foreground->mask.at(i).at(j) == true) {

				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 0] = 
					inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 0];

				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 1] = 
					inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 1];

				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 2] = 
					inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 2];

				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 3] = 255; // Opaque
			}

			else {
				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 3] = 0; // Transparent
			}
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,4);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);

}

void updateExtractedForegroundMat(IplImage** result, IplImage *foregroundMat, SDL_Rect *foregroundPosition, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC4);

	cv::Mat foreground(foregroundMat);

	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			if (j > foregroundPosition->x && j < foregroundPosition->x + foregroundPosition->w &&
				i > foregroundPosition->y && i < foregroundPosition->y + foregroundPosition->h) {

				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = 
					foreground.data[foreground.step[0]*(i-foregroundPosition->y) + foreground.step[1]*(j-foregroundPosition->x) + 0];
	 
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = 
					foreground.data[foreground.step[0]*(i-foregroundPosition->y) + foreground.step[1]*(j-foregroundPosition->x) + 1];

				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = 
					foreground.data[foreground.step[0]*(i-foregroundPosition->y) + foreground.step[1]*(j-foregroundPosition->x) + 2];

				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 3] = 
					foreground.data[foreground.step[0]*(i-foregroundPosition->y) + foreground.step[1]*(j-foregroundPosition->x) + 3];

			}
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,4);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}

void freeGraph(GraphType *g) {
	delete g;
}

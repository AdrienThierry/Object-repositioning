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
				foregroundWeight = GMMBackground->weightedLL.at(i).at(j);
				backgroundWeight = GMMForeground->weightedLL.at(i).at(j);
			}
			// Outside bounding box => positively background
			else {
				foregroundWeight = 0.0;
				backgroundWeight = std::numeric_limits<double>::max();
			}
			g -> add_tweights( i*cols+j,   /* capacities */  foregroundWeight, backgroundWeight );
		}
	}
}

void assignSmoothnessTerm(GraphType *g, IplImage *image, int rows, int cols) {
	cv::Mat imageMat(image, true);
	
	// Left-right weights
	for (int i = 0 ; i < rows ; i++) {
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
		}
	}

	// Top-bottom weights
	for (int i = 0 ; i < rows-1 ; i++) {
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
		}
	}
}

struct Foreground extractForeground(IplImage *input, IplImage **result, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);
	cv::Mat inputMat(input, true);

	struct Foreground foregroundResult;

	// Create graph
	GraphType *g = createGraph(rows, cols);

	// Assign weights
	assignDataTerm(g, GMMForeground, GMMBackground, bb, rows, cols);
	assignSmoothnessTerm(g, input, rows, cols);

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
	cv::Mat inputMat(input, true);

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
				tmpResult.data[tmpResult.step[0]*(i-yMin) + tmpResult.step[1]*(j-xMin) + 3] = 0; // Opaque
			}
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,4);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);

}

void convertForegroundMaskToCV_Mat(IplImage** result, struct Foreground *foreground, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);
	
	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			int color = 255 * (int)foreground->mask.at(i).at(j);

			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = color;
			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = color;
			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = color;
			
		}
	}

	// Draw bounding box
	int xMin = foreground->bb.ends[0];
	int xMax = foreground->bb.ends[1];
	int yMin = foreground->bb.ends[2];
	int yMax = foreground->bb.ends[3];
	for (int i = yMin ; i < yMax ; i++) {
		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMin + 0] = 0;
		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMin + 1] = 255;
		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMin + 2] = 0;

		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMax + 0] = 0;
		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMax + 1] = 255;
		tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*xMax + 2] = 0;
	}
	for (int i = xMin ; i < xMax ; i++) {
		tmpResult.data[tmpResult.step[0]*yMin + tmpResult.step[1]*i + 0] = 0;
		tmpResult.data[tmpResult.step[0]*yMin + tmpResult.step[1]*i + 1] = 255;
		tmpResult.data[tmpResult.step[0]*yMin + tmpResult.step[1]*i + 2] = 0;

		tmpResult.data[tmpResult.step[0]*yMax + tmpResult.step[1]*i + 0] = 0;
		tmpResult.data[tmpResult.step[0]*yMax + tmpResult.step[1]*i + 1] = 255;
		tmpResult.data[tmpResult.step[0]*yMax + tmpResult.step[1]*i + 2] = 0;
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}

void freeGraph(GraphType *g) {
	delete g;
}

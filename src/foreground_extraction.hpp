#ifndef FOREGROUND_EXTRACTION_H_   /* Include guard */
#define FOREGROUND_EXTRACTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <limits>
#include <math.h>

#include "BoundingBox.hpp"
#include "Point.hpp"
#include "Color.hpp"
#include "graph.h"
#include "GMM.hpp"


#define MAX_DATA_TERM 1000.0 // Maximum value for data term in graph cut (to prevent huge values due to -log(prob) when prob is small)
#define SIGMA 50.0 // Sigma for smoothness term
#define LAMBDA 1.0 // Relative importance of the smoothness term

typedef Graph<int,int,int> GraphType; // SOURCE = foreground, SINK = background

struct Foreground {
	std::vector<std::vector<bool> > mask; // true = foreground pixel
	struct BoundingBox bb; // Most fitting bounding box
};

// Create graph with a node for each pixel (rows * cols nodes)
GraphType* createGraph(int rows, int cols);

// Assign weights between pixels and terminal nodes (foreground and background)
void assignDataTerm(GraphType *g, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);

// Assign weights between pixels
void assignSmoothnessTerm(IplImage** leftRightImage, IplImage** topBottomImage, GraphType *g, IplImage *image, int rows, int cols);

// Perform graph cut
struct Foreground extractForeground(IplImage *input, IplImage **result, IplImage **leftRightImage, IplImage **topBottomImage, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);

// Compute bounding box that fits foreground the most
void computeForegroundBB(struct Foreground *foreground);

// Compute image with only foreground. The inputs are the original image and the struct foreground
void computeForegroundImage(IplImage *input, IplImage **result, struct Foreground *foreground);

// Update foreground image with new position
void updateExtractedForegroundMat(IplImage** result, IplImage *foregroundMat, SDL_Rect *foregroundPosition, int rows, int cols);

// Free graph
void freeGraph(GraphType *g);

#endif

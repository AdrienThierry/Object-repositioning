#ifndef FOREGROUND_EXTRACTION_H_   /* Include guard */
#define FOREGROUND_EXTRACTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <limits>

#include "BoundingBox.hpp"
#include "Point.hpp"
#include "Color.hpp"
#include "graph.h"
#include "GMM.hpp"

#define SIGMA 10.0 // Sigma for smoothness term
#define LAMBDA 1.0 // Relative importance of the smoothness term

#define dbg_print1(buffer, string, arg) write(1, buffer, sprintf(buffer, string, arg))

typedef Graph<int,int,int> GraphType; // SOURCE = foreground, SINK = background

struct Foreground {
	std::vector<std::vector<bool> > mask; // true = foreground pixel
	struct BoundingBox bb; // Most fitting bounding box
};

GraphType* createGraph(int rows, int cols);

void assignDataTerm(GraphType *g, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);
void assignSmoothnessTerm(GraphType *g, IplImage *image, int rows, int cols);
struct Foreground extractForeground(IplImage *input, IplImage **result, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);

void computeForegroundBB(struct Foreground *foreground);
void computeForegroundImage(IplImage *input, IplImage **result, struct Foreground *foreground);
void convertForegroundMaskToCV_Mat(IplImage** result, struct Foreground *foreground, int rows, int cols);

void freeGraph(GraphType *g);

#endif

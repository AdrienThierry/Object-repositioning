#ifndef FOREGROUND_EXTRACTION_H_   /* Include guard */
#define FOREGROUND_EXTRACTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <limits>

#include "BoundingBox.hpp"
#include "Point.hpp"
#include "graph.h"
#include "GMM.hpp"

typedef Graph<int,int,int> GraphType; // SOURCE = foreground, SINK = background

GraphType* createGraph(int rows, int cols);

void assignDataTerm(GraphType *g, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);
void assignSmoothnessTerm(GraphType *g, IplImage *image);
void extractForeground(IplImage *input, IplImage **result, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols);

void freeGraph(GraphType *g);

#endif

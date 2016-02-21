#ifndef BOUNDINGBOX_H_   /* Include guard */
#define BOUNDINGBOX_H_

#include <vector>

#include "Point.hpp"
#include "Superpixel.hpp"
#include "Color.hpp"

struct BoundingBox {
	Point points[4];
};

bool isInsideBB(Point point, BoundingBox bb);
void computeSuperpixelIntersectionWithBB(std::vector<struct Superpixel>* superpixels, BoundingBox bb);

// Normalised distance between a superpixel centroid and a bounding box
float distanceSuperpixelBB(struct Superpixel superpixel, BoundingBox bb, int rows, int cols);

// Normalised distance between two superpixel centroids
float distanceSuperpixels(struct Superpixel superpixel1, struct Superpixel superpixel2, int rows, int cols);

// Normalised distance between two Lab colors
float distanceLab(struct ColorLab color1, struct ColorLab color2);

void computeSaliencyMap(std::vector<struct Superpixel>* superpixels, BoundingBox bb, int rows, int cols);

#endif

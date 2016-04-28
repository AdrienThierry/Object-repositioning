#ifndef BOUNDINGBOX_H_   /* Include guard */
#define BOUNDINGBOX_H_

#include <vector>

#include "Point.hpp"
#include "Superpixel.hpp"
#include "Color.hpp"

struct BoundingBox {
	Point points[4];
	int ends[4]; // Ordered points (minX, maxX, minY and maxY)
};

// Returns true if point is inside bb
bool isInsideBB(Point point, BoundingBox bb);

// Updates bounding box ends
// MUST BE CALLED when points are modified
void computeBBEnds(BoundingBox *bb);

// Compute intersection values of all superpixels in superpixels
void computeSuperpixelIntersectionWithBB(std::vector<struct Superpixel>* superpixels, BoundingBox bb);

// Normalised distance between a superpixel centroid and a bounding box
// rows and cols are those of the image (used for normalization)
float distanceSuperpixelBB(struct Superpixel *superpixel, BoundingBox bb, int rows, int cols);

// Normalised distance between two superpixel centroids
// rows and cols are those of the image (used for normalization)
float distanceSuperpixels(struct Superpixel *superpixel1, struct Superpixel superpixel2, int rows, int cols);

// Normalised distance between two Lab colors
float distanceLab(struct ColorLab color1, struct ColorLab color2);

// Compute saliency values of all superpixels in superpixels
// rows and cols are those of the image
void computeSaliencyMap(std::vector<struct Superpixel>* superpixels, BoundingBox bb, int rows, int cols);

#endif

#ifndef BOUNDINGBOX_H_   /* Include guard */
#define BOUNDINGBOX_H_

#include <vector>

#include "Point.hpp"
#include "Superpixel.hpp"

struct BoundingBox {
	Point points[4];
};

bool isInsideBB(Point point, BoundingBox bb);
void computeSuperpixelIntersectionWithBB(std::vector<struct Superpixel>* superpixels, BoundingBox bb);

#endif

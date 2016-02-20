#include <stdio.h>

#include "BoundingBox.hpp"

bool isInsideBB(Point point, BoundingBox bb) {
	// Determine min x, max x, min y and max y in bounding box
	int minX = bb.points[0].x;
	int maxX = bb.points[0].x;
	int minY = bb.points[0].y;
	int maxY = bb.points[0].y;
	for (int i = 0 ; i < 4 ; i++) {
		if (bb.points[i].x < minX)
			minX = bb.points[i].x;

		if (bb.points[i].x > maxX)
			maxX = bb.points[i].x;

		if (bb.points[i].y < minY)
			minY = bb.points[i].y;
	
		if (bb.points[i].y > maxY)
			maxY = bb.points[i].y;
	}

	if (point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY)
		return true;
	else
		return false;
}

void computeSuperpixelIntersectionWithBB(std::vector<struct Superpixel>* superpixels, BoundingBox bb) {
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		bool hasAPointInsideBB = false;
		bool hasAPointOutsideBB = false;

		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			if (isInsideBB(superpixels->at(i).pixels.at(j), bb))
				hasAPointInsideBB = true;
			else
				hasAPointOutsideBB = true;

			if (hasAPointInsideBB && hasAPointOutsideBB)
				break;
		}

		if (hasAPointInsideBB && hasAPointOutsideBB) // Superpixel crosses BB
			superpixels->at(i).intersection = INTERSECT;
		else if (hasAPointInsideBB && !hasAPointOutsideBB) // Superpixel is entirely inside BB
			superpixels->at(i).intersection = INSIDE;
		else // Superpixel is entirely outside BB
			superpixels->at(i).intersection = OUTSIDE;
			
	}

}

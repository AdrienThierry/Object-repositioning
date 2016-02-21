#include <stdio.h>
#include <cmath>

#include "BoundingBox.hpp"
#include "Color.hpp"

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

float distanceSuperpixelBB(struct Superpixel superpixel, BoundingBox bb, int rows, int cols) {
	// Determine min x, max x, min y and max y in bounding box
	float minX = (float)bb.points[0].x;
	float maxX = (float)bb.points[0].x;
	float minY = (float)bb.points[0].y;
	float maxY = (float)bb.points[0].y;
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

	// Warning : distance has to be normalized (see article)
	float distance = (float)superpixel.center.x - minX;
	if (maxX - superpixel.center.x < distance)
		distance = (maxX - superpixel.center.x) / (float)cols;
	else if (superpixel.center.y - minY < distance)
		distance = (superpixel.center.y = minY) / (float)rows;
	else if (maxY - superpixel.center.y < distance)
		distance = (maxY - superpixel.center.y) / (float)rows;
	else
		distance /= (float)cols;

	return distance;
}

float distanceSuperpixels(struct Superpixel superpixel1, struct Superpixel superpixel2, int rows, int cols) {
	float distance = 0;
	float diffX = ((float)superpixel2.center.x - (float)superpixel1.center.x) / (float)cols;
	float diffY = ((float)superpixel2.center.y - (float)superpixel1.center.y) / (float)rows;

	distance = sqrt((diffX * diffX) + (diffY * diffY));

	return distance;
}

float distanceLab(struct ColorLab color1, struct ColorLab color2) {
	float distance = 0;
	float diffL = ((float)color2.L - (float)color1.L) / 100.0;
	float diffa = ((float)color2.a - (float)color1.a) / 300.0;
	float diffb = ((float)color2.b - (float)color1.b) / 300.0;

	distance = sqrt((diffL * diffL) + (diffa * diffa) + (diffb * diffb));

	return distance;
}

void computeSaliencyMap(std::vector<struct Superpixel>* superpixels, struct BoundingBox bb, int rows, int cols) {
	// Create a vector of all superpixels that cross the bounding box
	std::vector<struct Superpixel*> crossingSP;
	float nbOfPixelsInCrossingSP = 0.0; // Total number of pixels in all superpixels that cross the bounding box

	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		if (superpixels->at(i).intersection == INTERSECT) {
			crossingSP.push_back(&(superpixels->at(i)));
			nbOfPixelsInCrossingSP += (float)superpixels->at(i).pixels.size();
		}
	}

	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {

		// Every superpixel that is not entirely inside the bounding box is considered BACKGROUND		
		if (superpixels->at(i).intersection != INSIDE) {
			superpixels->at(i).saliency = 0.0;
		}
		else {
			float saliency = 0.0;
			for (unsigned int j = 0 ; j < crossingSP.size() ; j++) {
				float toAdd = 0;
				toAdd = exp(-1.0 * distanceSuperpixels(superpixels->at(i), *crossingSP.at(j), rows, cols)/(0.5*0.5));
				toAdd *= (1.0 - exp(-1.0 * distanceSuperpixelBB(superpixels->at(i), bb, rows, cols) / (0.5*0.5)));
				toAdd *= ((float)crossingSP.at(j)->pixels.size() / nbOfPixelsInCrossingSP);

				struct ColorLab colorI = convertRGB2Lab(superpixels->at(i).color);
				struct ColorLab colorJ = convertRGB2Lab(crossingSP.at(j)->color);

				toAdd *= distanceLab(colorI, colorJ);
				
				saliency += toAdd;
			}
			superpixels->at(i).saliency = saliency;
		}
	}
}

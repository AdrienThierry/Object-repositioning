#ifndef DEPTH_H_   /* Include guard */
#define DEPTH_H_

#include <unistd.h>
#include "Superpixel.hpp"
#include "BoundingBox.hpp"
#include "PolygonalChain.hpp"
#include "foreground_extraction.hpp"

std::vector<std::vector<float> > computeDepthMap(struct PolygonalChain *groundLine, SDL_Rect* foregroundPosition, IplImage *foreground, int rows, int cols);

void convertDepthToCV_Mat(IplImage **result, std::vector<std::vector<float> > *depthMap);

void computeFinalImage(std::vector<std::vector<float> > *backgroundDepthMap, std::vector<std::vector<float> > *foregroundDepthMap, IplImage **result, IplImage *foreground, IplImage *background, int rows, int cols);

#endif

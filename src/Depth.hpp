#ifndef DEPTH_H_   /* Include guard */
#define DEPTH_H_

#include <unistd.h>
#include "Superpixel.hpp"
#include "BoundingBox.hpp"
#include "PolygonalChain.hpp"
#include "foreground_extraction.hpp"

// Compute depth map
std::vector<std::vector<float> > computeDepthMap(struct PolygonalChain *groundLine, SDL_Rect* foregroundPosition, IplImage *foreground, int rows, int cols);

// Create image from depth map
void convertDepthToCV_Mat(IplImage **result, std::vector<std::vector<float> > *depthMap);

// Compute final image from :
// - depth map of the background (to know the depth of the original object)
// - depth map of the foreground (depth of the moving copy)
void computeFinalImage(std::vector<std::vector<float> > *backgroundDepthMap, std::vector<std::vector<float> > *foregroundDepthMap, IplImage **result, IplImage *foreground, IplImage *background, int rows, int cols);

#endif

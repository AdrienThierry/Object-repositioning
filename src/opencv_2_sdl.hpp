#ifndef OPENCV_2_SDL_H_   /* Include guard */
#define OPENCV_2_SDL_H_

#include <stdio.h>
#include <SDL2/SDL.h>
#include <opencv2/opencv.hpp>

void convertCV_MatToSDL_Surface(SDL_Surface **result, IplImage *image);
void convertCV_Mat_WithAlpha_ToSDL_Surface(SDL_Surface **result, IplImage *image);

#endif

#ifndef OPENCV_2_SDL_H_   /* Include guard */
#define OPENCV_2_SDL_H_

#include <stdio.h>
#include <SDL2/SDL.h>
#include <opencv2/opencv.hpp>

SDL_Surface* convertCV_MatToSDL_Surface(const cv::Mat &matImg);

#endif

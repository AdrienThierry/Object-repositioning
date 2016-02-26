#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <SDL2/SDL.h>

#include "opencv_2_sdl.hpp"
#include "Superpixel.hpp"
#include "Point.hpp"
#include "BoundingBox.hpp"
#include "MeanShift.h"
#include "foreground_extraction.hpp"
#include "GMM.hpp"

using namespace std;

typedef struct Superpixel Superpixel;
typedef struct Point Point;
typedef struct BoundingBox BoundingBox;

int main( int argc, char** argv )
{
	enum WhatToShow { BaseImage, Superpixels, SuperpixelsIntersection, Saliency, 
					GMMLabelsBackground, GMMLabelsForeground, GMMWeightedLLBackground, GMMWeightedLLForeground};
	WhatToShow currentlyShown = BaseImage;
	bool drawCentroids = false;
	bool showBoundingBox = true;
	bool boundingBoxDrawn = false;

	//--------------------------------------------------------------------------------
	// Image loading
	//--------------------------------------------------------------------------------
	cv::Mat img = cv::imread("data/musician.jpg"); // Input image

	// IplImage generation from img. Ipgimage is used as input for Meanshift segmentation
	IplImage* img2;
	img2 = cvCreateImage(cvSize(img.cols,img.rows),8,3);
	IplImage ipltemp=img;
	cvCopy(&ipltemp,img2);

	//--------------------------------------------------------------------------------
	// SDL Initialisation
	//--------------------------------------------------------------------------------
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		cout << "SDL_Init Error: " << SDL_GetError() << endl;
		return 1;
	}

	SDL_Window *win = SDL_CreateWindow("Object repositioning", 100, 100, img.cols, img.rows, SDL_WINDOW_SHOWN);
	if (win == NULL) {
		std::cout << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (ren == NULL){
		SDL_DestroyWindow(win);
		std::cout << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}
 
	//--------------------------------------------------------------------------------
	// Mean shift filtering
	//--------------------------------------------------------------------------------
	int **ilabels = new int *[img2->height];
	for(int i=0;i<img2->height;i++)
		ilabels[i] = new int [img2->width];
	MeanShift(img2, ilabels);

	//--------------------------------------------------------------------------------
	// Superpixels generation
	//--------------------------------------------------------------------------------
	vector<Superpixel> *superpixels = computeSuperpixels(ilabels, img, img.rows, img.cols);

	//--------------------------------------------------------------------------------
	// Matrix declarations
	//--------------------------------------------------------------------------------	
	IplImage *superpixelsMat = NULL;
	convertSuperpixelsToCV_Mat(&superpixelsMat, superpixels, img.rows, img.cols);

	IplImage *superpixelsIntersectionMat = NULL;
	IplImage *saliencyMat = NULL;
	IplImage *GMMLabelsBackgroundMat = NULL;
	IplImage *GMMLabelsForegroundMat = NULL;
	IplImage *GMMWeightedLLBackgroundMat = NULL;
	IplImage *GMMWeightedLLForegroundMat = NULL;

	SDL_Surface *surf = NULL;
	SDL_Texture *tex = NULL;

	//--------------------------------------------------------------------------------
	// SDL main loop and bounding box handling
	//--------------------------------------------------------------------------------	
	SDL_Event e;
	BoundingBox boundingBox;
	for (int i = 0 ; i < 4 ; i++) {
		boundingBox.points[i].x = 0;
		boundingBox.points[i].y = 0;
	}
	bool quit = false;
	bool clicking = false;
	struct Point clickCoord, mousePosition;
	while (!quit){
		while (SDL_PollEvent(&e)){
			if (e.type == SDL_QUIT){
				quit = true;
			}

			//--------------------------------------------------------------------------------
			// Keyboard key DOWN
			//--------------------------------------------------------------------------------
			if (e.type == SDL_KEYDOWN) {
				switch(e.key.keysym.sym) {
					case SDLK_KP_0:
						currentlyShown = BaseImage;
						break;
					case SDLK_KP_1:
						currentlyShown = Superpixels;
						break;
					case SDLK_KP_2:
						currentlyShown = SuperpixelsIntersection;
						break;
					case SDLK_KP_3:
						currentlyShown = Saliency;
						break;
					case SDLK_KP_4:
						currentlyShown = GMMLabelsBackground;
						break;
					case SDLK_KP_5:
						currentlyShown = GMMLabelsForeground;
						break;
					case SDLK_KP_6:
						currentlyShown = GMMWeightedLLBackground;
						break;
					case SDLK_KP_7:
						currentlyShown = GMMWeightedLLForeground;
						break;
					default:
						currentlyShown = BaseImage;
						break;
				}
			}

			//--------------------------------------------------------------------------------
			// Mouse button DOWN
			//--------------------------------------------------------------------------------
			if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
				clicking = true;
				clickCoord.x = e.motion.x;
				clickCoord.y = e.motion.y;
				for (int i = 0 ; i < 4 ; i++) {
					boundingBox.points[i].x = 0;
					boundingBox.points[i].y = 0;
				}
			}
			
			//--------------------------------------------------------------------------------
			// Mouse button UP
			//--------------------------------------------------------------------------------
			if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
				clicking = false;

				boundingBoxDrawn = true;
			}

			//--------------------------------------------------------------------------------
			// Mouse MOVING
			//--------------------------------------------------------------------------------
			if (clicking == true && e.type == SDL_MOUSEMOTION) {
				mousePosition.x = e.motion.x;
				mousePosition.y = e.motion.y;

				boundingBox.points[0].x = clickCoord.x;
				boundingBox.points[0].y = clickCoord.y;
				boundingBox.points[1].x = mousePosition.x;
				boundingBox.points[1].y = clickCoord.y;
				boundingBox.points[2].x = mousePosition.x;
				boundingBox.points[2].y = mousePosition.y;
				boundingBox.points[3].x = clickCoord.x;
				boundingBox.points[3].y = mousePosition.y;
			}
		}

		// Clear renderer
		SDL_RenderClear(ren);

		// Load image to show in SDL_Surface and texture
		switch(currentlyShown) {
			case BaseImage:
				convertCV_MatToSDL_Surface(&surf, img2);
				break;
			case Superpixels:
				if (superpixelsMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, superpixelsMat);
				}
				break;
			case SuperpixelsIntersection:
				if (superpixelsIntersectionMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, superpixelsIntersectionMat);
				}
				break;
			case Saliency:
				if (saliencyMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, saliencyMat);
				}
				break;
			case GMMLabelsBackground:
				if (GMMLabelsBackgroundMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, GMMLabelsBackgroundMat);
				}
				break;
			case GMMLabelsForeground:
				if (GMMLabelsForegroundMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, GMMLabelsForegroundMat);
				}
				break;
			case GMMWeightedLLBackground:
				if (GMMWeightedLLBackgroundMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, GMMWeightedLLBackgroundMat);
				}
				break;
			case GMMWeightedLLForeground:
				if (GMMWeightedLLForegroundMat != NULL) {
					convertCV_MatToSDL_Surface(&surf, GMMWeightedLLForegroundMat);
				}
				break;
			default:
				convertCV_MatToSDL_Surface(&surf, img2);
				break;
		}
		SDL_DestroyTexture(tex);
		tex = SDL_CreateTextureFromSurface(ren, surf);

		// Draw image
		SDL_RenderCopy(ren, tex, NULL, NULL);

		// Draw bounding box
		if (showBoundingBox) {
			SDL_SetRenderDrawColor(ren, 255, 0, 0, 255);
			for (int i = 0 ; i < 4 ; i++) {
				SDL_RenderDrawLine(ren,boundingBox.points[i].x, boundingBox.points[i].y, boundingBox.points[(i+1)%4].x, boundingBox.points[(i+1)%4].y);
			}
		}

		// Draw centroids
		if (drawCentroids) {
			showCentroids(ren, superpixels);
		}

		SDL_RenderPresent(ren);

		if (boundingBoxDrawn) {
			computeSuperpixelIntersectionWithBB(superpixels, boundingBox);
			convertSuperpixelsIntersectionToCV_Mat(&superpixelsIntersectionMat, superpixels, img.rows, img.cols);

			computeSaliencyMap(superpixels, boundingBox, img.rows, img.cols);
			convertSaliencyToCV_Mat(&saliencyMat, superpixels, img.rows, img.cols);

			struct GMM GMMBackground = computeGMM(superpixelsMat);
			convertGMMLabelsToCV_Mat(&GMMLabelsBackgroundMat, &GMMBackground, img.rows, img.cols);
			convertGMMWeightedLLToCV_Mat(&GMMWeightedLLBackgroundMat, &GMMBackground, img.rows, img.cols);

			currentlyShown = GMMLabelsBackground;

			boundingBoxDrawn = false;
		}
	}

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();

	free(superpixels);
	cvReleaseImage(&img2);
	delete ilabels;

	return 0;
}

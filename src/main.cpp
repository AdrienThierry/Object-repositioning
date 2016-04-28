#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <cmath>
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
#include "graph.h"
#include "PolygonalChain.hpp"
#include "Depth.hpp"

using namespace std;

typedef struct Superpixel Superpixel;
typedef struct Point Point;
typedef struct BoundingBox BoundingBox;
typedef struct GMM GMM;
typedef struct GMM_arg_struct GMM_arg_struct;
typedef struct PolygonalChain PolygonalChain;

int main( int argc, char** argv )
{
	string imagePath = "";

	enum State { WaitForGround, WaitForBB, Compute, ShowResult };
	enum WhatToShow { BaseImage, Superpixels, SuperpixelsIntersection, Saliency, 
					GMMWeightedProbsBackground, GMMWeightedProbsForeground, 
					SmoothnessLeftRight, SmoothnessTopBottom,
					ExtractedForeground };
	WhatToShow currentlyShown = BaseImage;
	State currentState = WaitForGround;
	bool drawCentroids = false;

	// Threads to compute GMMs
	pthread_t GMMBackgroundThread;
	pthread_t GMMForegroundThread;

	// Args for GMMBackgroundThread and GMMForegroundThread
	GMM_arg_struct GMM_args_foreground;
	GMM_arg_struct GMM_args_background;

	sem_t semGMMForeground; // Rendez-vous with GMMForegroundThread
	sem_t semGMMBackground; // Rendez-vous with GMMBackgroundThread

	sem_init(&semGMMForeground, 0, 0);
	sem_init(&semGMMBackground, 0, 0);

	//--------------------------------------------------------------------------------
	// Check arguments
	//--------------------------------------------------------------------------------
	if (argc > 2) {
		printf("ERROR : too many arguments");
		exit(EXIT_FAILURE);
	}
	else if (argc < 2) {
		printf("ERROR : too few arguments. Please enter the name of the image to load.");
		exit(EXIT_FAILURE);
	}
	else {
		imagePath = argv[1];
	}

	//--------------------------------------------------------------------------------
	// Image loading
	//--------------------------------------------------------------------------------
	cv::Mat img = cv::imread(imagePath); // Input image

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
	IplImage *superpixelsMat = NULL; // Superpixels
	convertSuperpixelsToCV_Mat(&superpixelsMat, superpixels, img.rows, img.cols);

	IplImage *superpixelsIntersectionMat = NULL; // Superpixels
	IplImage *saliencyMat = NULL; // Saliency map
	IplImage *GMMWeightedProbsBackgroundMat = NULL; // Background probs
	IplImage *GMMWeightedProbsForegroundMat = NULL; // Foreground probs
	IplImage *extractedForegroundMat = NULL; // Mat with only foreground but same size as original image
	IplImage *foregroundMat = NULL; // Mat with only foreground (size is foreground size)
	IplImage *foregroundMatScaled = NULL; // Mat with only scaled foreground (size is scaled foreground size)
	IplImage *foregroundWithDepthMat = NULL; // Final image
	IplImage *smoothnessLeftRightMat = NULL; // Left-right smoothness term
	IplImage *smoothnessTopBottomMat = NULL; // Top-bottom smoothness term

	SDL_Surface *surf = NULL; // Original image
	SDL_Surface *foregroundSurf = NULL; // Image with moved foreground
	SDL_Texture *tex = NULL;
	SDL_Texture *foregroundTex = NULL;

	//--------------------------------------------------------------------------------
	// Other variables declarations
	//--------------------------------------------------------------------------------
	GMM* GMMForeground = new GMM;
	GMM* GMMBackground = new GMM;

	struct Foreground foregroundStruct;

	PolygonalChain groundLine;
	groundLine.y = 0;

	SDL_Rect foregroundPosition;
	float ratio; // Foreground aspect ratio
	int originalHeight;
	int originalBottom;
	BoundingBox originalForegroundBB;

	bool clicking = false;
	bool movingForeground = false;

	BoundingBox boundingBox; // Bounding box drawn by the user

	vector<vector<float> > foregroundDepthMap;
	vector<vector<float> > backgroundDepthMap;

	int counter = 0; // Counter to prevent computing depth too often

	//--------------------------------------------------------------------------------
	// SDL main loop and bounding box handling
	//--------------------------------------------------------------------------------	
	SDL_Event e;
	for (int i = 0 ; i < 4 ; i++) {
		boundingBox.points[i].x = 0;
		boundingBox.points[i].y = 0;
	}
	bool quit = false;
	struct Point clickCoord, mousePosition, previousPosition, delta;
	while (!quit){
		while (SDL_PollEvent(&e)){
			if (e.type == SDL_QUIT){
				quit = true;
			}

			//--------------------------------------------------------------------------------
			// Keyboard key DOWN
			//--------------------------------------------------------------------------------
			if (e.type == SDL_KEYDOWN) {
				if (currentState == ShowResult) {
					drawCentroids = false;

					switch(e.key.keysym.sym) {
						case SDLK_KP_0:
							currentlyShown = BaseImage;
							break;
						case SDLK_KP_1:
							currentlyShown = Superpixels;
							drawCentroids = true;
							break;
						case SDLK_KP_2:
							currentlyShown = SuperpixelsIntersection;
							break;
						case SDLK_KP_3:
							currentlyShown = Saliency;
							break;
						case SDLK_KP_4:
							currentlyShown = GMMWeightedProbsBackground;
							break;
						case SDLK_KP_5:
							currentlyShown = GMMWeightedProbsForeground;
							break;
						case SDLK_KP_6:
							currentlyShown = SmoothnessLeftRight;
							break;
						case SDLK_KP_7:
							currentlyShown = SmoothnessTopBottom;
							break;
						case SDLK_KP_8:
							currentlyShown = ExtractedForeground;
							break;
						default:
							currentlyShown = BaseImage;
							break;
					}
				}
			}

			//--------------------------------------------------------------------------------
			// Mouse button DOWN
			//--------------------------------------------------------------------------------
			if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
				clicking = true;

				clickCoord.x = e.motion.x;
				clickCoord.y = e.motion.y;

				if (currentState == WaitForGround) {
					groundLine.y = e.motion.y;
				}

				else if (currentState == WaitForBB) {
					for (int i = 0 ; i < 4 ; i++) {
						boundingBox.points[i].x = 0;
						boundingBox.points[i].y = 0;
					}
				}

				else if (currentState == ShowResult) {
					// Check if the user clicked in the foreground region
					if (isInsideBB(clickCoord, foregroundStruct.bb)) {
						previousPosition.x = clickCoord.x;
						previousPosition.y = clickCoord.y;
						movingForeground = true;
					}
				}

			}
			
			//--------------------------------------------------------------------------------
			// Mouse button UP
			//--------------------------------------------------------------------------------
			if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
				clicking = false;

				if (currentState == WaitForGround) {
					currentState = WaitForBB;
				}

				else if (currentState == WaitForBB) {
					currentState = Compute;
				}

				else if (currentState == ShowResult) {
					movingForeground = false;
				}
			}

			//--------------------------------------------------------------------------------
			// Mouse MOVING
			//--------------------------------------------------------------------------------
			if (clicking == true && e.type == SDL_MOUSEMOTION) {
				mousePosition.x = e.motion.x;
				mousePosition.y = e.motion.y;

				if (currentState == WaitForBB) {
					boundingBox.points[0].x = clickCoord.x;
					boundingBox.points[0].y = clickCoord.y;
					boundingBox.points[1].x = mousePosition.x;
					boundingBox.points[1].y = clickCoord.y;
					boundingBox.points[2].x = mousePosition.x;
					boundingBox.points[2].y = mousePosition.y;
					boundingBox.points[3].x = clickCoord.x;
					boundingBox.points[3].y = mousePosition.y;
				}

				else if (currentState == ShowResult && movingForeground) {
					counter = (counter + 1) % 10;

					delta.x = mousePosition.x - previousPosition.x;
					delta.y = previousPosition.y - mousePosition.y;

					// Move X
					foregroundPosition.x += delta.x;

					// Move Y
					// Limit bottom line to ground line
					int previousH = foregroundPosition.h;
					int previousW = foregroundPosition.w;
					foregroundPosition.y = max(groundLine.y - foregroundPosition.h, foregroundPosition.y + delta.y);

					// Compute image size
					if (groundLine.y != ((foregroundPosition.y + foregroundPosition.h))) {

						foregroundPosition.h = originalHeight * 
							((float)(groundLine.y - (foregroundPosition.y + foregroundPosition.h)) / (float)(groundLine.y - originalBottom));

						foregroundPosition.w = (ratio * (float)foregroundPosition.h);

					}
					// Make image seem to have been resized by all sides uniformly
					foregroundPosition.y -= (foregroundPosition.h - previousH);
					foregroundPosition.x -= (foregroundPosition.w - previousW);

					// Update foreground bounding box
					Point p1, p2, p3, p4;
					p1.x = foregroundPosition.x;
					p1.y = foregroundPosition.y;
					p2.x = foregroundPosition.x + foregroundPosition.w;
					p2.y = foregroundPosition.y;
					p3.x = foregroundPosition.x + foregroundPosition.w;
					p3.y = foregroundPosition.y + foregroundPosition.h;
					p4.x = foregroundPosition.x;
					p4.y = foregroundPosition.y + foregroundPosition.h;
					foregroundStruct.bb.points[0] = p1;
					foregroundStruct.bb.points[1] = p2;
					foregroundStruct.bb.points[2] = p3;
					foregroundStruct.bb.points[3] = p4;
					computeBBEnds(&(foregroundStruct.bb));

					if (counter == 9) {

						// Scale foreground IplImage according to new dimensions
						if (foregroundMatScaled != NULL)
							cvReleaseImage(&foregroundMatScaled);
						foregroundMatScaled = cvCreateImage(cvSize(foregroundPosition.w, foregroundPosition.h), foregroundMat->depth, foregroundMat->nChannels);
						cvResize(foregroundMat, foregroundMatScaled);
						updateExtractedForegroundMat(&extractedForegroundMat, foregroundMatScaled, &foregroundPosition, img.rows, img.cols);

						// Compute depth map
						foregroundDepthMap = computeDepthMap(&groundLine, &foregroundPosition, extractedForegroundMat, img.rows, img.cols);

						// Compute final image
						computeFinalImage(&backgroundDepthMap, &foregroundDepthMap, &foregroundWithDepthMat, extractedForegroundMat, img2, img.rows, img.cols);

					}
					
				}

				previousPosition = mousePosition;
			}
		}

		// Clear renderer
		SDL_RenderClear(ren);


		if (currentState == WaitForGround || currentState == WaitForBB || currentState == Compute) {
			convertCV_MatToSDL_Surface(&surf, img2);
		}
		if (currentState == ShowResult) {
			// Load image to show in SDL_Surface and texture
			switch(currentlyShown) {
				case BaseImage:
					convertCV_MatToSDL_Surface(&surf, img2);
					if (foregroundMat != NULL) {
						convertCV_Mat_WithAlpha_ToSDL_Surface(&foregroundSurf, foregroundWithDepthMat);
					}
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
				case GMMWeightedProbsBackground:
					if (GMMWeightedProbsBackgroundMat != NULL) {
						convertCV_MatToSDL_Surface(&surf, GMMWeightedProbsBackgroundMat);
					}
					break;
				case GMMWeightedProbsForeground:
					if (GMMWeightedProbsForegroundMat != NULL) {
						convertCV_MatToSDL_Surface(&surf, GMMWeightedProbsForegroundMat);
					}
					break;
				case SmoothnessLeftRight:
					if (smoothnessLeftRightMat != NULL) {
						convertCV_MatToSDL_Surface(&surf, smoothnessLeftRightMat);
					}
					break;
				case SmoothnessTopBottom:
					if (smoothnessTopBottomMat != NULL) {
						convertCV_MatToSDL_Surface(&surf, smoothnessTopBottomMat);
					}
					break;
				case ExtractedForeground:
					if (extractedForegroundMat != NULL) {
						convertCV_MatToSDL_Surface(&surf, extractedForegroundMat);
					}
					break;
				default:
					convertCV_MatToSDL_Surface(&surf, img2);
					break;
			}
		}
		SDL_DestroyTexture(tex);
		tex = SDL_CreateTextureFromSurface(ren, surf);

		if (foregroundSurf != NULL) {
			SDL_DestroyTexture(foregroundTex);
			foregroundTex = SDL_CreateTextureFromSurface(ren, foregroundSurf);
		}

		// Draw image
		SDL_RenderCopy(ren, tex, NULL, NULL);
		if (foregroundTex != NULL && currentlyShown == BaseImage) {
			SDL_RenderCopy(ren, foregroundTex, NULL, NULL);
		}

		// Draw ground line
		if (currentState == WaitForGround || currentState == WaitForBB || currentState == Compute) {
			SDL_SetRenderDrawColor(ren, 0, 0, 255, 255);
			SDL_RenderDrawLine(ren, 0, groundLine.y, img.cols, groundLine.y);
		}

		// Draw bounding box
		if (currentState == WaitForBB || currentState == Compute || (currentState == ShowResult && (currentlyShown == SuperpixelsIntersection || currentlyShown == Saliency))) {
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

		if (currentState == Compute) {

			// Compute superpixels intersection with bounding box
			computeSuperpixelIntersectionWithBB(superpixels, boundingBox);
			convertSuperpixelsIntersectionToCV_Mat(&superpixelsIntersectionMat, superpixels, img.rows, img.cols);

			// Compute saliency map
			computeSaliencyMap(superpixels, boundingBox, img.rows, img.cols);
			convertSaliencyToCV_Mat(&saliencyMat, superpixels, img.rows, img.cols);

			// Compute saliency LUT
			std::vector<float> saliencyForegroundLUT = getSaliencyForegroundLUT(superpixels, boundingBox);
			std::vector<float> saliencyBackgroundLUT = getSaliencyBackgroundLUT(superpixels, img.rows, img.cols);

			// Set args for background GMM computation
			GMM_args_background.result = GMMBackground;
			GMM_args_background.imageIpl = superpixelsMat;
			GMM_args_background.saliencyLUT = &saliencyBackgroundLUT;
			GMM_args_background.semaphore = &semGMMBackground;

			// Create an image that only contains the pixels inside the bounding box for foreground GMM computation
			IplImage foreground = preProcessingForegroundGMM(superpixelsMat, boundingBox);

			// Set args for foreground GMM computation
			GMM_args_foreground.result = GMMForeground;
			GMM_args_foreground.imageIpl = &foreground;
			GMM_args_foreground.saliencyLUT = &saliencyForegroundLUT;
			GMM_args_foreground.semaphore = &semGMMForeground;

			// Launch GMM computation threads
			pthread_create(&GMMBackgroundThread, NULL, &computeGMM, (void*)&GMM_args_background);
			pthread_create(&GMMForegroundThread, NULL, &computeGMM, (void*)&GMM_args_foreground);

			// Wait for GMM computation threads to finish
			sem_wait(&semGMMBackground);
			sem_wait(&semGMMForeground);

			// Create OpenCV matrices to see background GMM result
			convertGMMWeightedProbsToCV_Mat(&GMMWeightedProbsBackgroundMat, GMMBackground, img.rows, img.cols);

			// Create OpenCV matrices to see foreground GMM result
			postProcessingForegroundGMM(GMMForeground, boundingBox, img.rows, img.cols);			
			convertGMMWeightedProbsToCV_Mat(&GMMWeightedProbsForegroundMat, GMMForeground, img.rows, img.cols);

			// Graph cut to extract foreground
			foregroundStruct = extractForeground(
					img2, 
					&extractedForegroundMat,
					&smoothnessLeftRightMat,
					&smoothnessTopBottomMat,
					GMMForeground,
					GMMBackground,
					boundingBox,
					img.rows,
					img.cols );

			// Compute movable foreground image
			computeForegroundBB(&foregroundStruct);
			computeForegroundImage(img2, &foregroundMat, &foregroundStruct);

			// Initial foreground position
			foregroundPosition.x = foregroundStruct.bb.ends[0];
			foregroundPosition.y = foregroundStruct.bb.ends[2];
			foregroundPosition.h = foregroundStruct.bb.ends[3] - foregroundStruct.bb.ends[2];
			foregroundPosition.w = foregroundStruct.bb.ends[1] - foregroundStruct.bb.ends[0];

			originalForegroundBB = foregroundStruct.bb;

			ratio = (float)foregroundPosition.w / (float)foregroundPosition.h;
			originalHeight = foregroundPosition.h;
			originalBottom = foregroundPosition.y + originalHeight;

			// Background depth map
			backgroundDepthMap = computeDepthMap(&groundLine, &foregroundPosition, extractedForegroundMat, img.rows, img.cols);

			foregroundDepthMap = computeDepthMap(&groundLine, &foregroundPosition, extractedForegroundMat, img.rows, img.cols);
			computeFinalImage(&backgroundDepthMap, &foregroundDepthMap, &foregroundWithDepthMat, extractedForegroundMat, img2, img.rows, img.cols);

			currentlyShown = ExtractedForeground;

			currentState = ShowResult;

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

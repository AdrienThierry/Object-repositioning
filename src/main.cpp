#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <SDL2/SDL.h>

#include "opencv_2_sdl.hpp"
#include "Superpixel.hpp"
#include "Point.hpp"
#include "BoundingBox.hpp"

using namespace std;

typedef struct Superpixel Superpixel;
typedef struct Point Point;
typedef struct BoundingBox BoundingBox;

int main( int argc, char** argv )
{
	cv::Mat img = cv::imread("data/outside.jpg");
	cv::Mat outImg;

	bool boundingBoxDrawn = false;

	//--------------------------------------------------------------------------------
	// SDL Initialisation
	//--------------------------------------------------------------------------------
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		cout << "SDL_Init Error: " << SDL_GetError() << endl;
		return 1;
	}

	SDL_Window *win = SDL_CreateWindow("Hello World!", 100, 100, img.cols, img.rows, SDL_WINDOW_SHOWN);
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
	pyrMeanShiftFiltering(img, outImg, 50, 50, 3);
	//outImg = img;

	//--------------------------------------------------------------------------------
	// Superpixels generation
	//--------------------------------------------------------------------------------
	vector<Superpixel> *superpixels = computeSuperpixels(outImg);

	//--------------------------------------------------------------------------------
	// TEST : show superpixels
	//--------------------------------------------------------------------------------	
	cv::Mat superpixelsMat = convertSuperpixelsToCV_Mat(superpixels, img.rows, img.cols);
	cv::Mat superpixelsIntersectionMat;

	SDL_Surface *surf = convertCV_MatToSDL_Surface(superpixelsMat);
	SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surf);


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

			if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
				clicking = true;
				clickCoord.x = e.motion.x;
				clickCoord.y = e.motion.y;
				for (int i = 0 ; i < 4 ; i++) {
					boundingBox.points[i].x = 0;
					boundingBox.points[i].y = 0;
				}

				surf = convertCV_MatToSDL_Surface(superpixelsMat);
				tex = SDL_CreateTextureFromSurface(ren, surf);

			}
			
			if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
				clicking = false;

				boundingBoxDrawn = true;
			}

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

		//Render the scene
		SDL_RenderClear(ren);

		// Draw image
		SDL_RenderCopy(ren, tex, NULL, NULL);

		// Draw bounding box
		SDL_SetRenderDrawColor(ren, 255, 0, 0, 255);
		for (int i = 0 ; i < 4 ; i++) {
			SDL_RenderDrawLine(ren,boundingBox.points[i].x, boundingBox.points[i].y, boundingBox.points[(i+1)%4].x, boundingBox.points[(i+1)%4].y);
		}

		SDL_RenderPresent(ren);

		if (boundingBoxDrawn) {
			computeSuperpixelIntersectionWithBB(superpixels, boundingBox);
			superpixelsIntersectionMat = convertSuperpixelsIntersectionToCV_Mat(superpixels, img.rows, img.cols);

			surf = convertCV_MatToSDL_Surface(superpixelsIntersectionMat);
			tex = SDL_CreateTextureFromSurface(ren, surf);

			boundingBoxDrawn = false;
		}
	}

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();

	free(superpixels);

	return 0;
}

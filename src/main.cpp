#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <SDL2/SDL.h>

#include "opencv_2_sdl.hpp"
#include "Superpixel.hpp"

using namespace cv;
using namespace std;

typedef struct Superpixel Superpixel;

int main( int argc, char** argv )
{
	Mat img = imread("data/outside.jpg");
	Mat outImg;

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

	//--------------------------------------------------------------------------------
	// Superpixels generation
	//--------------------------------------------------------------------------------
	vector<Superpixel> *superpixels = computeSuperpixels(outImg);

	//--------------------------------------------------------------------------------
	// TEST : show superpixels
	//--------------------------------------------------------------------------------	
	Mat superpixelsMat = convertSuperpixelsToCV_Mat(superpixels, img.rows, img.cols);

	SDL_Surface *surf = convertCV_MatToSDL_Surface(superpixelsMat);
	SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surf);

	SDL_Event e;
	bool quit = false;
	while (!quit){
		while (SDL_PollEvent(&e)){
			if (e.type == SDL_QUIT){
				quit = true;
			}
		}
		//Render the scene
		SDL_RenderClear(ren);

		SDL_RenderCopy(ren, tex, NULL, NULL);

		showCentroids(ren, superpixels);

		SDL_RenderPresent(ren);
	}

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();

	free(superpixels);

	return 0;
}

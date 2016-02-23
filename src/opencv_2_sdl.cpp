#include "opencv_2_sdl.hpp"

SDL_Surface* convertCV_MatToSDL_Surface(const cv::Mat &matImg)
{
	IplImage opencvimg2 = (IplImage)matImg;
	IplImage* opencvimg = &opencvimg2;

	// Convert to SDL_Surface
	SDL_Surface *frameSurface = SDL_CreateRGBSurfaceFrom(
						(void*)opencvimg->imageData,
						opencvimg->width, opencvimg->height,
						opencvimg->depth*opencvimg->nChannels,
						opencvimg->widthStep,
						0xff0000, 0x00ff00, 0x0000ff, 0);

	if(frameSurface == NULL)
	{
		SDL_Log("Couldn't convert Mat to Surface.");
		return NULL;
	}

	else {
		return frameSurface;
	}

	cvReleaseImage(&opencvimg);
}

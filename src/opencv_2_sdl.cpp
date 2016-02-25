#include <unistd.h>
#include <stdio.h>
#include "opencv_2_sdl.hpp"

void convertCV_MatToSDL_Surface(SDL_Surface **result, IplImage *image)
{
	if (*result != NULL)
		SDL_FreeSurface(*result);

	// Convert to SDL_Surface
	*result = SDL_CreateRGBSurfaceFrom(
						(void*)image->imageData,
						image->width, image->height,
						image->depth*image->nChannels,
						image->widthStep,
						0xff0000, 0x00ff00, 0x0000ff, 0);

}

#ifndef COLOR_H_   /* Include guard */
#define COLOR_H_

struct Color {
	int r;
	int g;
	int b;
};

struct ColorLab {
	int L;
	int a;
	int b;
};

struct ColorLab convertRGB2Lab(struct Color color);

#endif

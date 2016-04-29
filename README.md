# Object repositioning
Interface to move objects in an image in a user-friendly way.

Basically, this program implements part of the method described in the article *Object Repositioning Based on the Perspective in a Single Image* by S.Iizuka, Y. Endo, M. Hirose, Y. Kanamori, J. Mitani, Y. Fukui. (http://www.npal.cs.tsukuba.ac.jp/~iizuka/projects/reposition/data/cgf_repos.pdf)

## Dependencies
* cmake 
* SDL2
* OpenCV (The project uses version 2.4.12.3-1. It may not work with OpenCV 3)

## Compilation
```
cmake .
make
```

## Use

Input images are available in the **images** folder

Syntax :
```
./object_repositioning image_to_load
```
For example :
```
./object_repositioning images/beach.jpg
```

1. **Wait** for a few seconds until the image is shown (superpixels computation is done at launch)

2. Click where you want to place the **ground line** (max depth in the scene)

3. **Draw a bounding box** around the object you want to move

4. **Wait** (~1min depending on your processing power). When computation is finished, only the segmented object is shown.

5. Use the **keypad** to go through all the visualization modes

  0 - Interactive mode. You can **drag&drop** a copy of the object you chose in the scene. The size of the object changed in real-time according to where you place it
  
  1 - Superpixels
  
  2 -	Intersection between superpixels and bounding box 
    * **White** : superpixels *outside* the bounding box
    * **Black** : superpixels which *intersect* the bounding box
    * **Grey** : superpixels *inside* the bounding box
    
   3 -	Saliency map
   
   4 - Background probabilities (the more white a pixel is, the more likely it is to belong to the background)
   
	5 - Foreground probabilities (the more white a pixel is, the more likely it is to belong to the foreground)
	
	6 - Smoothness term for graph cut (weight between a pixel and its right neighbor)
	
	7 - Smoothness term for graph cut (weight between a pixel and its bottom neighbor)
	
	8 - Segmentation result (extracted object)

cmake_minimum_required(VERSION 2.8)
project( object_repositioning )

add_definitions ("-Wall")
add_definitions ("-Wno-write-strings")
add_definitions ("-Wno-parentheses")

INCLUDE(FindPkgConfig)
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

find_package( OpenCV REQUIRED )
add_executable( object_repositioning src/main.cpp src/opencv_2_sdl.cpp src/Superpixel.cpp src/BoundingBox.cpp src/MeanShift.cpp src/Color.cpp src/foreground_extraction.cpp src/GMM.cpp src/graph.cpp src/maxflow.cpp )
target_link_libraries( object_repositioning ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${SDL2_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(object_repositioning ${SDL2_LIBRARIES})

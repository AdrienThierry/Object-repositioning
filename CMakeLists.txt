cmake_minimum_required(VERSION 2.8)
project( test_opencv )

INCLUDE(FindPkgConfig)
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

find_package( OpenCV REQUIRED )
add_executable( test_opencv src/main.cpp src/opencv_2_sdl.cpp src/Superpixel.cpp)
target_link_libraries( test_opencv ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${SDL2_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(test_opencv ${SDL2_LIBRARIES})

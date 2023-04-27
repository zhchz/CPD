cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall  -O3  -fopenmp")

# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()
# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PLATFORM_COMPILE_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PLATFORM_COMPILE_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS}  -g2 -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}  -g2  -ggdb")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(NOT TARGET_CXX_SUPPORT)
   set(TARGET_CXX_SUPPORT "11")
endif()

if(${TARGET_CXX_SUPPORT} STREQUAL "17")
   if(COMPILER_SUPPORTS_CXX17)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
      add_definitions(-DCOMPILEDWITHC17)
      message(STATUS "Using flag -std=c++17")
   else()
      message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no c++17 support!")
   endif()
elseif(${TARGET_CXX_SUPPORT} STREQUAL "14")
   if(COMPILER_SUPPORTS_CXX14)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
      add_definitions(-DCOMPILEDWITHC14)
      message(STATUS "Using flag -std=c++14")
   else()
      message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no c++14 support!")
   endif()  
elseif(${TARGET_CXX_SUPPORT} STREQUAL "11")
   if(COMPILER_SUPPORTS_CXX11)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      add_definitions(-DCOMPILEDWITHC11)
      message(STATUS "Using flag -std=c++11")
   else()
      message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no c++11 support!")
   endif()
else()
   message(FATAL_ERROR "Choose C11, C14 or C17 only!")
endif()

include_directories(
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include
   ${PROJECT_SOURCE_DIR}/3rd/include
   /opt/libraries/opencv4.5.4/include/opencv4
   /opt/libraries/Eigen/include
   /usr/lib/openmpi/include
)

link_directories(
   ${PROJECT_SOURCE_DIR}/3rd/lib
   /opt/libraries/opencv4.5.4/lib
   /usr/lib/openmpi/lib
   /usr/lib/x86_64-linux-gnu
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
   set(EXECUTABLE_NAME ${PROJECT_NAME}_debug)
else()
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
   set(EXECUTABLE_NAME ${PROJECT_NAME})
endif()

add_executable(${EXECUTABLE_NAME} 
${PROJECT_NAME}.cpp
${SRC}
)

target_link_libraries(${EXECUTABLE_NAME}
# libopencv_objdetect.so
# libopencv_dnn.so
# libopencv_ml.so
# libopencv_shape.so
# libopencv_stitching.so
# libopencv_viz.so
# libopencv_videostab.so
# libopencv_calib3d.so
# libopencv_features2d.so
# libopencv_flann.so
# libopencv_superres.so
# libopencv_video.so
# libopencv_photo.so
# libopencv_highgui.so
# libopencv_videoio.so
# libopencv_imgcodecs.so
# libopencv_imgproc.so
# libopencv_core.so
libopencv_world.so
-lpthread
)

if(COMPILER_SUPPORTS_CXX17)
   target_link_libraries(${EXECUTABLE_NAME}
   libmatplot.a
   )
endif()
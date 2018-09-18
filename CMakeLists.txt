cmake_minimum_required( VERSION 2.8 )
project(lkicp)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( OpenCV 3.1 REQUIRED)
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_executable(lkicp lkicp.cpp)
target_link_libraries(lkicp    ${OpenCV_LIBS}    
            g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension    
            ${CSPARSE_LIBRARY})


cmake_minimum_required(VERSION 3.16)
project(imagecmp_next LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d features2d video)

add_library(imagecmp_next
    src/image_comparison_service.cpp
)
target_include_directories(imagecmp_next PUBLIC include)
target_link_libraries(imagecmp_next PUBLIC ${OpenCV_LIBS})

add_executable(imagecmp_characterize src/characterize_main.cpp)
target_link_libraries(imagecmp_characterize PRIVATE imagecmp_next)

include(CTest)
if(BUILD_TESTING)
    add_executable(imagecmp_next_tests tests/image_comparison_service_test.cpp)
    target_link_libraries(imagecmp_next_tests PRIVATE imagecmp_next)
    add_test(NAME imagecmp_next_tests COMMAND imagecmp_next_tests)
endif()

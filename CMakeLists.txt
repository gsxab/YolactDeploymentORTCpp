cmake_minimum_required(VERSION 3.12)
project(onnxnettest)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# conda
set(Python3_ROOT_DIR "$ENV{HOME}/software/yolact/venv/yolact-env/")
find_package(Python3 COMPONENTS Development REQUIRED)

set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(onnxnettest main.cpp common.cpp common.h)
target_include_directories(onnxnettest PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnxnettest PUBLIC ${OpenCV_LIBS})

set(ONNXRUNTIME_INCLUDE_DIRS /usr/local/include/onnxruntime/)
set(ONNXRUNTIME_LIBRARY /usr/local/lib/libonnxruntime.so)

add_executable(onnxnettest2 main2.cpp common.cpp common.h)
target_include_directories(onnxnettest2 PUBLIC
        ${ONNXRUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        )
target_link_directories(onnxnettest2 PUBLIC ../yolact/external/DCNv2/)
target_compile_definitions(onnxnettest2 PUBLIC USE_CUDA WITH_CUDA _GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(onnxnettest2 PUBLIC
        -l:_ext.cpython-37m-x86_64-linux-gnu.so
        ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} tiff ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        )


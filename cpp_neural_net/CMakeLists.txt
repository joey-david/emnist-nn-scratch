cmake_minimum_required(VERSION 3.18)
project(music_cuda_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(music_cuda_nn src/main.cu)
set_target_properties(music_cuda_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(music_cuda_nn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
)

find_package(CUDAToolkit REQUIRED)


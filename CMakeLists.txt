option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_TESTING "third party tests" OFF)

cmake_minimum_required(VERSION 3.28)

include(CMakeDependentOption)
cmake_dependent_option(BUILD_TEST_CORE "Build tests for core components" ON BUILD_TEST OFF)
project(infini_train VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

set(WITH_GFLAGS OFF CACHE BOOL "Disable glog finding system gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable glog finding system gtest" FORCE)

# Add glog
add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# Add eigen
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    include_directories(/opt/homebrew/opt/libomp/include)
endif()
find_package(OpenMP REQUIRED)
if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/eigen/CMakeLists.txt)
    add_subdirectory(third_party/eigen)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
else()
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
list(FILTER SRC EXCLUDE REGEX ".*kernels/cpu/.*")

file (GLOB_RECURSE CPU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cpu/*.cc)
add_library(infini_train_cpu_kernels STATIC ${CPU_KERNELS})
target_link_libraries(infini_train_cpu_kernels glog Eigen3::Eigen OpenMP::OpenMP_CXX)

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    include(FindCUDAToolkit)

    # enable CUDA-related compilation options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
    set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;80")
    target_link_libraries(infini_train_cuda_kernels glog CUDA::cudart CUDA::cublas)

    add_library(infini_train STATIC ${SRC})
    if(APPLE)
        target_link_libraries(infini_train glog gflags
            "-Wl,-force_load,$<TARGET_FILE:infini_train_cpu_kernels>"
            "-Wl,-force_load,$<TARGET_FILE:infini_train_cuda_kernels>")
    else()
        target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels infini_train_cuda_kernels "-Wl,--no-whole-archive")
    endif()
else()
    add_library(infini_train STATIC ${SRC})
    if(APPLE)
        target_link_libraries(infini_train glog gflags OpenMP::OpenMP_CXX
            "-Wl,-force_load,$<TARGET_FILE:infini_train_cpu_kernels>")
    else()
        target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels "-Wl,--no-whole-archive")
    endif()
endif()

if(BUILD_TEST)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Do not build gmock" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "Do not install gtest" FORCE)
  add_subdirectory(third_party/googletest)
  include_directories(third_party/googletest/googletest/include)
endif()

add_library(example_gpt2 STATIC
    example/common/tiny_shakespeare_dataset.cc
    example/common/tokenizer.cc
    example/gpt2/net.cc
)
target_link_libraries(example_gpt2 infini_train) 

function(build_test files)
  # Non-recursive glob for skip failed tests
  file(GLOB TEST_SOURCES ${files})
  foreach(testsourcefile ${TEST_SOURCES})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} infini_train example_gpt2 GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach(testsourcefile ${TEST_SOURCES})
endfunction()

if(BUILD_TEST)
  add_compile_definitions(BUILD_TEST=1)
  enable_testing()
  if(BUILD_TEST_CORE)
    build_test(test/autograd/test_elementwise.cc)
    build_test(test/kernels/test_matmul.cc)
    build_test(test/kernels/test_dispatcher.cc)
    build_test(test/tensor/test_tensor.cc)
    build_test(test/optimizer/test_adam.cc)
    build_test(test/example/test_gpt2.cc)
    if(USE_CUDA)
      build_test(test/kernels/test_matmul_cuda.cc)
      build_test(test/optimizer/test_adam_cuda.cc)
    endif()
  endif()
endif()
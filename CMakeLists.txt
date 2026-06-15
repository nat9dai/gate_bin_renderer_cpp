cmake_minimum_required(VERSION 3.8)
project(gate_bin_renderer_cpp LANGUAGES CXX)

# Latency matters (publishes at the control rate) — default to an optimized build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Scope to C++ only: nvcc-generated #line directives trip -Wpedantic.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenMP QUIET)

# ───────────────────────── optional CUDA (GPU backend) ──────────────────────
# Builds the CUDA backend when a CUDA toolkit + compiler are available; falls
# back to a CPU(+OpenMP)-only build otherwise. Pass -DGBR_DISABLE_CUDA=ON to
# force CPU-only, or -DCMAKE_CUDA_ARCHITECTURES=87 to pin the arch (e.g. Orin).
option(GBR_DISABLE_CUDA "Force a CPU-only build (no CUDA backend)" OFF)
set(GBR_HAVE_CUDA FALSE)
if(NOT GBR_DISABLE_CUDA)
  include(CheckLanguage)
  # Help CMake find nvcc when it isn't on PATH (the common colcon case): probe
  # the standard CUDA install locations and the CUDA_HOME/CUDA_PATH env vars.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
    find_program(_GBR_NVCC nvcc
      HINTS ENV CUDA_HOME ENV CUDA_PATH /usr/local/cuda /usr/local/cuda-13
            /usr/local/cuda-12 /usr/local/cuda-11
      PATH_SUFFIXES bin)
    if(_GBR_NVCC)
      set(CMAKE_CUDA_COMPILER "${_GBR_NVCC}")
      get_filename_component(_GBR_CUDA_BIN "${_GBR_NVCC}" DIRECTORY)
      get_filename_component(CUDAToolkit_ROOT "${_GBR_CUDA_BIN}" DIRECTORY)
    endif()
  endif()
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      # Resolve the GPU arch BEFORE enable_language(CUDA): on CMake 3.18-3.21 the
      # compiler/test compile is configured at enable_language time, so a late
      # CMAKE_CUDA_ARCHITECTURES can be ignored. We also fall back to a -gencode
      # flag on CMake < 3.18, which predates CMAKE_CUDA_ARCHITECTURES entirely
      # (keeps cmake_minimum_required at 3.8 so the CPU build stays portable).
      set(_GBR_ARCH "")
      if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_GBR_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
      else()
        find_program(_NVSMI nvidia-smi)
        if(_NVSMI)
          execute_process(
            COMMAND ${_NVSMI} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _cc RESULT_VARIABLE _rc ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
          if(_rc EQUAL 0 AND _cc)
            string(REGEX MATCH "[0-9]+\\.[0-9]+" _cc0 "${_cc}")
            string(REPLACE "." "" _GBR_ARCH "${_cc0}")
          endif()
        endif()
        if(NOT _GBR_ARCH)
          if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(_GBR_ARCH 87)   # Jetson Orin (Ampere, sm_87)
          else()
            set(_GBR_ARCH 75)
          endif()
        endif()
        if(CMAKE_VERSION VERSION_LESS 3.18)
          set(CMAKE_CUDA_FLAGS
              "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${_GBR_ARCH},code=sm_${_GBR_ARCH}")
        else()
          set(CMAKE_CUDA_ARCHITECTURES ${_GBR_ARCH})
        endif()
      endif()
      enable_language(CUDA)
      set(GBR_HAVE_CUDA TRUE)
      message(STATUS "gate_bin_renderer_cpp: CUDA backend ON (sm_${_GBR_ARCH})")
    endif()
  endif()
endif()
if(NOT GBR_HAVE_CUDA)
  message(STATUS "gate_bin_renderer_cpp: CUDA backend OFF — CPU-only build")
endif()

# ───────────────────────────── core render library ─────────────────────────
set(GBR_CORE_SRC src/camera_presets.cpp src/render_cpu.cpp)
if(GBR_HAVE_CUDA)
  list(APPEND GBR_CORE_SRC src/render_cuda.cu)
endif()

add_library(gate_render_core ${GBR_CORE_SRC})
target_include_directories(gate_render_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(gate_render_core PUBLIC cxx_std_17)
set_target_properties(gate_render_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(GBR_HAVE_CUDA)
  target_compile_definitions(gate_render_core PUBLIC GBR_HAVE_CUDA)
  set_target_properties(gate_render_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(gate_render_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─────────────────────────────── ROS 2 node ────────────────────────────────
add_executable(gate_bin_renderer_node src/gate_bin_renderer_node.cpp)
target_compile_features(gate_bin_renderer_node PUBLIC cxx_std_17)
target_link_libraries(gate_bin_renderer_node gate_render_core)
ament_target_dependencies(gate_bin_renderer_node
  rclcpp px4_msgs geometry_msgs nav_msgs sensor_msgs std_msgs
  visualization_msgs tf2_ros)

# ─────────────────── offline CLI (parity test / debugging) ──────────────────
add_executable(gate_render_cli src/gate_render_cli.cpp)
target_compile_features(gate_render_cli PUBLIC cxx_std_17)
target_link_libraries(gate_render_cli gate_render_core)

install(TARGETS gate_bin_renderer_node gate_render_cli
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS gate_render_core
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The OAK-FFC extrinsics constants exceed the copyright linter's comfort; the
  # project is BSD-3 with headers, so skip the auto copyright/cpplint hooks.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

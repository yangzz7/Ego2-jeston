cmake_minimum_required(VERSION 3.16)
project(cyperstereo_isp_portable VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(CISP_BUILD_EXAMPLES "Build offline raw-to-PNG example" ON)
option(CISP_BUILD_TESTS "Build ISP, ABI and parameter checks" ON)
option(CISP_NATIVE_OPTIMIZATION "Optimize for the build host CPU (less portable)" OFF)
option(CISP_CUDA_DISPLAY "Build the optional CUDA display ISP" OFF)
# Avoid implicitly mixing an activated Conda OpenCV with the system compiler.
# An explicitly supplied OpenCV_DIR takes precedence.
if(NOT DEFINED OpenCV_DIR AND
   EXISTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/opencv4/OpenCVConfig.cmake")
  set(OpenCV_DIR "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/opencv4" CACHE PATH "OpenCV package")
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
find_package(Threads REQUIRED)
file(GLOB ISP_MODULES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/isp/sources/modules/*.cpp")

add_library(cyperstereo_isp STATIC
  src/tunable_isp.cpp src/isp_frame_storage.cpp src/parameters.cpp src/processor.cpp
  ${ISP_MODULES}
  vendor/isp/sources/common/common.cpp
  vendor/isp/sources/common/modules.cpp
  vendor/isp/sources/common/pipeline.cpp
  vendor/easylog/easylogging++.cc vendor/uvc/fast_isp.cc)
add_library(CyperStereo::isp ALIAS cyperstereo_isp)
set_target_properties(cyperstereo_isp PROPERTIES POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES EXPORT_NAME isp)
target_compile_features(cyperstereo_isp PUBLIC cxx_std_17)
target_include_directories(cyperstereo_isp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include> ${OpenCV_INCLUDE_DIRS})
target_include_directories(cyperstereo_isp PRIVATE
  include/camera_tuner vendor/uvc vendor/isp/include vendor/easylog)
target_compile_definitions(cyperstereo_isp PRIVATE ELPP_THREAD_SAFE ELPP_NO_DEFAULT_LOG_FILE)
target_link_libraries(cyperstereo_isp PUBLIC opencv_core opencv_imgproc Threads::Threads)
if(CISP_CUDA_DISPLAY)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(cyperstereo_isp PRIVATE src/cuda_display.cu)
  target_compile_definitions(cyperstereo_isp PRIVATE CISP_CUDA_DISPLAY)
  target_link_libraries(cyperstereo_isp PUBLIC CUDA::cudart)
  set_target_properties(cyperstereo_isp PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  target_compile_options(cyperstereo_isp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

add_library(cyperstereo_isp_c SHARED src/c_api.cpp)
add_library(CyperStereo::c_api ALIAS cyperstereo_isp_c)
target_include_directories(cyperstereo_isp_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cyperstereo_isp_c PRIVATE cyperstereo_isp)
target_compile_definitions(cyperstereo_isp_c PRIVATE CISP_BUILDING_LIBRARY)
set_target_properties(cyperstereo_isp_c PROPERTIES CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES VERSION 1.0.0 SOVERSION 1 EXPORT_NAME c_api)
if(UNIX AND NOT APPLE)
  target_link_options(cyperstereo_isp_c PRIVATE "-Wl,--exclude-libs,ALL")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cyperstereo_isp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
  if(CISP_NATIVE_OPTIMIZATION AND NOT CMAKE_CROSSCOMPILING)
    target_compile_options(cyperstereo_isp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>)
  endif()
endif()

if(CISP_BUILD_EXAMPLES)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
  add_executable(isp_convert examples/convert_raw.cpp)
  target_link_libraries(isp_convert PRIVATE CyperStereo::isp opencv_imgcodecs)
endif()
if(CISP_BUILD_TESTS)
  enable_testing()
  foreach(name isp parameters transport)
    add_executable(cisp_test_${name} tests/test_${name}.cpp)
    target_include_directories(cisp_test_${name} PRIVATE include/camera_tuner)
    target_link_libraries(cisp_test_${name} PRIVATE CyperStereo::isp)
    add_test(NAME ${name} COMMAND cisp_test_${name} "${CMAKE_CURRENT_SOURCE_DIR}/configs/s200032.yaml")
  endforeach()
  add_executable(cisp_test_abi tests/test_abi.cpp)
  target_link_libraries(cisp_test_abi PRIVATE cyperstereo_isp_c)
  add_test(NAME c_abi COMMAND cisp_test_abi)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy" RESULT_VARIABLE CISP_NUMPY_STATUS)
    if(CISP_NUMPY_STATUS EQUAL 0)
      add_test(NAME python_abi COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        "PYTHONDONTWRITEBYTECODE=1"
        "CYPERSTEREO_ISP_LIBRARY=$<TARGET_FILE:cyperstereo_isp_c>"
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py")
    endif()
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS cyperstereo_isp cyperstereo_isp_c EXPORT CyperStereoIspTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY configs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/cyperstereo_isp/configs)
install(FILES LICENSE THIRD_PARTY_NOTICES.md SOURCE_MANIFEST.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cyperstereo_isp)
install(DIRECTORY LICENSES/ DESTINATION ${CMAKE_INSTALL_DATADIR}/cyperstereo_isp/LICENSES)
install(EXPORT CyperStereoIspTargets NAMESPACE CyperStereo:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CyperStereoIsp)
configure_package_config_file(cmake/CyperStereoIspConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CyperStereoIspConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CyperStereoIsp)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/CyperStereoIspConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CyperStereoIspConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CyperStereoIspConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CyperStereoIsp)

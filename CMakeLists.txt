cmake_minimum_required(VERSION 3.16)
project(ctc_beam_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Everything lands in the build directory itself, not in a per-configuration
# subdirectory of it.
#
# Visual Studio and Xcode are multi-config generators: left alone they write
# to build/cmake/Release/, which the Python binding does not search and which
# separates the decoder from the KenLM library it loads by a relative path. So
# the documented build would succeed and the import would then fail to find
# anything — on Windows, which is the platform this project exists to unblock.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER ${config} CONFIG_UPPER)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
endforeach()

# A language model is optional. Without it the decoder still does beam search,
# hotwords and the allowlist — which is the configuration the app ships — and
# asking for one raises rather than quietly decoding without it.
option(CTCBD_WITH_KENLM "Build with KenLM language model fusion" ON)

add_library(ctc_beam_decoder SHARED
  src/api.cpp
  src/beam_search.cpp
  src/hotwords.cpp
  src/language_model.cpp
)

if(CTCBD_WITH_KENLM)
  include(cmake/kenlm.cmake)
  target_compile_definitions(ctc_beam_decoder PRIVATE CTCBD_WITH_KENLM=1)
  target_link_libraries(ctc_beam_decoder PRIVATE kenlm)
  install(TARGETS kenlm LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
endif()

target_include_directories(ctc_beam_decoder
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Scores have to accumulate exactly as the reference implementation's do, so
# no reassociation, no contraction of a multiply-add, and no flushing of
# denormals. -ffast-math here would not be an optimisation, it would be a
# change of output.
if(MSVC)
  target_compile_options(ctc_beam_decoder PRIVATE /fp:precise /W4)
else()
  target_compile_options(ctc_beam_decoder PRIVATE
    -ffp-contract=off -fno-fast-math -Wall -Wextra)
endif()

# Windows gets no "lib" prefix, everywhere else does.
#
# Not cosmetic: the Python binding looks for `ctc_beam_decoder.dll` on
# Windows, following the same MSVC convention parakeet.cpp ships under, and an
# unconditional prefix would have produced `libctc_beam_decoder.dll` — built
# correctly, never found. On the one platform this project exists to unblock.
set_target_properties(ctc_beam_decoder PROPERTIES OUTPUT_NAME ctc_beam_decoder)
if(WIN32)
  set_target_properties(ctc_beam_decoder PROPERTIES PREFIX "")
else()
  set_target_properties(ctc_beam_decoder PROPERTIES PREFIX "lib")
endif()

# Carry our own C++ runtime on Linux.
#
# Without this, the .so records a dependency on whatever libstdc++ the build
# machine had. The v0.1.0 Linux release was built on ubuntu-latest — Ubuntu
# 24.04, GCC 13/14 — and therefore demands GLIBCXX_3.4.32, which Ubuntu 22.04
# LTS does not have. It failed to load there with:
#
#   libstdc++.so.6: version `GLIBCXX_3.4.32' not found
#     (required by libctc_beam_decoder.so)
#
# 22.04 is supported until 2027, and Debian 12 and RHEL 9 sit in the same
# bracket, so that is a large share of the Linux machines anyone would actually
# run this on. This project's own CI could never have caught it: it builds and
# tests on the same new toolchain, so it is self-consistent and broken for
# everybody else. It took building a downstream app on an older runner.
#
# Linking the C++ runtime statically makes the library indifferent to what the
# distro ships. It also makes the manylinux wheel tags honest — those tags are
# a promise about exactly this, and a wheel built the old way would have been
# claiming compatibility the binary did not have.
#
# Not needed on macOS (libc++ is part of the OS and stable) or Windows (the
# MSVC runtime is handled separately).
if(UNIX AND NOT APPLE)
  target_link_options(ctc_beam_decoder PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Look for KenLM next to ourselves once installed.
#
# CMake strips the build-tree rpath on install, so without this the installed
# library fails to load with "Library not loaded: @rpath/libkenlm.dylib" while
# every build-tree test passes — the build tree keeps an rpath pointing back
# at itself, so it cannot show the failure. scripts/check-install.sh installs
# to a clean prefix and loads it from there, which can.
#
# Windows needs nothing: DLLs resolve from the loading module's directory.
if(APPLE)
  set_target_properties(ctc_beam_decoder PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(ctc_beam_decoder PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS ctc_beam_decoder LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/ctc_beam_decoder.h DESTINATION include)

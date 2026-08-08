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

# Tried and rejected: -static-libstdc++ on Linux.
#
# The v0.1.0 Linux release would not load on Ubuntu 22.04 —
# "libstdc++.so.6: version `GLIBCXX_3.4.32' not found" — because it was built on
# ubuntu-latest, now 24.04. Linking the C++ runtime statically looked like the
# obvious belt-and-braces fix alongside pinning the build to 22.04.
#
# It segfaults. This library is loaded into processes that already contain other
# C++ extensions — pyctcdecode's kenlm module in this project's own parity job,
# and libparakeet (C++ with ggml) in classroom-captions — and a statically
# linked C++ runtime alongside the system one means two copies of that runtime's
# global state in one process. The parity job died with exit 139 and a core
# dump; the same thing would have happened in the shipped Linux app, where the
# decoder and libparakeet sit in the same interpreter.
#
# Building on the oldest supported distro is sufficient on its own and has no
# such hazard. See .github/workflows/release.yml, where the Linux job is pinned
# to ubuntu-22.04. Do not reintroduce this without a way to test the
# loaded-alongside-another-C++-extension case.

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

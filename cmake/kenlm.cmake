# Build KenLM as its own shared library.
#
# Shared, not static, and that is a licensing decision rather than a technical
# one. KenLM is LGPL; keeping it a separately loaded dynamic module is what
# makes it replaceable by whoever receives the binary, which is the obligation
# LGPL actually imposes. Statically linking it into an Apache-2.0 library
# would work fine and create a compliance chore instead.
#
# KenLM's own CMakeLists builds tools that want Boost. We compile only the
# sources needed to load a model and score against it, the same set KenLM's
# Python bindings compile.

set(KENLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kenlm")

if(NOT EXISTS "${KENLM_DIR}/lm/model.cc")
  message(FATAL_ERROR
    "KenLM sources not found in ${KENLM_DIR}. Run scripts/fetch-kenlm.sh, "
    "or configure with -DCTCBD_WITH_KENLM=OFF to build without a language model.")
endif()

file(GLOB KENLM_SOURCES
  "${KENLM_DIR}/util/*.cc"
  "${KENLM_DIR}/util/double-conversion/*.cc"
  "${KENLM_DIR}/lm/*.cc"
)
# Command-line tools and unit tests, which are not part of a library.
list(FILTER KENLM_SOURCES EXCLUDE REGEX "(main|test)\\.cc$")

add_library(kenlm SHARED ${KENLM_SOURCES})

# SYSTEM, so KenLM's warnings do not bury ours in the files that
# include its headers.
target_include_directories(kenlm SYSTEM PUBLIC "${KENLM_DIR}")

# KENLM_MAX_ORDER fixes the size of the state that gets carried between words,
# so it must match whatever built the model files being loaded. 6 is KenLM's
# own default and what its Python bindings use.
target_compile_definitions(kenlm PUBLIC KENLM_MAX_ORDER=6)
target_compile_definitions(kenlm PRIVATE NDEBUG)

set_target_properties(kenlm PROPERTIES
  CXX_STANDARD 11
  CXX_VISIBILITY_PRESET default   # it is meant to be linked against
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Third-party code, built with its own warnings suppressed: they are not
# actionable here and would bury ours.
if(MSVC)
  target_compile_options(kenlm PRIVATE /w /wd4996)
  target_compile_definitions(kenlm PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(kenlm PRIVATE -w)
endif()

if(NOT WIN32)
  target_compile_definitions(kenlm PRIVATE HAVE_CLOCKGETTIME)
endif()

# Language models ship gzipped (lm/3-gram.pruned.1e-7.arpa.gz), so this is
# required rather than a nicety — without it the model simply will not load.
#
# Vendored rather than found on the system: Windows runners have no zlib, and
# a build that silently drops gzip support there would fail only when someone
# enabled the language model, on one platform. See cmake/zlib.cmake.
include(${CMAKE_CURRENT_LIST_DIR}/zlib.cmake)
target_compile_definitions(kenlm PRIVATE HAVE_ZLIB)
target_link_libraries(kenlm PRIVATE ${CTCBD_ZLIB_TARGET})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(kenlm PRIVATE rt)
  # No -static-libstdc++ here either; see the note in the top-level
  # CMakeLists.txt for why it was tried and reverted.
endif()

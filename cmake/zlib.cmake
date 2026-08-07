# zlib, vendored and pinned, because KenLM needs it to read a gzipped model.
#
# The language model this is built for ships as `3-gram.pruned.1e-7.arpa.gz`.
# Without zlib, KenLM loads plain ARPA and binary models and refuses that one.
#
# Vendored rather than found on the system. macOS and Linux runners have zlib;
# Windows does not, and `find_package(ZLIB REQUIRED)` failed there — which
# would have left three choices, all worse than this one:
#
#   - make it optional, and have gzipped models load everywhere except
#     Windows. The language model ships disabled today, so that would not
#     break anything now; it would break whenever someone enables it, on one
#     platform, with no obvious cause.
#   - install zlib on the Windows runner, adding a system dependency that
#     anyone building locally would also need, plus a zlib1.dll to ship.
#   - ship an uncompressed or pre-converted model, which is a change to the
#     consumer's assets to work around a build problem.
#
# Building it here costs about a second and makes every platform identical.
# zlib's licence is permissive and attribution-only; see NOTICE.

include(FetchContent)

FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG        v1.3.1
  GIT_SHALLOW    TRUE
)

# Static, so nothing extra has to be shipped beside the decoder and KenLM.
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(zlib)

# zlib 1.3.1 names its static target zlibstatic on every platform; older
# releases and some configurations only produce `zlib`. Take whichever exists
# rather than assuming, so a version bump does not fail obscurely.
if(TARGET zlibstatic)
  set(CTCBD_ZLIB_TARGET zlibstatic)
elseif(TARGET zlib)
  set(CTCBD_ZLIB_TARGET zlib)
else()
  message(FATAL_ERROR "zlib was fetched but produced no library target")
endif()

# Its headers live in two places: the source for zlib.h, the build directory
# for the generated zconf.h.
target_include_directories(${CTCBD_ZLIB_TARGET} INTERFACE
  ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

set_property(TARGET ${CTCBD_ZLIB_TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

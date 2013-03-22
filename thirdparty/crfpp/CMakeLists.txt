include (ConfigureChecks)

# -----------------------------------------------
# configure section
# -----------------------------------------------

check_include_file_concat ("ctype.h"        HAVE_CTYPE_H)
check_include_file_concat ("dlfcn.h"        HAVE_DLFCN_H)
check_include_file_concat ("fcntl.h"        HAVE_FCNTL_H)
check_include_file_concat ("inttypes.h"     HAVE_INTTYPES_H)
check_include_file_concat ("math.h"         HAVE_MATH_H)
check_include_file_concat ("memory.h"       HAVE_MEMORY_H)
check_include_file_concat ("pthread.h"      HAVE_PTHREAD_H)
check_include_file_concat ("stdint.h"       HAVE_STDINT_H)
check_include_file_concat ("stdlib.h"       HAVE_STDLIB_H)
check_include_file_concat ("strings.h"      HAVE_STRINGS_H)
check_include_file_concat ("string.h"       HAVE_STRING_H)
check_include_file_concat ("sys/mman.h"     HAVE_SYS_MMAN_H)
check_include_file_concat ("sys/stat.h"     HAVE_SYS_STAT_H)
check_include_file_concat ("sys/times.h"    HAVE_SYS_TIMES_H)
check_include_file_concat ("sys/types.h"    HAVE_SYS_TYPES_H)
check_include_file_concat ("unistd.h"       HAVE_UNISTD_H)

check_function_exists ("getpagesize"    HAVE_GETPAGESIZE)
check_function_exists ("mmap"           HAVE_MMAP)

check_library_exists_concat ("m"        random          HAVE_LIBM)
check_library_exists_concat ("pthread"  pthread_create  HAVE_LIBPTHREAD)

set (PACKAGE    "CRF++")
set (VERSION    "0.50")

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

set (CMAKE_CXX_FLAGS "-DHAVE_CONFIG_H")
set (crfpp_SRC
    common.h
    config.h
    config-win32.h
    crfpp.h
    darts.h
    encoder.h
    feature_cache.h
    feature_index.h
    freelist.h
    lbfgs.h
    mmap.h
    node.h
    param.h
    path.h
    scoped_ptr.h
    stream_wrapper.h
    tagger.h
    thread.h
    timer.h
    encoder.cpp
    feature_cache.cpp
    feature.cpp
    feature_index.cpp
    lbfgs.cpp
    libcrfpp.cpp
    node.cpp
    param.cpp
    path.cpp
    tagger.cpp)

add_library (crfpp ${crfpp_SRC})

# -----------------------------------------------
# add executables
# -----------------------------------------------

set (crf_learn_SRC  crf_learn.cpp)
add_executable (crf_learn   ${crf_learn_SRC})
target_link_libraries (crf_learn crfpp pthread)
set_target_properties (crf_learn
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${TOOLS_DIR}/train)

set (crf_test_SRC   crf_test.cpp)
add_executable (crf_test    ${crf_test_SRC})
target_link_libraries (crf_test crfpp pthread)
set_target_properties (crf_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${TOOLS_DIR}/train)



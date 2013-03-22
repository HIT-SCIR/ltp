include (ConfigureChecks)

# -----------------------------------------------
# check section
# -----------------------------------------------

check_function_exists (getopt_long HAVE_GETOPT_LONG)
check_function_exists (getpagesize HAVE_GETPAGESIZE)
check_function_exists (gettimeofday HAVE_GETTIMEOFDAY)

check_include_file_concat ("ieeefp.h" HAVE_IEEEFP_H)
check_include_file_concat ("inttypes.h" HAVE_INTTYPES_H)
check_include_file_concat ("memory.h" HAVE_MEMORY_H)
check_include_file_concat ("stdint.h" HAVE_STDINT_H)
check_include_file_concat ("stdlib.h" HAVE_STDLIB_H)
check_include_file_concat ("strings.h" HAVE_STRINGS_H)
check_include_file_concat ("string.h" HAVE_STRING_H)
check_include_file_concat ("sys/stat.h" HAVE_SYS_STAT_H)
check_include_file_concat ("sys/types.h" HAVE_SYS_TYPES_H)
check_include_file_concat ("unistd.h" HAVE_UNISTD_H)

check_library_exists_concat ("gnugetopt" getopt HAVE_LIBGNUGETOPT)
check_library_exists_concat ("z" deflateEnd HAVE_LIBZ)
if (NOT WINDOWS)
    check_library_exists_concat ("m" random HAVE_LIBM)
endif (NOT WINDOWS)

include (CMakeDetermineFortranCompiler)

if (CMAKE_Fortran_COMPILER)
    set (HAVE_FORTRAN   1)
    enable_language (Fortran)
endif (CMAKE_Fortran_COMPILER)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
        ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

# -----------------------------------------------
# compiling section
# -----------------------------------------------
include_directories (. ${THIRDPARTY_DIR}/boost/include)

set (OPT_FILES "")

if (CMAKE_Fortran_COMPILER)
    set (OPT_FILES      lbfgs.f lbfgstrainer.cpp)
else (CMAKE_Fortran_COMPILER)
    message (STATUS
        "[Warning] Fortran compiler not available, "
        "LBFGS module will not be built.")
endif (CMAKE_Fortran_COMPILER)

set (maxent_lib_SRC
    display.cpp
    modelfile.cpp
    trainer.cpp
    gistrainer.cpp
    maxentmodel.cpp
    mmapfile.c
    lbfgs_wrapper.c
    ${OPT_FILES})

add_library (maxent_lib ${maxent_lib_SRC})
set_target_properties (maxent_lib PROPERTIES OUTPUT_NAME maxent)

set (GETOPT_FILES "")

if (WIN32 OR MINGW)
    set (GETOPT_FILES getopt.c getopt1.c)
endif ()

set (maxent_exe_SRC
    maxent.cpp
    display.cpp
    maxent_cmdline.c
    f77_dummy_main.cpp
    ${GETOPT_FILES})

add_executable (maxent_exe ${maxent_exe_SRC})
set_target_properties (maxent_exe PROPERTIES OUTPUT_NAME maxent)
target_link_libraries (maxent_exe maxent_lib z m)

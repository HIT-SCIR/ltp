include_directories (${SOURCE_DIR}/ ${THIRDPARTY_DIR}/boost/include/)

set (postagger_VERSION "0.0.3")
set (postagger_SRC
    settings.h
    decoder.h
    decoder.cpp
    extractor.h
    extractor.cpp
    postagger.h
    postagger.cpp)

# -----------------------------------------------
# STATIC LIBRARY
# -----------------------------------------------
add_library (postagger_static_lib
    postag_dll.h
    postag_dll.cpp
    ${postagger_SRC})
set_target_properties (postagger_static_lib PROPERTIES
    OUTPUT_NAME postagger
    PREFIX "lib")

# -----------------------------------------------
# SHARED LIBRARY
# -----------------------------------------------
add_library (postagger_shared_lib SHARED
    postag_dll.h
    postag_dll.cpp
    ${postagger_SRC})
set_target_properties (postagger_shared_lib PROPERTIES
    VERSION     ${postagger_VERSION}
    OUTPUT_NAME postagger)

# ------------------------------------------------
# EXECUTABLE
# ------------------------------------------------
add_executable (otpos otpos.cpp
    postagger_frontend.h
    postagger_frontend.cpp
    io.h
    io.cpp
    ${postagger_SRC})
target_link_libraries (otpos boost_program_options_static_lib)
set_target_properties (otpos PROPERTIES
    OUTPUT_NAME otpos
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    postag_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/postag_dll.h)

include_directories (  ${SOURCE_DIR}/ )

set (postagger_VERSION "0.0.2")

set (postagger_SRC
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    postagger.cpp)

# -----------------------------------------------
# STATIC LIBRARY
# -----------------------------------------------
add_library (postagger_static_lib postag_dll.cpp ${postagger_SRC})

set_target_properties (postagger_static_lib PROPERTIES
    OUTPUT_NAME postagger)

# -----------------------------------------------
# SHARED LIBRARY
# -----------------------------------------------
add_library (postagger_shared_lib SHARED postag_dll.cpp ${postagger_SRC})

set_target_properties (postagger_shared_lib PROPERTIES
    VERSION     ${postagger_VERSION}
    OUTPUT_NAME postagger)


add_library (postagger postag_dll.cpp ${postagger_SRC})

add_executable (otpos otpos.cpp postagger_frontend.cpp io.cpp ${postagger_SRC})
# redirect the output binary to tools/train
target_link_libraries (otpos boost_program_options_static_lib)
set_target_properties (otpos PROPERTIES 
    OUTPUT_NAME otpos
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    postag_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/postag_dll.h)

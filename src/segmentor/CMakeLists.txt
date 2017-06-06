include_directories ( ${SOURCE_DIR} ${THIRDPARTY_DIR}/boost/include )

set (segment_VERSION "0.2.0")

set (segment_SRC
    settings.h
    special_tokens.h
    options.h
    decoder.h
    decoder.cpp
    preprocessor.h
    preprocessor.cpp
    model.h
    model.cpp
    extractor.h
    extractor.cpp
    segmentor.h
    segmentor.cpp
    )

# -----------------------------------------------
# STATIC LIBRARY
# -----------------------------------------------
add_library (segmentor_static_lib
    segment_dll.h
    segment_dll.cpp
    ${segment_SRC})
target_link_libraries (segmentor_static_lib boost_regex_static_lib)
set_target_properties (segmentor_static_lib
    PROPERTIES
    OUTPUT_NAME segmentor
    PREFIX "lib")

# -----------------------------------------------
# SHARED LIBRARY
# -----------------------------------------------
add_library (segmentor_shared_lib SHARED
    segment_dll.h
    segment_dll.cpp
    ${segment_SRC})
#target_link_libraries (segmentor_shared_lib boost_regex_shared_lib)
target_link_libraries (segmentor_shared_lib boost_regex_static_lib)
set_target_properties (segmentor_shared_lib PROPERTIES
    VERSION     ${segment_VERSION}
    OUTPUT_NAME segmentor)

link_directories ( ${LIBRARY_OUTPUT_PATH} )

# -----------------------------------------------
# TOOLKIT #1
# -----------------------------------------------
add_executable (otcws otcws.cpp
    segmentor_frontend.h
    segmentor_frontend.cpp
    partial_segmentation.h
    partial_segmentation.cpp
    customized_segmentor_frontend.h
    customized_segmentor_frontend.cpp
    io.h
    io.cpp
    ${segment_SRC})
target_link_libraries (otcws boost_regex_static_lib boost_program_options_static_lib)
set_target_properties (otcws PROPERTIES  OUTPUT_NAME otcws
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

# -----------------------------------------------
# CONFIG header FILE
# -----------------------------------------------
configure_file (
    segment_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/segment_dll.h)

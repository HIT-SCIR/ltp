include_directories ( ${SOURCE_DIR} ${THIRDPARTY_DIR}/boost/include )

set (segment_VERSION "0.0.1")

set (segment_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    segmentor.cpp)

# -----------------------------------------------
# STATIC LIBRARY
# -----------------------------------------------
add_library (segmentor_static_lib segment_dll.cpp ${segment_SRC})

target_link_libraries (segmentor_static_lib boost_regex_static_lib)

set_target_properties (segmentor_static_lib
    PROPERTIES
    OUTPUT_NAME segmentor)

# -----------------------------------------------
# SHARED LIBRARY
# -----------------------------------------------
add_library (segmentor_shared_lib SHARED segment_dll.cpp ${segment_SRC})

target_link_libraries (segmentor_shared_lib boost_regex_shared_lib)

set_target_properties (segmentor_shared_lib PROPERTIES
    VERSION     ${segment_VERSION}
    OUTPUT_NAME segmentor)

# -----------------------------------------------
# TOOLKIT
# -----------------------------------------------
add_executable (otcws otcws.cpp ${segment_SRC})
message (STATUS ${LIBRARY_OUTPUT_PATH} )
link_directories ( ${LIBRARY_OUTPUT_PATH} )

target_link_libraries (otcws boost_regex_static_lib)

# redirect the output binary to tools/train
set_target_properties (otcws
    PROPERTIES
    OUTPUT_NAME otcws
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    segment_dll.h
    ${INCLUDE_OUTPUT_PATH}/segment_dll.h)

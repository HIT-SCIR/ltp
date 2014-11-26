include_directories ( ${SOURCE_DIR} ${THIRDPARTY_DIR}/boost/include )

set (segment_VERSION "0.1.0")

set (segment_SRC
    decoder.cpp
    featurespace.cpp
    rulebase.cpp
    model.cpp
    extractor.cpp
    segmentor.cpp
    customized_segmentor.cpp)

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

link_directories ( ${LIBRARY_OUTPUT_PATH} )

# -----------------------------------------------
# TOOLKIT #1
# -----------------------------------------------
add_executable (otcws otcws.cpp ${segment_SRC})

target_link_libraries (otcws boost_regex_static_lib)

set_target_properties (otcws
    PROPERTIES
    OUTPUT_NAME otcws
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

# -----------------------------------------------
# TOOLKIT #2
# -----------------------------------------------
add_executable (otcws-customized otcws_customized.cpp
    ${segment_SRC} customized_segmentor.cpp)

target_link_libraries (otcws-customized boost_regex_static_lib)

set_target_properties (otcws-customized
    PROPERTIES
    OUTPUT_NAME otcws-customized
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    segment_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/segment_dll.h)

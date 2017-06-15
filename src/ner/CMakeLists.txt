include_directories (${SOURCE_DIR}/ ${THIRDPARTY_DIR}/boost/include/)

set (ner_VERSION "0.0.2")
set (ner_SRC
    settings.h
    decoder.h
    decoder.cpp
    extractor.h
    extractor.cpp
    ner.h
    ner.cpp)

# -----------------------------------------------
# STATIC LIBRARY
# -----------------------------------------------
add_library (ner_static_lib STATIC
    ner_dll.h
    ner_dll.cpp
    ${ner_SRC})
set_target_properties (ner_static_lib PROPERTIES OUTPUT_NAME ner
                       PREFIX "lib")

# -----------------------------------------------
# SHARED LIBRARY
# -----------------------------------------------
add_library (ner_shared_lib SHARED
    ner_dll.h
    ner_dll.cpp
    ${ner_SRC})
set_target_properties (ner_shared_lib PROPERTIES
    VERSION     ${ner_VERSION}
    OUTPUT_NAME ner)

# ------------------------------------------------
# EXECUTABLE
# ------------------------------------------------
add_executable (otner otner.cpp
    ner_frontend.h    
    ner_frontend.cpp
    io.h
    io.cpp
    ${ner_SRC})
target_link_libraries (otner boost_program_options_static_lib)
set_target_properties (otner PROPERTIES
    OUTPUT_NAME otner
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    ner_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/ner_dll.h)

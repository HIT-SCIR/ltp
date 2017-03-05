INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_program_options_VERSION "1.61")

set (boost_program_options_SRC
    src/cmdline.cpp
    src/config_file.cpp
    src/convert.cpp
    src/options_description.cpp
    src/parsers.cpp
    src/positional_options.cpp
    src/split.cpp
    src/utf8_codecvt_facet.cpp
    src/value_semantic.cpp
    src/variables_map.cpp
    src/winmain.cpp)

add_library(boost_program_options_static_lib STATIC ${boost_program_options_SRC})
add_library(boost_program_options_shared_lib SHARED ${boost_program_options_SRC})
set_target_properties (boost_program_options_static_lib PROPERTIES
    OUTPUT_NAME boost_program_options)
set_target_properties (boost_program_options_shared_lib PROPERTIES
    VERSION     ${boost_program_options_VERSION}
    OUTPUT_NAME boost_program_options)

INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_program_options_VERSION "1.61")

FILE(GLOB boost_program_options_SRC src/*.cpp)

add_library(boost_program_options_static_lib STATIC ${boost_program_options_SRC})
add_library(boost_program_options_shared_lib SHARED ${boost_program_options_SRC})
set_target_properties (boost_program_options_static_lib PROPERTIES
    OUTPUT_NAME boost_program_options
    PREFIX "lib")
set_target_properties (boost_program_options_shared_lib PROPERTIES
    VERSION     ${boost_program_options_VERSION}
    OUTPUT_NAME boost_program_options)

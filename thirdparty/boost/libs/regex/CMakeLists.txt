INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_regex_VERSION "1.61")

FILE(GLOB boost_regex_SRC src/*.cpp src/*.hpp)

add_library(boost_regex_static_lib STATIC ${boost_regex_SRC})

set_target_properties (boost_regex_static_lib   PROPERTIES
    OUTPUT_NAME boost_regex
    PREFIX "lib")

add_library(boost_regex_shared_lib SHARED ${boost_regex_SRC})

set_target_properties (boost_regex_shared_lib   PROPERTIES
    VERSION     ${boost_regex_VERSION}
    OUTPUT_NAME boost_regex)

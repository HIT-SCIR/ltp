INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_serialization_VERSION "1.61")

FILE(GLOB boost_serialization_SRC src/*.cpp src/*.ipp)

add_library (boost_serialization_static_lib STATIC ${boost_serialization_SRC})

set_target_properties (boost_serialization_static_lib  PROPERTIES
    OUTPUT_NAME boost_serialization
    PREFIX "lib")

add_library (boost_serialization_shared_lib SHARED ${boost_serialization_SRC})

set_target_properties (boost_serialization_shared_lib  PROPERTIES
    VERSION     ${boost_serialization_VERSION}
    OUTPUT_NAME boost_serialization)

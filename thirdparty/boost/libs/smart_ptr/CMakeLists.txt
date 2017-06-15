INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_smart_ptr_VERSION "1.61")

FILE(GLOB boost_smart_ptr_SRC src/*.cpp)

add_library(boost_smart_ptr_static_lib STATIC ${boost_smart_ptr_SRC})
add_library(boost_smart_ptr_shared_lib SHARED ${boost_smart_ptr_SRC})
set_target_properties (boost_smart_ptr_static_lib PROPERTIES
    OUTPUT_NAME boost_smart_ptr
    PREFIX "lib")
set_target_properties (boost_smart_ptr_shared_lib PROPERTIES
    VERSION     ${boost_smart_ptr_VERSION}
    OUTPUT_NAME boost_smart_ptr)

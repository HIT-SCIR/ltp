set (util_VERSION "0.0.1")

set (util_source
    MyLib.cpp
    MyLib.h
    STL.h)

add_library (util_static_lib STATIC ${util_source})
set_target_properties (util_static_lib PROPERTIES OUTPUT_NAME util)

add_library (util_shared_lib SHARED ${util_source})
set_target_properties (util_shared_lib PROPERTIES
    VERSION     ${util_VERSION}
    OUTPUT_NAME util)



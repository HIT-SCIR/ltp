include_directories(${Boost_INCLUDE_DIR})

set(base_src
        base/config.cpp
        base/debug.cpp
        )

add_library(base_static_lib STATIC ${base_src})
target_link_libraries(base_static_lib ${Boost_LIBRARIES})
add_library(base_shared_lib SHARED ${base_src})
target_link_libraries(base_shared_lib ${Boost_LIBRARIES})

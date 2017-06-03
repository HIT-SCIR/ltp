# look for Boost
if(DEFINED ENV{BOOST_ROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(base_src
        base/config.cpp
        base/debug.cpp
        )

add_library(base_static_lib STATIC ${base_src})
target_link_libraries(base_static_lib ${Boost_LIBRARIES})
add_library(base_shared_lib SHARED ${base_src})
target_link_libraries(base_shared_lib ${Boost_LIBRARIES})
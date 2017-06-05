include_directories (.
        ${THIRDPARTY_DIR}/boost/include
        ${THIRDPARTY_DIR}/eigen
        ${THIRDPARTY_DIR}/dynet
        )

#link_directories(${THIRDPARTY_DIR}/dynet/build/dynet) # -L
set(Boost_LIBRARIES boost_program_options_static_lib boost_serialization_static_lib)

set(LIBS ${LIBS} ${Boost_LIBRARIES})

add_subdirectory(include)
include_directories(include)

set(LIBS ${LIBS} base_static_lib)

set (srl_VERSION "0.0.1")


include_directories(common)
add_subdirectory(common)

set(LIBS ${LIBS} common_static_lib)

add_subdirectory(Pi)
#set(LIBS ${LIBS} srl_pi_static_lib)

add_subdirectory(Srl)
set(LIBS ${LIBS} srl_srl_static_lib)

add_subdirectory(tool)

set (srl_SRC
        DepSRL.cpp
        DepSRL.h
        SRL_DLL.cpp
        SRL_DLL.h)


add_library (srl_static_lib STATIC ${srl_SRC})
target_link_libraries (srl_static_lib dynet ${LIBS})

set_target_properties (srl_static_lib    PROPERTIES
        OUTPUT_NAME     srl
        PREFIX "lib")

add_library(srl_shared_lib SHARED ${srl_SRC})
target_link_libraries (srl_shared_lib dynet ${LIBS})
set_target_properties (srl_shared_lib    PROPERTIES
        VERSION         ${srl_VERSION}
        OUTPUT_NAME     srl)

#add_library (ppsrl_shared_lib SHARED ${srl_SRC})
#
#set_target_properties (ppsrl_shared_lib   PROPERTIES
#    VERSION         ${srl_VERSION}
#    OUTPUT_NAME     srl)

target_link_libraries (srl_static_lib dynet ${LIBS})
#target_link_libraries (ppsrl_shared_lib maxent_shared_lib)

configure_file (
        SRL_DLL.h
        ${INCLUDE_OUTPUT_PATH}/ltp/srl_dll.h)

link_directories (${LIBRARY_OUTPUT_PATH})

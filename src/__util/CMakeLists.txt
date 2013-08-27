set (util_VERSION "0.0.1")

set (util_source
    conversion_utf.h
    decode_gbk.h
    encode.cpp
    EncodeUtil.cpp
    EncodeUtil.h
    gbk_u16.h
    IniReader.cpp
    IniReader.h
    md5.cpp
    md5.h
    MyLib.cpp
    MyLib.h
    SBC2DBC.cpp
    SBC2DBC.h
    STL.h
    TextProcess.cpp
    TextProcess.h
    Timer.h
    encode.cpp 
    encode.hpp)

add_library (util_static_lib STATIC ${util_source})
set_target_properties (util_static_lib PROPERTIES OUTPUT_NAME util)

add_library (util_shared_lib SHARED ${util_source})
set_target_properties (util_shared_lib PROPERTIES
    VERSION     ${util_VERSION}
    OUTPUT_NAME util)



SET(util_source
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

ADD_LIBRARY(util ${util_source})

set (tinyxml_SRC
    tinystr.h
    tinyxml.h
    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp)

add_library (tinyxml ${tinyxml_SRC})

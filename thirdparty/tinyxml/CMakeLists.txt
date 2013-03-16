SET(tinyxml_source
    tinystr.h
    tinyxml.h
    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp)

ADD_LIBRARY(tinyxml ${tinyxml_source})

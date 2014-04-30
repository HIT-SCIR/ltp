INCLUDE_DIRECTORIES(${SOURCE_DIR}
    ${THIRDPARTY_DIR}/tinyxml)

SET(xml4nlp_source
    Xml4nlp.h
    Xml4nlp.cpp)

ADD_LIBRARY(xml4nlp ${xml4nlp_source})

TARGET_LINK_LIBRARIES(xml4nlp tinyxml)

include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/tinyxml)

SET(ltp_SRC Ltp.cpp LTPResource.cpp)

ADD_LIBRARY(ltp ${ltp_SRC})

include_directories (./
    ${SOURCE_DIR}
    ${segmentor_DIR}
    ${xml4nlp_DIR}
    ${util_DIR}
    ${utils_DIR}
    ${splitsnt_DIR}
    ${postagger_DIR}
    ${parser_DIR}
    ${ner_DIR}
    ${srl_DIR}
    ${THIRDPARTY_DIR}/tinyxml)

SET(ltp_SRC
    Ltp.cpp
    Ltp.h
    LTPOption.h
    LTPResource.cpp
    LTPResource.h)

ADD_LIBRARY(ltp ${ltp_SRC})

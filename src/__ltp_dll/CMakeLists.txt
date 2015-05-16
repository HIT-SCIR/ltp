include_directories (./
    ${xml4nlp_DIR}
    ${util_DIR}
    ${utils_DIR}
    ${splitsnt_DIR}
    ${segmentor_DIR}
    ${postagger_DIR}
    ${parser_DIR}
    ${ner_DIR}
    ${srl_DIR}
    ${THIRDPARTY_DIR}/tinyxml)

SET(ltp_SRC Ltp.cpp LTPResource.cpp)

ADD_LIBRARY(ltp ${ltp_SRC})

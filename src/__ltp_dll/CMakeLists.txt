include_directories (./
    ${xml4nlp_DIR}
    ${util_DIR}
    ${crfwordseg_DIR}
    ${svmtagger_DIR}
    ${gparser_DIR}
    ${ner_DIR}
    ${srl_DIR}
    ${splitsnt_DIR}
    ${THIRDPARTY_DIR}/tinyxml)

SET(ltp_SRC
    Ltp.cpp
    Ltp.h
    LTPOption.h
    LTPResource.cpp
    LTPResource.h)

ADD_LIBRARY(ltp ${ltp_SRC})

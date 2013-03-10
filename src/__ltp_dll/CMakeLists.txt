INCLUDE_DIRECTORIES(./
    ${xml4nlp_dir}
    ${util_dir}
    ${crfwordseg_dir}
    ${svmtagger_dir}
    ${gparser_dir}
    ${ner_dir}
    ${srl_dir}
    ${splitsnt_dir}
    ${THIRDPARTY}/tinyxml)

SET(ltp_source
    Ltp.cpp
    Ltp.h
    LTPOption.h
    LTPResource.cpp
    LTPResource.h)

ADD_LIBRARY(ltp ${ltp_source})

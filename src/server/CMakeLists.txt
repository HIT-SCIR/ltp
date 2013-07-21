include_directories (./
    ${SOURCE_DIR}/__util
    ${SOURCE_DIR}/__xml4nlp
    ${SOURCE_DIR}/_split_sentence
    ${SOURCE_DIR}/_crfwordsegment
    ${SOURCE_DIR}/_svmtagger
    ${SOURCE_DIR}/_ner
    ${SOURCE_DIR}/_gparser
    ${SOURCE_DIR}/_srl
    ${SOURCE_DIR}/__ltp_dll
    ${THIRDPARTY_DIR}/crfpp
    ${THIRDPARTY_DIR}/maxent
    ${THIRDPARTY_DIR}/tinyxml)

set (ltp_server_SRC           ltp_server.cpp mongoose.c mongoose.h)

link_directories ( ${LIBRARY_OUTPUT_PATH} )

add_executable (ltp_server     ${ltp_server_SRC})

target_link_libraries (ltp_server
    maxent
    util
    ltp
    splitsnt
    segmentor
    postagger
    parser
    ner
    srl
    xml4nlp
    dl
    pthread
    boost_regex)


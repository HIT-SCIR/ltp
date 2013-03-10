SET(ltp_test_source ltp_test.cpp)
ADD_EXECUTABLE(ltp_test ${ltp_test_source})
TARGET_LINK_LIBRARIES(ltp_test
    maxent
    crfpp
    util
    ltp
    splitsnt
    crfwordseg
    postagger
    gparser
    ner
    srl
    xml4nlp
    dl
    pthread
    boost_regex)

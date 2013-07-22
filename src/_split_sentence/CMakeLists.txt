include_directories (./
    ${utils_DIR})

SET(splitsnt_source
    define.h
    Reader.cpp
    Reader.h
    SentenceIterator.h
    SplitSentence.cpp
    SplitSentence.h)

ADD_LIBRARY(splitsnt ${splitsnt_source})

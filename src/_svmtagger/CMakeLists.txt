INCLUDE_DIRECTORIES(./ ${util_dir} ${THIRDPARTY}/boost/include)

SET(postagger_source
    common.cpp
    common.h
    dict.cpp
    dict.h
    er.cpp
    er.h
    hash.cpp
    hash.h
    list.cpp
    list.h
    marks.h
    stack.cpp
    stack.h
    svmtagger_dll.cpp
    svmtagger_dll.h
    svmtagger_dll_x.cpp
    swindow.cpp
    swindow.h
    tagger.cpp
    tagger.h
    test_svmtagger_dll.cpp
    weight.cpp
    weight.h)

ADD_LIBRARY(postagger ${postagger_source})

include_directories (./ 
    ${util_DIR}
    ${THIRDPARTY_DIR}/boost/include)

set (postagger_SRC
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

add_library (postagger ${postagger_SRC})

configure_file (
    svmtagger_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/svmtagger_dll.h)

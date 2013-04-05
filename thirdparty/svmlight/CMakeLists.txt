set (svm_learn_SRC
    kernel.h
    svm_common.c
    svm_common.h
    svm_hideo.c
    svm_learn.c
    svm_learn.h
    svm_learn_main.c)

add_executable (svm_learn
    ${svm_learn_SRC})

set_target_properties (svm_learn 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY  ${TOOLS_DIR}/train/)

target_link_libraries (svm_learn m)

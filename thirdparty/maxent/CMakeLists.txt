
set (maxent_SRC
    opmath.h
    maxent.h
    quasi_newton.h
    maxent.cpp
    sgd.cpp
    owlqn.cpp
    lbfgs.cpp)

add_library (maxent ${maxent_SRC})

#-----------------------------------
# add executables
#-----------------------------------

#set (me_train_SRC train.cpp)
#add_executable (me_train ${me_train_SRC})
#target_link_libraries (me_train maxent)
#set_target_properties (me_train
#    PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train)
#
#set (me_predict_SRC predict.cpp)
#add_executable (me_predict ${me_predict_SRC})
#target_link_libraries (me_predict maxent)
#set_target_properties (me_predict
#    PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train)


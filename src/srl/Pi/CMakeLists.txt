SET(SrlPiSRC
        config/SrlPiConfig.h
        model/SrlPiModel.h
        )

add_executable(srl_pi_train train.cpp process/TrainSrlPi.cpp ${SrlPiSRC})
target_link_libraries(srl_pi_train dynet ${LIBS})

set_target_properties (srl_pi_train PROPERTIES
        OUTPUT_NAME srl_pi_train
        RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

add_executable(Pipred pred.cpp process/PredSrlPi.cpp ${SrlPiSRC})
target_link_libraries(Pipred dynet ${LIBS})

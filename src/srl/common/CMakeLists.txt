set(CommonSRC
        model/AffineTransformModelBuilder.cpp
        model/CNN1dLayerBuilder.cpp
        process/LabelModelSGDSeqTrainer.h
        model/PiSrlModel.h
        structure/SrlPiSample.cpp
        structure/WordEmbBuilder.h
        model/RNNModelBuilder.h model/BiRNNModelBuilder.h)


add_library(common_static_lib STATIC ${CommonSRC})
target_link_libraries(common_static_lib ${LIBS})
#add_library(common_shared_lib SHARED ${CommonSRC})
#target_link_libraries(common_shared_lib ${LIBS})
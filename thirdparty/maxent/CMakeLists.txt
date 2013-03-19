INCLUDE_DIRECTORIES(. ${THIRDPARTY}/boost/include)
SET(maxent_source
    display.cpp
    display.hpp
    eventspace.hpp
    ext_algorithm.hpp
    finite.h
    gistrainer.cpp
    gistrainer.hpp
    hash_map.hpp
    itemmap.hpp
    lbfgs.h
    maxentmodel.cpp
    maxentmodel.hpp
    meevent.hpp
    mmapfile.c
    mmapfile.h
    modelfile.cpp
    modelfile.hpp
    rfevent.hpp
    trainer.cpp
    trainer.hpp)

INCLUDE_DIRECTORIES(./)
ADD_LIBRARY(maxent ${maxent_source})



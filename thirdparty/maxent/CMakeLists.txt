include_directories (. ${THIRDPARTY_DIR}/boost/include)

set (maxent_SRC
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

add_library (maxent ${maxent_SRC})



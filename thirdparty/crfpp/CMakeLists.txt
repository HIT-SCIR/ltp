SET(CMAKE_CXX_FLAGS "-DHAVE_CONFIG_H")
SET(crfpp_source
    common.h
    config.h
    config-win32.h
    crfpp.h
    darts.h
    encoder.h
    feature_cache.h
    feature_index.h
    freelist.h
    lbfgs.h
    mmap.h
    node.h
    param.h
    path.h
    scoped_ptr.h
    stream_wrapper.h
    tagger.h
    thread.h
    timer.h
    encoder.cpp
    feature_cache.cpp
    feature.cpp
    feature_index.cpp
    lbfgs.cpp
    libcrfpp.cpp
    node.cpp
    param.cpp
    path.cpp
    tagger.cpp)

ADD_LIBRARY(crfpp ${crfpp_source})

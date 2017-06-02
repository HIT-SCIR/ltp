# ########## dynet library ##########
# Sources:
set(dynet_library_SRCS
    aligned-mem-pool.cc
    cfsm-builder.cc
    dynet.cc
    deep-lstm.cc
    devices.cc
    dict.cc
    dim.cc
    exec.cc
    expr.cc
    fast-lstm.cc
    globals.cc
    grad-check.cc
    graph.cc
    gru.cc
    hsm-builder.cc
    init.cc
    lstm.cc
    mem.cc
    model.cc
    mp.cc
    nodes.cc
    nodes-common.cc
    nodes-contract.cc
    nodes-conv.cc
    nodes-conv2d.cc
    param-nodes.cc
    pretrain.cc
    rnn.cc
    rnn-state-machine.cc
    saxe-init.cc
    shadow-params.cc
    tensor.cc
    training.cc
    treelstm.cc
    weight-decay.cc
)

# Headers:
set(dynet_library_HDRS
    aligned-mem-pool.h
    cfsm-builder.h
    cudnn-ops.h
    c2w.h
    dynet.h
    cuda.h
    devices.h
    dict.h
    dim.h
    exec.h
    expr.h
    fast-lstm.h
    functors.h
    globals.h
    gpu-kernels.h
    gpu-ops.h
    graph.h
    gru.h
    hsm-builder.h
    init.h
    lstm.h
    mem.h
    model.h
    mp.h
    nodes.h
    nodes-contract.h
    nodes-conv.h
    op-helper.h
    param-nodes.h
    rnn-state-machine.h
    rnn.h
    saxe-init.h
    shadow-params.h
    simd-functors.h
    tensor.h
    timing.h
    training.h
    treelstm.h
    except.h
    nodes-macros.h
    weight-decay.h
    io-macros.h
)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tests/*.cc)
if (NOT MSVC)
  set(BUILD_SHARED_LIBS ON)
endif()

#foreach(test_src ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
#  get_filename_component(testName ${test_src} NAME_WE)

  #Add compile target
#  add_executable(${testName} ${test_src})

  #link to Boost libraries AND your targets and dependencies
#  target_link_libraries(${testName} dynet ${LIBS})

#  set_target_properties(${testName} PROPERTIES
#      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin)

  #Finally add it to test execution -
  #Notice the WORKING_DIRECTORY and COMMAND
#  add_test(NAME ${testName}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin/${testName} )
#endforeach(test_src)

# Build cpu library
add_library(dynet ${dynet_library_SRCS} ${dynet_library_HDRS})
target_link_libraries(dynet ${LIBS})
if(WITH_CUDA_BACKEND)
  # cuda flags
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA;")
  if(CUDNN_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-DHAVE_CUDNN")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      # gcc 4.9 or later versions raise SEGV due to the optimization problem.
      # Use -O1 instead for now.
      list(APPEND CUDA_NVCC_FLAGS "-O1")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-O2")
    endif()
  else()
    list(APPEND CUDA_NVCC_FLAGS "-O2")
  endif()
  if(MSVC)
    # If MSVC, we need the boost flag because nvcc doesn't properly parse part of the boost template definitions
    list(APPEND CUDA_NVCC_FLAGS "-DBOOST_NO_CXX11_ALLOCATOR")
    list(APPEND CUDA_NVCC_FLAGS_DEBUG "--compiler-options \"/MDd\"")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "--compiler-options \"/MD\"")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_add_library(gdynet ${dynet_library_SRCS} ${dynet_library_HDRS} cuda.cc cudnn-ops.cu gpu-ops.cu gpu-nodes.cu gpu-nodes-contract.cu gpu-nodes-conv.cu gpu-nodes-conv2d.cu gpu-param-nodes.cu gpu-tensor.cu gpu-training.cu gpu-model.cu)
  else()
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_add_library(gdynet ${dynet_library_SRCS} ${dynet_library_HDRS} cuda.cc cudnn-ops.cu gpu-ops.cu gpu-nodes.cu gpu-nodes-contract.cu gpu-nodes-conv.cu gpu-nodes-conv2d.cu gpu-param-nodes.cu gpu-tensor.cu gpu-training.cu gpu-model.cu OPTIONS --compiler-options "-fPIC")
  endif()
  set_target_properties(gdynet PROPERTIES
                        COMPILE_DEFINITIONS HAVE_CUDA)
  if(CUDNN_FOUND)
    target_compile_definitions(gdynet PRIVATE HAVE_CUDNN)
  endif()
  cuda_add_cublas_to_target(gdynet)
  target_link_libraries(gdynet ${LIBS})

endif(WITH_CUDA_BACKEND)

install(FILES ${dynet_library_HDRS} DESTINATION include/dynet)
install(TARGETS dynet DESTINATION lib)
if(WITH_CUDA_BACKEND)
  install(TARGETS gdynet DESTINATION lib)
endif(WITH_CUDA_BACKEND)

# target_compile_features(dynet PRIVATE cxx_range_for)


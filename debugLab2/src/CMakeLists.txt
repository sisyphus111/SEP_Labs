if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    # 启用 AddressSanitizer 并添加调试信息
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    # 也可以启用内存泄漏检测
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=leak -g")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -g")
endif()
add_executable(run "main.cpp" "core.cpp")
target_link_libraries(run PRIVATE llama_tokenizer CLI11::CLI11 ggml fmt)
target_include_directories(run PUBLIC .)

# Find GTest
find_package(GTest)

if(GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})

    # Test correctness executable
    add_executable(test_correctness test_correctness.cpp)
    target_link_libraries(test_correctness
        matrix_common
        matrix_cpu
        matrix_gpu
        ${GTEST_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # CPU tests
    add_executable(test_cpu test_cpu.cpp)
    target_link_libraries(test_cpu
        matrix_common
        matrix_cpu
        ${GTEST_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # GPU tests
    add_executable(test_gpu test_gpu.cpp)
    target_link_libraries(test_gpu
        matrix_common
        matrix_gpu
        ${GTEST_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # Add tests
    add_test(NAME TestCorrectness COMMAND test_correctness)
    add_test(NAME TestCPU COMMAND test_cpu)
    add_test(NAME TestGPU COMMAND test_gpu)
else()
    message(WARNING "Google Test not found, tests will not be built")
endif()

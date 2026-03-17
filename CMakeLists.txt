cmake_minimum_required(VERSION 3.10) # 指定cmake版本

# 设置MPI编译器（需在project之前）
# 与 PETSc 安装时使用的 MPICH wrapper 保持一致，避免与系统 OpenMPI 混用
set(CMAKE_C_COMPILER "/home/JTZ_DL/petsc-3.23.0/x86_64/bin/mpicc")
set(CMAKE_CXX_COMPILER "/home/JTZ_DL/petsc-3.23.0/x86_64/bin/mpicxx")

project(DUAL_MAC LANGUAGES C CXX) # 指定项目名称

# 生成compile_commands.json，供VSCode自动识别头文件路径
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 获取源文件
#aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
file(GLOB_RECURSE SRC_LIST CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
#设置头文件路径
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 静态库生成的路径
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 测试程序生成的路径
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 设置动态库名字
set(MAC_LIB mac_lib)

# 使用pkg-config自动发现PETSc安装路径（避免硬编码路径导致VSCode无法识别）
#set(PETSC_DIR "/opt/petsc")
#set(PETSC_ARCH "arch-linux-c-debug")
set(PETSC_DIR "/home/JTZ_DL/petsc-3.23.0")
set(PETSC_ARCH "x86_64")
include_directories(
    ${PETSC_DIR}/include
    ${PETSC_DIR}/${PETSC_ARCH}/include
    ${PETSC_DIR}/include/petsc
    ${PETSC_DIR}/${PETSC_ARCH}/include/petsc
    ${HEAD_PATH}
)
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)#指定C++标准
option(DUAL_MAC_ENABLE_DEBUG_LOG "Enable DUAL_MAC debug logs" OFF)
option(DUAL_MAC_ENABLE_CONVECTION "Enable convective term assembly" ON)
add_executable(DUAL_MAC ${SRC_LIST} )#指定可执行文件和源文件
target_include_directories(DUAL_MAC PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(DUAL_MAC PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(DUAL_MAC PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(DUAL_MAC PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

# 测试可执行文件：包含 test/main.cpp，避免与 src/main.cpp 重复定义 main
set(TEST_SRC ${SRC_LIST})
list(REMOVE_ITEM TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_executable(DUAL_MAC_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp
    ${TEST_SRC}
)
target_include_directories(DUAL_MAC_TEST PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(DUAL_MAC_TEST PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(DUAL_MAC_TEST PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(DUAL_MAC_TEST PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

# 收敛率测试可执行文件：使用 test/convergence_test.cpp
add_executable(convergence_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/convergence_test.cpp
    ${TEST_SRC}
)
target_include_directories(convergence_test PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(convergence_test PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(convergence_test PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(convergence_test PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(preserving
${CMAKE_CURRENT_SOURCE_DIR}/test/preserving.cpp
${TEST_SRC}
)
target_include_directories(preserving PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(preserving PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(preserving PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(preserving PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(stokes_convergence_test
${CMAKE_CURRENT_SOURCE_DIR}/test/stokes_convergence_test.cpp
${TEST_SRC}
)
target_include_directories(stokes_convergence_test PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(stokes_convergence_test PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(stokes_convergence_test PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(stokes_convergence_test PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(convergence_test_periodic_p
    ${CMAKE_CURRENT_SOURCE_DIR}/test/convergence_test_periodic_p.cpp
    ${TEST_SRC}
)
target_include_directories(convergence_test_periodic_p PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(convergence_test_periodic_p PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(convergence_test_periodic_p PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(convergence_test_periodic_p PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(convergence_test_CFL
    ${CMAKE_CURRENT_SOURCE_DIR}/test/convergence_test_CFL.cpp
    ${TEST_SRC}
)
target_include_directories(convergence_test_CFL PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(convergence_test_CFL PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(convergence_test_CFL PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(convergence_test_CFL PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(Reynold_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/Reynold_test.cpp
    ${TEST_SRC}
)
target_include_directories(Reynold_test PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(Reynold_test PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(Reynold_test PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(Reynold_test PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(Pressure_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/Pressure_test.cpp
    ${TEST_SRC}
)
target_include_directories(Pressure_test PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(Pressure_test PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(Pressure_test PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(Pressure_test PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(Reynold_test2
    ${CMAKE_CURRENT_SOURCE_DIR}/test/Reynold_test2.cpp
    ${TEST_SRC}
)
target_include_directories(Reynold_test2 PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(Reynold_test2 PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(Reynold_test2 PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(Reynold_test2 PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()

add_executable(Pressure_test2
    ${CMAKE_CURRENT_SOURCE_DIR}/test/Pressure_test2.cpp
    ${TEST_SRC}
)
target_include_directories(Pressure_test2 PRIVATE
    ${HEAD_PATH}
)
target_link_libraries(Pressure_test2 PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
if(DUAL_MAC_ENABLE_DEBUG_LOG)
    target_compile_definitions(Pressure_test2 PRIVATE DUAL_MAC_DEBUG=1)
endif()
if(NOT DUAL_MAC_ENABLE_CONVECTION)
    target_compile_definitions(Pressure_test2 PRIVATE DUAL_MAC_ENABLE_CONVECTION=0)
endif()
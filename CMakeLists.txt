cmake_minimum_required(VERSION 3.12)
project(cpp_mlp)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 设置 libtorch 和 pybind11 的路径
set(Torch_DIR "D:/Anaconda3/envs/DL/Lib/site-packages/libtorch/share/cmake/Torch")
set(pybind11_DIR "D:/Anaconda3/envs/DL/Lib/site-packages/pybind11/share/cmake/pybind11")

message(STATUS "Torch CMake path: ${Torch_DIR}")
message(STATUS "Pybind11 CMake path: ${pybind11_DIR}")

# 查找库：PyTorch 和 pybind11
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# 新增：查找 Python3 开发环境（包含 Python.h 和 Python 库）
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 library: ${Python3_LIBRARIES}")

# 打印是否成功找到 Torch
if (Torch_FOUND)
    message(STATUS "Torch found successfully!")
else()
    message(FATAL_ERROR "Torch not found!")
endif()

# 打印 TORCH_LIBRARIES 的内容
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

# 添加扩展模块
add_library(cpp_mlp MODULE cpp_mlp.cpp)

# 包含路径：PyTorch、pybind11、Python3
target_include_directories(cpp_mlp PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# 链接库：libtorch、pybind11、Python3
target_link_libraries(cpp_mlp PRIVATE
    ${TORCH_LIBRARIES}
    pybind11::module
    Python3::Python            # 新增：链接 Python 库
)

# 设置输出目录和扩展名
set_target_properties(cpp_mlp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "E:/Python/DLL/notebooks"
    PREFIX ""                 # 去掉默认的 lib 前缀
    SUFFIX ".pyd"             # Windows 平台使用 .pyd
)

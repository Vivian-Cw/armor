cmake_minimum_required(VERSION 3.10)
project(ArmorDetector)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(armor_detector src/main.cpp include/armor_detect.hpp)

# 链接OpenCV库
target_link_libraries(armor_detector ${OpenCV_LIBS})

# 包含OpenCV头文件目录
target_include_directories(armor_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
# 包含项目头文件目录
target_include_directories(armor_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 输出信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

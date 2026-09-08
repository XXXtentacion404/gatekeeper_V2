cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(gatekeeper_V4 VERSION 4.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SERIAL "Enable RS-485 serial communication" OFF)
if(ENABLE_SERIAL)
    add_compile_definitions(ENABLE_SERIAL)
endif()

set(CMAKE_C_FLAGS   "-O2 -pipe -fPIC -w")
set(CMAKE_CXX_FLAGS "-O2 -pipe -fPIC -w -std=c++17")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(LIB_ARCH "aarch64")

find_package(X11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)

# === RKNN ===
set(RKNN_BASE ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
if(EXISTS ${RKNN_BASE}/include/rknn_api.h)
    set(RKNN_INC ${RKNN_BASE}/include)
elseif(EXISTS /usr/include/rknn_api.h)
    set(RKNN_INC /usr/include)
else()
    message(FATAL_ERROR "rknn_api.h not found!")
endif()
if(EXISTS ${RKNN_BASE}/aarch64/librknnrt.so)
    set(RKNN_LIB ${RKNN_BASE}/aarch64/librknnrt.so)
else()
    set(RKNN_LIB "rknnrt")
endif()

# === RGA ===
set(RGA_BASE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/RK3588)
if(EXISTS ${RGA_BASE}/lib/Linux/aarch64/librga.so)
    set(RGA_LIB ${RGA_BASE}/lib/Linux/aarch64/librga.so)
    set(RGA_INC ${RGA_BASE}/include)
else()
    # 头文件搜索：系统路径 + 系统子目录(rga/) + 项目自带 3rdparty/rga/include
    find_path(RGA_INC im2d.hpp
        PATHS /usr/include /usr/local/include ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/include
        PATH_SUFFIXES rga)
    find_path(RGA_INC_EXTRA rga.h
        PATHS /usr/include /usr/local/include ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/include
        PATH_SUFFIXES rga)
    find_library(RGA_LIB NAMES rga librga PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
    if(NOT RGA_INC OR NOT RGA_INC_EXTRA OR NOT RGA_LIB)
        set(RGA_LIB "")
        set(RGA_INC "")
        set(RGA_INC_EXTRA "")
    elseif(NOT RGA_INC STREQUAL RGA_INC_EXTRA)
        list(APPEND RGA_INC ${RGA_INC_EXTRA})
    endif()
endif()
if(RGA_INC AND RGA_LIB)
    add_compile_definitions(HAS_RGA=1)
    message(STATUS "RGA enabled: inc=${RGA_INC} lib=${RGA_LIB}")
else()
    message(STATUS "RGA disabled (headers/lib not found), fallback to OpenCV preprocess")
endif()

# === 海康SDK ===
set(HIK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/hiklib)
set(HIK_LIBS -L${HIK_PATH} -Wl,-rpath=${HIK_PATH} -lhcnetsdk -lPlayCtrl -lssl -lcrypto -ldl X11)

# === include ===
include_directories(
    ${OpenCV_INCLUDE_DIRS} ${RKNN_INC} ${RGA_INC} ${HIK_PATH}
    ${SQLite3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bytetrack
)
link_directories(${HIK_PATH} /usr/lib/aarch64-linux-gnu)

# === 库 ===
add_library(nn_process SHARED src/preprocess.cpp src/postprocess.cpp)
target_link_libraries(nn_process ${OpenCV_LIBS} ${RGA_LIB})

add_library(rknn_engine SHARED src/rknn_engine.cpp)
target_link_libraries(rknn_engine ${RKNN_LIB})

add_library(yolov8_lib SHARED src/yolov8_custom.cpp)
target_link_libraries(yolov8_lib rknn_engine nn_process)

add_library(bytetrack_lib SHARED src/tracker.cpp)
target_link_libraries(bytetrack_lib ${OpenCV_LIBS})

add_library(draw_lib SHARED src/cv_draw.cpp)
target_link_libraries(draw_lib ${OpenCV_LIBS})

# === 可执行文件 ===
add_executable(main_tracking src/main_tracking.cpp src/yolov8_thread_pool.cpp src/mask_utils.cpp src/gate_id_counter.cpp src/comm.cpp)
target_link_libraries(main_tracking draw_lib yolov8_lib bytetrack_lib ${HIK_LIBS} ${OpenCV_LIBS} pthread dl ${SQLite3_LIBRARIES})
install(TARGETS main_tracking DESTINATION bin)

add_executable(video_debug src/video_debug.cpp src/yolov8_thread_pool.cpp src/mask_utils.cpp src/tripwire_counter.cpp)
target_link_libraries(video_debug draw_lib yolov8_lib bytetrack_lib ${OpenCV_LIBS} pthread dl)
install(TARGETS video_debug DESTINATION bin)

add_executable(main_headless
    src/main_headless.cpp
    src/yolov8_thread_pool.cpp
    src/mask_utils.cpp
    src/comm.cpp
    src/yolov8_custom.cpp
    src/tracker.cpp
    src/rknn_engine.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)
target_link_libraries(main_headless ${HIK_LIBS} ${OpenCV_LIBS} pthread dl ${SQLite3_LIBRARIES} ${RKNN_LIB} ${RGA_LIB})
install(TARGETS main_headless DESTINATION bin)

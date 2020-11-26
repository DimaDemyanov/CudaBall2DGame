# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.9)

project ("CudaKursachCMakeV2.0")
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/external/freeglut)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

SET(GLUT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/freeglut)
SET(GLEW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/glew)
SET(FREETYPE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/freetype/include)
SET(FREEGLUT_LIBRARY ${PROJECT_SOURCE_DIR}/external/freeglut/lib/x64/freeglut.lib)
SET(GLEW_LIBRARY ${PROJECT_SOURCE_DIR}/external/glew/lib/Release/x64/glew32.lib)
SET(GLUT_LIBRARY ${PROJECT_SOURCE_DIR}/external/glut/glut32.lib)
SET(FREETYPE_LIBRARY ${PROJECT_SOURCE_DIR}/external/freetype/static/win64/freetype.lib)
SET(GLTEXT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/gltext)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS})

SET(DRAW_TEXT_SOURCES
    src/libdrawtext/draw.c
    src/libdrawtext/drawgl.c
    src/libdrawtext/drawrast.c
    src/libdrawtext/font.c
    src/libdrawtext/tpool.c
    src/libdrawtext/utf8.c
)

find_package(CUDA QUIET REQUIRED)

enable_language(CUDA) 

#message("DIMA:", ${GLUT_LIBRARY} , ${GLUT_INCLUDE_DIRS}, ${GLUT_FOUND})
message("DIMA:", ${OPENGL_LIBRARY} , ${OPENGL_INCLUDE_DIRS}, ${OPENGL_FOUND})

add_executable (CudaKursachCMakeV2.0 "src/Main.cpp" "src/Ball.cu")

target_link_libraries(CudaKursachCMakeV2.0 ${OPENGL_LIBRARY} ${GLUT_LIBRARY} ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${GLTEXT_LIBRARIES} ${FREETYPE_LIBRARY})
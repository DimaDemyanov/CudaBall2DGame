# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.9)

project ("CudaKursachCMakeV2.0")
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/external/freeglut)
enable_language(CUDA) 

SET(CUDA_TOOLKIT_ROOT_DIR "C:\\")
SET(CUDA_BIN_PATH "C:\\")
SET(CUDA_PATH "C:\\")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)

SET(GLUT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/freeglut)
SET(GLEW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/glew)
SET(FREETYPE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/freetype/include)
SET(FREEGLUT_LIBRARY ${PROJECT_SOURCE_DIR}/external/freeglut/lib/x64/freeglut.lib)
SET(GLEW_LIBRARY ${PROJECT_SOURCE_DIR}/external/glew/lib/Release/x64/glew32.lib)
SET(GLUT_LIBRARY ${PROJECT_SOURCE_DIR}/external/glut/glut32.lib)
SET(FREETYPE_LIBRARY ${PROJECT_SOURCE_DIR}/external/freetype/static/win64/freetype.lib)
SET(GLTEXT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/gltext)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS})


#add_subdirectory (src/CudaLibrary)

#message("DIMA:", ${GLUT_LIBRARY} , ${GLUT_INCLUDE_DIRS}, ${GLUT_FOUND})
#message("DIMA:", ${OPENGL_LIBRARY} , ${OPENGL_INCLUDE_DIRS}, ${OPENGL_FOUND})

cuda_add_executable (CudaKursachCMakeV2.0 "src/Main.cpp" "src/CudaBall.cu")

target_link_libraries(CudaKursachCMakeV2.0 ${OPENGL_LIBRARY} ${GLUT_LIBRARY} ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${GLTEXT_LIBRARIES} ${FREETYPE_LIBRARY})
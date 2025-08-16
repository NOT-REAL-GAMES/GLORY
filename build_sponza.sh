#!/bin/bash

# GLORY Sponza Build Script - handles problematic dependencies
set -e

echo "🏛️ Building GLORY for Sponza..."

# Clean start
rm -rf build_sponza
mkdir build_sponza
cd build_sponza

echo "📦 Phase 1: Configuring essential dependencies..."

# Use the minimal CMakeLists.txt for now
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(GLORY)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
include(FetchContent)

# Essential dependencies only
FetchContent_Declare(
    SDL3 STATIC
    GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
    GIT_TAG "main"
    GIT_SHALLOW TRUE
)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(SDL3)

FetchContent_Declare(
    glm STATIC
    GIT_REPOSITORY "https://github.com/g-truc/glm.git"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    volk STATIC
    GIT_REPOSITORY "https://github.com/zeux/volk.git"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

FetchContent_Declare(
    glslang STATIC
    GIT_REPOSITORY "https://github.com/KhronosGroup/glslang.git"
    GIT_TAG "main"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glslang)

FetchContent_Declare(
    VulkanMemoryAllocator STATIC
    GIT_REPOSITORY "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git"
    GIT_TAG "v3.3.0"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

FetchContent_Declare(
    assimp STATIC
    GIT_REPOSITORY "https://github.com/assimp/assimp.git"
    GIT_TAG "v5.4.3"
    GIT_SHALLOW TRUE
)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_GLTF_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_OBJ_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

FetchContent_Declare(
    stb STATIC
    GIT_REPOSITORY "https://github.com/nothings/stb.git"
    GIT_TAG "master"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

add_executable(GLORY
    ../main.cpp
    ../vma_impl.cpp
)

target_link_libraries(GLORY PUBLIC 
    SDL3::SDL3 
    Vulkan::Vulkan 
    volk::volk
    VulkanMemoryAllocator 
    glm::glm 
    glslang::glslang
    assimp
)

target_include_directories(GLORY PRIVATE ${stb_SOURCE_DIR})
EOF

echo "⚙️ Phase 2: Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release .

echo "🔨 Phase 3: Building GLORY..."
make -j$(nproc)

echo "✅ Build complete! Executable: $(pwd)/GLORY"
echo "📁 Assets directory: $(pwd)/../assets/"
echo ""
echo "🏛️ To test Sponza:"
echo "   1. Download Sponza GLTF model"
echo "   2. Place in $(pwd)/../assets/models/sponza/Sponza.gltf"
echo "   3. Run: $(pwd)/GLORY"
EOF
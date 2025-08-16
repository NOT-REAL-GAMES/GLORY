cmake_minimum_required(VERSION 3.16)
project(GLORY)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force these settings for all targets
set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard" FORCE)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    add_compile_options(/std:c++latest)
    add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR)
endif()

set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

find_package(Vulkan REQUIRED)

include(FetchContent)
# Temporarily commented out to avoid build hangs - can re-enable after Sponza works
# FetchContent_Declare(
#     JoltPhysics STATIC
#     GIT_REPOSITORY "https://github.com/jrouwe/JoltPhysics.git"
#     GIT_TAG "v5.3.0"  # Try latest version
#     SOURCE_SUBDIR "Build"
#     GIT_SHALLOW TRUE
# )
# 
# # Optimize Jolt build
# set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
# set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
# set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
# set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
# set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
# 
# FetchContent_MakeAvailable(JoltPhysics)

FetchContent_Declare(
    SDL3 STATIC
    GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
    GIT_TAG "main"
    GIT_SHALLOW TRUE
)

# Optimize SDL3 build 
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(SDL3)

FetchContent_Declare(
    glm STATIC
    GIT_REPOSITORY "https://github.com/g-truc/glm.git"
)

FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    spirv-cross STATIC
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Cross.git"
    GIT_TAG "main"
)

FetchContent_MakeAvailable(spirv-cross)

FetchContent_Declare(
    SPIRV-Headers STATIC
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Headers.git"
    GIT_TAG "main"
)

FetchContent_MakeAvailable(SPIRV-Headers)

FetchContent_Declare(
    SPIRV-Tools STATIC
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Tools.git"
    GIT_TAG "main"
)

FetchContent_MakeAvailable(SPIRV-Tools)

FetchContent_Declare(
    volk STATIC
    GIT_REPOSITORY "https://github.com/zeux/volk.git"
)

FetchContent_MakeAvailable(volk)

FetchContent_Declare(
    glslang STATIC
    GIT_REPOSITORY "https://github.com/KhronosGroup/glslang.git"
    GIT_TAG "main"

)

FetchContent_MakeAvailable(glslang)

# Temporarily commented out to avoid build hangs - can re-enable after Sponza works  
# FetchContent_Declare(
#     flecs STATIC
#     GIT_REPOSITORY "https://github.com/SanderMertens/flecs.git"
#     GIT_TAG "v4.1.0"
#     GIT_SHALLOW TRUE
# )
# 
# # Optimize flecs build - be more aggressive
# set(FLECS_STATIC ON CACHE BOOL "" FORCE)
# set(FLECS_SHARED OFF CACHE BOOL "" FORCE)
# set(FLECS_PIC OFF CACHE BOOL "" FORCE)
# set(FLECS_TESTS OFF CACHE BOOL "" FORCE)
# set(FLECS_EXAMPLES OFF CACHE BOOL "" FORCE)
# set(FLECS_APPS OFF CACHE BOOL "" FORCE)
# set(FLECS_SOFT_ASSERT OFF CACHE BOOL "" FORCE)
# set(FLECS_CPP_NO_AUTO_REGISTRATION ON CACHE BOOL "" FORCE)
# 
# FetchContent_MakeAvailable(flecs)

FetchContent_Declare(
    VulkanMemoryAllocator STATIC
    GIT_REPOSITORY "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git"
    GIT_TAG "v3.3.0"  # Try latest version
)

FetchContent_MakeAvailable(VulkanMemoryAllocator)

FetchContent_Declare(
    tinyexr STATIC
    GIT_REPOSITORY "https://github.com/syoyo/tinyexr.git"
    GIT_TAG "v1.0.12"  # Try latest version
)

FetchContent_MakeAvailable(tinyexr)

FetchContent_Declare(
    assimp STATIC
    GIT_REPOSITORY "https://github.com/assimp/assimp.git"
    GIT_TAG "v5.4.3"
    GIT_SHALLOW TRUE
)

# Disable most Assimp features to speed up build and avoid hangs
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_GLTF_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_OBJ_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_FBX_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_VIEW OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_MINIZIP OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(assimp)

FetchContent_Declare(
    stb STATIC
    GIT_REPOSITORY "https://github.com/nothings/stb.git"
    GIT_TAG "master"
)

FetchContent_MakeAvailable(stb)



add_executable(GLORY
    main.cpp
    vma_impl.cpp
)

target_link_libraries(GLORY PUBLIC 
    SDL3::SDL3 
    Vulkan::Vulkan 
    volk::volk
    # Jolt::Jolt  # Commented out temporarily
    VulkanMemoryAllocator 
    # flecs       # Commented out temporarily
    glm::glm 
    glslang::glslang
    SPIRV-Tools
    SPIRV-Headers
    tinyexr
    assimp
    )

target_include_directories(GLORY PRIVATE ${stb_SOURCE_DIR})

if(MSVC)
    target_link_options(GLORY PRIVATE "/STACK:8388608")  # 8MB stack instead of default 1MB
    target_compile_options(GLORY PRIVATE /Zp8)  # Force 8-byte alignment
endif()

if(WIN32)
   set(VULKAN_DLL_SOURCE_PATH "C:\\Windows\\System32\\vulkan-1.dll")

    add_custom_command(
        TARGET GLORY # Replace with the name of your executable
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${VULKAN_DLL_SOURCE_PATH}"
                "$<TARGET_FILE_DIR:GLORY>" # Copies to the directory of your executable
        COMMENT "Copying vulkan-1.dll to build directory"
    )


    add_custom_command(
        TARGET GLORY
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_RUNTIME_DLLS:GLORY>
                $<TARGET_FILE_DIR:GLORY>
        COMMAND_EXPAND_LISTS
    )
endif()
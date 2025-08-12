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
FetchContent_Declare(
    JoltPhysics STATIC
    GIT_REPOSITORY "https://github.com/jrouwe/JoltPhysics.git"
    GIT_TAG "v5.3.0"  # Try latest version
    SOURCE_SUBDIR "Build"
)
FetchContent_MakeAvailable(JoltPhysics)

FetchContent_Declare(
    SDL3 STATIC
    GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
    GIT_TAG "main"
)

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

FetchContent_Declare(
    flecs STATIC
    GIT_REPOSITORY "https://github.com/SanderMertens/flecs.git"
    GIT_TAG "v4.1.0"  # Try latest version
)

FetchContent_MakeAvailable(flecs)

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



add_executable(GLORY
    main.cpp
    vma_impl.cpp
)

target_link_libraries(GLORY PUBLIC 
    SDL3::SDL3 
    Vulkan::Vulkan 
    volk::volk
    Jolt::Jolt 
    VulkanMemoryAllocator 
    flecs 
    glm::glm 
    glslang::glslang
    SPIRV-Tools
    SPIRV-Headers
    tinyexr
    )

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
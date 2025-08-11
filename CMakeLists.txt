cmake_minimum_required(VERSION 3.16)
project(GLORY)

set(CMAKE_CXX_STANDARD 23)

if (MSVC)
    add_compile_options(/LTCG /MDd) 
endif()

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

add_executable(GLORY
    main.cpp
)

target_link_libraries(GLORY PUBLIC SDL3::SDL3 Vulkan::Vulkan Jolt::Jolt VulkanMemoryAllocator flecs glm::glm)


if(WIN32)
    add_custom_command(
        TARGET GLORY
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_RUNTIME_DLLS:GRIFT>
                $<TARGET_FILE_DIR:GRIFT>
        COMMAND_EXPAND_LISTS
    )
endif()
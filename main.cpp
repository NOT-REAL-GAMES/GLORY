#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#define VK_NO_PROTOTYPES
#include <volk.h>

#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_ASSERT_ON_RESULT
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0  // Force dynamic loading
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <chrono>

vk::detail::DispatchLoaderDynamic dl;

// Vertex structure for PBR rendering
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
    
    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);
    }
    
    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, tangent))
        }};
    }
};

// PBR Material properties
struct PBRMaterial {
    alignas(16) glm::vec3 albedo{1.0f, 1.0f, 1.0f};
    alignas(4) float metallic{0.0f};
    alignas(4) float roughness{0.5f};
    alignas(4) float ao{1.0f};
};

// Camera/View uniform buffer
struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 viewPos;
    alignas(16) glm::mat4 invView;

};


// Lighting uniform buffer
struct LightUBO {
    glm::vec4 lightPositions[4];
    glm::vec4 lightColors[4];
};


class PBRRenderer {
private:
    SDL_Window* window;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    
    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    
    // Synchronization objects
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    
    // Vertex data
    vk::Buffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;
    vk::Buffer indexBuffer;
    VmaAllocation indexBufferAllocation;
    
    // Uniform buffers
    std::vector<vk::Buffer> cameraUniformBuffers;
    std::vector<VmaAllocation> cameraUniformBufferAllocations;
    std::vector<vk::Buffer> lightUniformBuffers;
    std::vector<VmaAllocation> lightUniformBufferAllocations;
    std::vector<vk::Buffer> materialUniformBuffers;
    std::vector<VmaAllocation> materialUniformBufferAllocations;
    
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    
    VmaAllocator allocator;
    uint32_t currentFrame = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 2;
    
    // Simple sphere vertices for PBR demo
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Shader source strings
    const std::string vertShaderSource = R"(
#version 450

layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 viewPos;
    vec3 invViewPos;
} camera;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragViewPos;
layout(location = 4) out vec3 fragInvViewPos;

void main() {
    fragPos = inPosition;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    fragViewPos = camera.viewPos;
    fragInvViewPos = camera.invViewPos;
    
    gl_Position = camera.proj * camera.view * vec4(inPosition, 1.0);
}
)";
    
    const std::string fragShaderSource = R"(
#version 450

layout(binding = 1) uniform LightUBO {
    vec3 lightPositions[4];
    vec3 lightColors[4];
} lights;

layout(binding = 2) uniform PBRMaterial {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
} material;

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragViewPos;
layout(location = 4) in vec3 fragInvViewPos;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

vec3 agx(vec3 val) {
    const mat3 agx_mat = mat3(
        0.842479062253094, 0.0784335999999992,  0.0792237451477643,
        0.0423282422610123, 0.878468636469772,  0.0791661274605434,
        0.0423756549057051, 0.0784336,          0.879142973793104
    );
    
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    
    // Input transform
    val = agx_mat * val;
    
    // Log2 space encoding
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);
    
    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);
    
    return val;
}

vec3 agxEotf(vec3 val) {
    const mat3 agx_mat_inv = mat3(
        1.19687900512017, -0.0980208811401368, -0.0990297440797205,
        -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
        -0.0529716355144438, -0.0980434501171241, 1.15107367264116
    );
    
    // Inverse tonemap
    val = agx_mat_inv * val;
    
    return val;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r)/8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 1.0);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(fragPos - fragViewPos);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.albedo, material.metallic);
    
    vec3 ambient = vec3(0.0);

    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lights.lightPositions[i] - fragPos);
        vec3 H = normalize((V + L));
        float distance = -length( fragPos - lights.lightPositions[i]);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lights.lightColors[i] * attenuation;
        
        float NDF = DistributionGGX(N, H, material.roughness);
        float G = GeometrySmith(N, V, L, material.roughness);

        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float VdotL = max(dot(V, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);

        float wrap = 0.001;

        vec3 F = fresnelSchlick(max(dot(H, V), 0.0)*2.0, F0);
            
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - material.metallic;

        vec3 specular = vec3(0.0);

        if (NdotL > 0.0 && NdotV > 0.0) {


            //G = min(1.0, min(2.0 * NdotH * NdotV / VdotH, 2.0 * NdotH * NdotL / VdotH));


            
            vec3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
            specular = numerator / denominator;
        }

        vec3 diffuse = kD * material.albedo / PI * NdotL;

            // For rough surfaces, use a softer falloff
            float softNdotL = smoothstep(-material.roughness, 1.0, NdotL);
            // Standard BRDF for smooth surfaces
            Lo += (diffuse * radiance) + (specular * radiance * softNdotL);
            
            float viewRim = 1.0 - max(6.0 * softNdotL * NdotH * NdotV / -VdotH*8.0, 0.0);
            viewRim = pow(viewRim, 2.5); // Tighten the rim

            // Modulate rim by light contribution
            float lightInfluence = smoothstep(-0.01, viewRim, softNdotL); // How much this light affects the rim
            float rim = viewRim * lightInfluence;

            // Make rim stronger on smooth surfaces
            rim *= (1.0 - material.roughness * 0.5);

            vec3 rimColor = lights.lightColors[i] * rim * 0.02; // Use actual light color
            Lo += rimColor * radiance;


    }
    
    ambient += vec3(0.003) * material.albedo * material.ao;
    vec3 color = ambient + Lo;
    
    // HDR tonemapping
    color = agx(color);
    color = pow(agxEotf(color),vec3(2.0));

    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    outColor = vec4(color, 1.0);
}
)";

public:
    bool initialize() {
        if (!initSDL()) return false;
        if (!initVulkan()) return false;
        if (!createVMA()) return false;
        if (!createSwapChain()) return false;
        if (!createRenderPass()) return false;
        if (!createDescriptorSetLayout()) return false;
        if (!createGraphicsPipeline()) return false;
        if (!createFramebuffers()) return false;
        if (!createCommandPool()) return false;
        if (!createVertexBuffer()) return false;
        if (!createUniformBuffers()) return false;
        if (!createDescriptorPool()) return false;
        if (!createDescriptorSets()) return false;
        if (!createCommandBuffers()) return false;
        if (!createSyncObjects()) return false;
        
        return true;
    }
    
    void run() {
        bool running = true;
        SDL_Event event;
        
        while (running) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) {
                    running = false;
                }
            }
            
            drawFrame();
        }
        
        device.waitIdle();
    }
    
    void cleanup() {
        if (device) {
            device.waitIdle();
            
            // Cleanup sync objects
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                device.destroySemaphore(renderFinishedSemaphores[i]);
                device.destroySemaphore(imageAvailableSemaphores[i]);
                device.destroyFence(inFlightFences[i]);
            }
            
            // Cleanup buffers
            if (vertexBuffer) {
                vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferAllocation);
            }
            if (indexBuffer) {
                vmaDestroyBuffer(allocator, indexBuffer, indexBufferAllocation);
            }
            
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vmaDestroyBuffer(allocator, cameraUniformBuffers[i], cameraUniformBufferAllocations[i]);
                vmaDestroyBuffer(allocator, lightUniformBuffers[i], lightUniformBufferAllocations[i]);
                vmaDestroyBuffer(allocator, materialUniformBuffers[i], materialUniformBufferAllocations[i]);
            }
            
            // Cleanup Vulkan objects
            device.destroyDescriptorPool(descriptorPool);
            device.destroyCommandPool(commandPool);
            
            for (auto framebuffer : swapChainFramebuffers) {
                device.destroyFramebuffer(framebuffer);
            }
            
            device.destroyPipeline(graphicsPipeline);
            device.destroyPipelineLayout(pipelineLayout);
            device.destroyRenderPass(renderPass);
            device.destroyDescriptorSetLayout(descriptorSetLayout);
            
            for (auto imageView : swapChainImageViews) {
                device.destroyImageView(imageView);
            }
            
            device.destroySwapchainKHR(swapChain);
            
            // Cleanup VMA
            if (allocator != VK_NULL_HANDLE) {
                vmaDestroyAllocator(allocator);
            }
            
            device.destroy();
        }
        
        if (surface) {
            instance.destroySurfaceKHR(surface);
        }
        if (instance) {
            instance.destroy();
        }
        
        if (window) {
            SDL_DestroyWindow(window);
        }
        SDL_Quit();
    }
    
private:
    bool initSDL() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "Failed to initialize SDL\n";
            return false;
        }
        
        window = SDL_CreateWindow("PBR Vulkan Example", 1600, 900, SDL_WINDOW_VULKAN);
        if (!window) {
            std::cerr << "Failed to create window\n";
            return false;
        }
        
        return true;
    }
    
    bool initVulkan() {
        try {

        if (!SDL_Vulkan_LoadLibrary(nullptr)) {
            std::cerr << "Failed to load Vulkan library: " << SDL_GetError() << std::endl;
            return false;
        }
                        

        dl.init();
        
        VULKAN_HPP_DEFAULT_DISPATCHER.init(dl.vkGetInstanceProcAddr);

        // Create Vulkan instance
        auto appInfo = vk::ApplicationInfo(
            "PBR Example",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_4
        );
        
        // Get required extensions from SDL
        uint32_t extensionCount = 0;
        auto extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        
        std::vector<const char*> requiredExtensions;
        for (uint32_t i = 0; i < extensionCount; i++) {
            requiredExtensions.push_back(extensions[i]);
        }

        #ifdef __APPLE__
        // Add MoltenVK-specific extensions
        requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        requiredExtensions.push_back("VK_KHR_get_physical_device_properties2");
        #endif

        auto createInfo = vk::InstanceCreateInfo();

        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = 0;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;
        
        #ifdef __APPLE__
        // Enable portability enumeration for MoltenVK
        createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        #endif

        instance = vk::createInstance(createInfo).value;
        
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

        // Load instance-level functions into dispatch loader
        dl.init(instance);
        
        // Create surface
        VkSurfaceKHR s;

        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &s)) {
            std::cerr << "Failed to create surface\n";
            return false;
        }

        surface = s;
        
        // Pick physical device and create logical device
        if (!pickPhysicalDevice() || !createLogicalDevice()) {
            return false;
        }
        
        return true;
        } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    }
    
    bool pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices().value;
        
        if (devices.empty()) {
            std::cerr << "No GPUs with Vulkan support\n";
            return false;
        }
        
        // Just pick the first suitable device
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        
        if (!physicalDevice) {
            std::cerr << "No suitable GPU found\n";
            return false;
        }
        
        return true;
    }
    
    bool isDeviceSuitable(vk::PhysicalDevice device) {
        auto queueFamilies = device.getQueueFamilyProperties();
        
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                if (device.getSurfaceSupportKHR(i, surface).value) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    uint32_t findGraphicsQueueFamily() {
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                if (physicalDevice.getSurfaceSupportKHR(i, surface).value) {
                    return i;
                }
            }
        }
        
        return UINT32_MAX;
    }
    
    bool createLogicalDevice() {
        uint32_t graphicsFamily = findGraphicsQueueFamily();
        if (graphicsFamily == UINT32_MAX) {
            return false;
        }
        
        float queuePriority = 1.0f;
        auto queueCreateInfo = vk::DeviceQueueCreateInfo();
            queueCreateInfo.queueFamilyIndex = graphicsFamily,
            queueCreateInfo.queueCount = 1,
            queueCreateInfo.pQueuePriorities = &queuePriority;
        
        auto deviceFeatures = vk::PhysicalDeviceFeatures{};
        
        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
        
        auto createInfo = vk::DeviceCreateInfo();
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.enabledLayerCount = 0;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
            createInfo.pEnabledFeatures = &deviceFeatures;
        
        auto result = physicalDevice.createDevice(&createInfo, nullptr, &device);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create logical device\n";
            return false;
        }
        
        graphicsQueue = device.getQueue(graphicsFamily, 0);
        
        return true;
    }
    
    bool createVMA() {
        VmaVulkanFunctions vulkanFunctions = {};
        vulkanFunctions.vkGetInstanceProcAddr = dl.vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = dl.vkGetDeviceProcAddr;
        
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.instance = instance;
        allocatorInfo.pVulkanFunctions = &vulkanFunctions;
        
        if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
            std::cerr << "Failed to create VMA allocator\n";
            return false;
        }
        
        return true;
    }
    
    bool createSwapChain() {
        // Query swap chain support
        auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        auto formats = physicalDevice.getSurfaceFormatsKHR(surface);
        auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
        
        // Choose surface format
        vk::SurfaceFormatKHR surfaceFormat = formats.value[0];
        for (const auto& availableFormat : formats.value) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && 
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                surfaceFormat = availableFormat;
                break;
            }
        }
        
        // Choose present mode
        vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
        for (const auto& availablePresentMode : presentModes.value) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                presentMode = availablePresentMode;
                break;
            }
        }
        
        // Choose extent
        vk::Extent2D extent;
        if (capabilities.value.currentExtent.width != UINT32_MAX) {
            extent = capabilities.value.currentExtent;
        } else {
            int width, height;
            SDL_GetWindowSizeInPixels(window, &width, &height);
            extent = vk::Extent2D{
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
        }
        
        uint32_t imageCount = capabilities.value.minImageCount + 1;
        if (capabilities.value.maxImageCount > 0 && imageCount > capabilities.value.maxImageCount) {
            imageCount = capabilities.value.maxImageCount;
        }
        
        auto createInfo = vk::SwapchainCreateInfoKHR();
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.preTransform = capabilities.value.currentTransform;
            createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;
            createInfo.oldSwapchain = nullptr;
        
        auto result = device.createSwapchainKHR(&createInfo, nullptr, &swapChain);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create swap chain\n";
            return false;
        }
        
        swapChainImages = device.getSwapchainImagesKHR(swapChain).value;
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
        
        // Create image views
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            auto viewCreateInfo = vk::ImageViewCreateInfo();
                viewCreateInfo.image = swapChainImages[i];
                viewCreateInfo.viewType = vk::ImageViewType::e2D;
                viewCreateInfo.format = swapChainImageFormat;
                viewCreateInfo.components = {
                    vk::ComponentSwizzle::eIdentity,
                    vk::ComponentSwizzle::eIdentity,
                    vk::ComponentSwizzle::eIdentity,
                    vk::ComponentSwizzle::eIdentity
                };
                viewCreateInfo.subresourceRange = {
                    vk::ImageAspectFlagBits::eColor,
                    0,
                    1,
                    0,
                    1
                };
            
            auto result = device.createImageView(&viewCreateInfo, nullptr, &swapChainImageViews[i]);
            if (result != vk::Result::eSuccess) {
                std::cerr << "Failed to create image view\n";
                return false;
            }
        }
        
        return true;
    }
    
    bool createRenderPass() {
        auto colorAttachment = vk::AttachmentDescription();
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = vk::SampleCountFlagBits::e1;
            colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
            colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
            colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
            colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
        
        auto colorAttachmentRef = vk::AttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
        
        auto subpass = vk::SubpassDescription();
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
        
        auto renderPassInfo = vk::RenderPassCreateInfo();
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
        
        auto result = device.createRenderPass(&renderPassInfo, nullptr, &renderPass);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create render pass\n";
            return false;
        }
        
        return true;
    }
    
    bool createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        
        // Camera uniform buffer
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        bindings[0].pImmutableSamplers = nullptr;
        bindings[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
        
        // Light uniform buffer
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].descriptorType = vk::DescriptorType::eUniformBuffer;
        bindings[1].pImmutableSamplers = nullptr;
        bindings[1].stageFlags = vk::ShaderStageFlagBits::eFragment;
        
        // Material uniform buffer
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].descriptorType = vk::DescriptorType::eUniformBuffer;
        bindings[2].pImmutableSamplers = nullptr;
        bindings[2].stageFlags = vk::ShaderStageFlagBits::eFragment;
        
        auto layoutInfo = vk::DescriptorSetLayoutCreateInfo();
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        auto result = device.createDescriptorSetLayout(&layoutInfo, nullptr, &descriptorSetLayout);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create descriptor set layout\n";
            return false;
        }
        
        return true;
    }
    
    const TBuiltInResource* getDefaultResources() {
        static const TBuiltInResource defaultResources = {
             .maxLights =  32,
             .maxClipPlanes =  6,
             .maxTextureUnits =  32,
             .maxTextureCoords =  32,
             .maxVertexAttribs =  64,
             .maxVertexUniformComponents =  4096,
             .maxVaryingFloats =  64,
             .maxVertexTextureImageUnits =  32,
             .maxCombinedTextureImageUnits =  80,
             .maxTextureImageUnits =  32,
             .maxFragmentUniformComponents =  4096,
             .maxDrawBuffers =  32,
             .maxVertexUniformVectors =  128,
             .maxVaryingVectors =  8,
             .maxFragmentUniformVectors =  16,
             .maxVertexOutputVectors =  16,
             .maxFragmentInputVectors =  15,
             .minProgramTexelOffset =  -8,
             .maxProgramTexelOffset =  7,
             .maxClipDistances =  8,
             .maxComputeWorkGroupCountX =  65535,
             .maxComputeWorkGroupCountY =  65535,
             .maxComputeWorkGroupCountZ =  65535,
             .maxComputeWorkGroupSizeX =  1024,
             .maxComputeWorkGroupSizeY =  1024,
             .maxComputeWorkGroupSizeZ =  64,
             .maxComputeUniformComponents =  1024,
             .maxComputeTextureImageUnits =  16,
             .maxComputeImageUniforms =  8,
             .maxComputeAtomicCounters =  8,
             .maxComputeAtomicCounterBuffers =  1,
             .maxVaryingComponents =  60,
             .maxVertexOutputComponents =  64,
             .maxGeometryInputComponents =  64,
             .maxGeometryOutputComponents =  128,
             .maxFragmentInputComponents =  128,
             .maxImageUnits =  8,
             .maxCombinedImageUnitsAndFragmentOutputs =  8,
             .maxCombinedShaderOutputResources =  8,
             .maxImageSamples =  0,
             .maxVertexImageUniforms =  0,
             .maxTessControlImageUniforms =  0,
             .maxTessEvaluationImageUniforms =  0,
             .maxGeometryImageUniforms =  0,
             .maxFragmentImageUniforms =  8,
             .maxCombinedImageUniforms =  8,
             .maxGeometryTextureImageUnits =  16,
             .maxGeometryOutputVertices =  256,
             .maxGeometryTotalOutputComponents =  1024,
             .maxGeometryUniformComponents =  1024,
             .maxGeometryVaryingComponents =  64,
             .maxTessControlInputComponents =  128,
             .maxTessControlOutputComponents =  128,
             .maxTessControlTextureImageUnits =  16,
             .maxTessControlUniformComponents =  1024,
             .maxTessControlTotalOutputComponents =  4096,
             .maxTessEvaluationInputComponents =  128,
             .maxTessEvaluationOutputComponents =  128,
             .maxTessEvaluationTextureImageUnits =  16,
             .maxTessEvaluationUniformComponents =  1024,
             .maxTessPatchComponents =  120,
             .maxPatchVertices =  32,
             .maxTessGenLevel =  64,
             .maxViewports =  16,
             .maxVertexAtomicCounters =  0,
             .maxTessControlAtomicCounters =  0,
             .maxTessEvaluationAtomicCounters =  0,
             .maxGeometryAtomicCounters =  0,
             .maxFragmentAtomicCounters =  8,
             .maxCombinedAtomicCounters =  8,
             .maxAtomicCounterBindings =  1,
             .maxVertexAtomicCounterBuffers =  0,
             .maxTessControlAtomicCounterBuffers =  0,
             .maxTessEvaluationAtomicCounterBuffers =  0,
             .maxGeometryAtomicCounterBuffers =  0,
             .maxFragmentAtomicCounterBuffers =  1,
             .maxCombinedAtomicCounterBuffers =  1,
             .maxAtomicCounterBufferSize =  16384,
             .maxTransformFeedbackBuffers =  4,
             .maxTransformFeedbackInterleavedComponents =  64,
             .maxCullDistances =  8,
             .maxCombinedClipAndCullDistances =  8,
             .maxSamples =  4,
             .maxMeshOutputVerticesNV =  256,
             .maxMeshOutputPrimitivesNV =  512,
             .maxMeshWorkGroupSizeX_NV =  32,
             .maxMeshWorkGroupSizeY_NV =  1,
             .maxMeshWorkGroupSizeZ_NV =  1,
             .maxTaskWorkGroupSizeX_NV =  32,
             .maxTaskWorkGroupSizeY_NV =  1,
             .maxTaskWorkGroupSizeZ_NV =  1,
             .maxMeshViewCountNV =  4,
             .limits =  {
                 .nonInductiveForLoops =  1,
                 .whileLoops =  1,
                 .doWhileLoops =  1,
                 .generalUniformIndexing =  1,
                 .generalAttributeMatrixVectorIndexing =  1,
                 .generalVaryingIndexing =  1,
                 .generalSamplerIndexing =  1,
                 .generalVariableIndexing =  1,
                 .generalConstantMatrixVectorIndexing =  1,
            }
        };
        return &defaultResources;
    }
    
    std::vector<uint32_t> compileShader(const std::string& source, EShLanguage stage) {
        const char* shaderStrings[1];
        shaderStrings[0] = source.c_str();
        
        glslang::TShader shader(stage);
        shader.setStrings(shaderStrings, 1);
        
        // Set up the environment
        int ClientInputSemanticsVersion = 100;
        glslang::EShTargetClientVersion VulkanClientVersion = glslang::EShTargetVulkan_1_0;
        glslang::EShTargetLanguageVersion TargetVersion = glslang::EShTargetSpv_1_0;
        
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, ClientInputSemanticsVersion);
        shader.setEnvClient(glslang::EShClientVulkan, VulkanClientVersion);
        shader.setEnvTarget(glslang::EShTargetSpv, TargetVersion);
        
        const TBuiltInResource* Resources = getDefaultResources();
        
        EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
        
        if (!shader.parse(Resources, 100, false, messages)) {
            std::cerr << "GLSL Parsing Failed for shader stage " << stage << std::endl;
            std::cerr << shader.getInfoLog() << std::endl;
            std::cerr << shader.getInfoDebugLog() << std::endl;
            return {};
        }
        
        glslang::TProgram program;
        program.addShader(&shader);
        
        if (!program.link(messages)) {
            std::cerr << "GLSL Linking Failed" << std::endl;
            std::cerr << program.getInfoLog() << std::endl;
            std::cerr << program.getInfoDebugLog() << std::endl;
            return {};
        }
        
        std::vector<uint32_t> spirv;
        glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
        
        return spirv;
    }
    
    vk::ShaderModule createShaderModule(const std::vector<uint32_t>& code) {
        auto createInfo = vk::ShaderModuleCreateInfo();
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();
        
        return device.createShaderModule(createInfo).value;
    }
    
    bool createGraphicsPipeline() {
        // Initialize glslang
        if (!glslang::InitializeProcess()) {
            std::cerr << "Failed to initialize glslang process\n";
            return false;
        }
        
        // Compile vertex shader
        auto vertSpirv = compileShader(vertShaderSource, EShLangVertex);
        if (vertSpirv.empty()) {
            std::cerr << "Failed to compile vertex shader\n";
            glslang::FinalizeProcess();
            return false;
        }
        
        // Compile fragment shader  
        auto fragSpirv = compileShader(fragShaderSource, EShLangFragment);
        if (fragSpirv.empty()) {
            std::cerr << "Failed to compile fragment shader\n";
            glslang::FinalizeProcess();
            return false;
        }
        
        auto vertShaderModule = createShaderModule(vertSpirv);
        auto fragShaderModule = createShaderModule(fragSpirv);
        
        auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo();
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        
        auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo();
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};
        
        // Create pipeline layout
        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        
        auto result = device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create pipeline layout\n";
            device.destroyShaderModule(fragShaderModule);
            device.destroyShaderModule(vertShaderModule);
            glslang::FinalizeProcess();
            return false;
        }
        
        // Vertex input
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        
        auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo();
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        
        auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo();
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        auto viewport = vk::Viewport();
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        
        auto scissor = vk::Rect2D();
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent = swapChainExtent;
        
        auto viewportState = vk::PipelineViewportStateCreateInfo();
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
        
        auto rasterizer = vk::PipelineRasterizationStateCreateInfo();
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.lineWidth = 1.0f;
        
        auto multisampling = vk::PipelineMultisampleStateCreateInfo();
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
        multisampling.sampleShadingEnable = VK_FALSE;
        
        auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState();
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | 
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        
        auto colorBlending = vk::PipelineColorBlendStateCreateInfo();
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = vk::LogicOp::eCopy;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
        
        auto pipelineInfo = vk::GraphicsPipelineCreateInfo();
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;
        
        auto pipelineResult = device.createGraphicsPipeline(nullptr, pipelineInfo);
        if (pipelineResult.result != vk::Result::eSuccess) {
            std::cerr << "Failed to create graphics pipeline\n";
            device.destroyShaderModule(fragShaderModule);
            device.destroyShaderModule(vertShaderModule);
            glslang::FinalizeProcess();
            return false;
        }
        graphicsPipeline = pipelineResult.value;
        
        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
        
        glslang::FinalizeProcess();
        
        return true;
    }
    
    bool createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 1> attachments = {
                swapChainImageViews[i]
            };
            
            auto framebufferInfo = vk::FramebufferCreateInfo();
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            
            auto result = device.createFramebuffer(&framebufferInfo, nullptr, &swapChainFramebuffers[i]);
            if (result != vk::Result::eSuccess) {
                std::cerr << "Failed to create framebuffer\n";
                return false;
            }
        }
        
        return true;
    }
    
    bool createCommandPool() {
        uint32_t graphicsFamily = findGraphicsQueueFamily();
        
        auto poolInfo = vk::CommandPoolCreateInfo();
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsFamily;
        
        auto result = device.createCommandPool(&poolInfo, nullptr, &commandPool);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create command pool\n";
            return false;
        }
        
        return true;
    }
    
    void createSphere() {
        const int latitudeBands = 30;
        const int longitudeBands = 30;
        const float radius = 1.0f;
        const float PI = 3.14159265359f;
        
        vertices.clear();
        indices.clear();
        
        for (int lat = 0; lat <= latitudeBands; lat++) {
            float theta = lat * PI / latitudeBands;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            
            for (int lon = 0; lon <= longitudeBands; lon++) {
                float phi = lon * 2 * PI / longitudeBands;
                float sinPhi = sin(phi);
                float cosPhi = cos(phi);
                
                Vertex vertex;
                vertex.pos.x = cosPhi * sinTheta * radius;
                vertex.pos.y = cosTheta * radius;
                vertex.pos.z = sinPhi * sinTheta * radius;
                
                vertex.normal = glm::normalize(vertex.pos);
                vertex.texCoord.x = 1.0f - (float)lon / longitudeBands;
                vertex.texCoord.y = 1.0f - (float)lat / latitudeBands;
                
                // Simple tangent calculation
                glm::vec3 tangent = glm::vec3(-sinPhi * sinTheta, 0.0f, cosPhi * sinTheta);
                glm::vec3 bitangent = glm::vec3(cosPhi * cosTheta, -sinTheta, sinPhi * cosTheta);

                // Normalize
                tangent = glm::normalize(tangent);
                bitangent = glm::normalize(bitangent);

                // Ensure orthogonality (Gram-Schmidt)
                tangent = glm::normalize(tangent - glm::dot(tangent, vertex.normal) * vertex.normal);

                // Check handedness and fix if needed
                if (glm::dot(glm::cross(vertex.normal, tangent), bitangent) < 0.0f) {
                    tangent = -tangent;
                }

                vertex.tangent = tangent;
                
                vertices.push_back(vertex);
            }
        }
        
        for (int lat = 0; lat < latitudeBands; lat++) {
            for (int lon = 0; lon < longitudeBands; lon++) {
                int first = (lat * (longitudeBands + 1)) + lon;
                int second = first + longitudeBands + 1;
                
                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);
                
                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }
    }
    
    bool createVertexBuffer() {
        createSphere();
        
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        auto bufferInfo = vk::BufferCreateInfo();
        bufferInfo.size = bufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        
        if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                           reinterpret_cast<VkBuffer*>(&vertexBuffer), &vertexBufferAllocation, nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex buffer\n";
            return false;
        }
        
        void* data;
        vmaMapMemory(allocator, vertexBufferAllocation, &data);
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        vmaUnmapMemory(allocator, vertexBufferAllocation);
        
        // Create index buffer
        vk::DeviceSize indexBufferSize = sizeof(indices[0]) * indices.size();
        
        auto indexBufferInfo = vk::BufferCreateInfo();
        indexBufferInfo.size = indexBufferSize;
        indexBufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
        indexBufferInfo.sharingMode = vk::SharingMode::eExclusive;
        
        if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&indexBufferInfo), &allocInfo, 
                           reinterpret_cast<VkBuffer*>(&indexBuffer), &indexBufferAllocation, nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create index buffer\n";
            return false;
        }
        
        vmaMapMemory(allocator, indexBufferAllocation, &data);
        memcpy(data, indices.data(), static_cast<size_t>(indexBufferSize));
        vmaUnmapMemory(allocator, indexBufferAllocation);
        
        return true;
    }
    
    bool createUniformBuffers() {
        vk::DeviceSize cameraBufferSize = sizeof(CameraUBO);
        vk::DeviceSize lightBufferSize = sizeof(LightUBO);
        vk::DeviceSize materialBufferSize = sizeof(PBRMaterial);
        
        cameraUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        cameraUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        materialUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        materialUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Camera uniform buffer
            auto bufferInfo = vk::BufferCreateInfo();
            bufferInfo.size = cameraBufferSize;
            bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
            bufferInfo.sharingMode = vk::SharingMode::eExclusive;
            
            if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                               reinterpret_cast<VkBuffer*>(&cameraUniformBuffers[i]), &cameraUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
            
            // Light uniform buffer
            bufferInfo.size = lightBufferSize;
            if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                               reinterpret_cast<VkBuffer*>(&lightUniformBuffers[i]), &lightUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
            
            // Material uniform buffer
            bufferInfo.size = materialBufferSize;
            if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                               reinterpret_cast<VkBuffer*>(&materialUniformBuffers[i]), &materialUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
        }
        
        return true;
    }
    
    bool createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 3);
        
        auto poolInfo = vk::DescriptorPoolCreateInfo();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        
        auto result = device.createDescriptorPool(&poolInfo, nullptr, &descriptorPool);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }
        
        return true;
    }
    
    bool createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        auto allocInfo = vk::DescriptorSetAllocateInfo();
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        auto result = device.allocateDescriptorSets(&allocInfo, descriptorSets.data());
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to allocate descriptor sets\n";
            return false;
        }
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto cameraBufferInfo = vk::DescriptorBufferInfo();
            cameraBufferInfo.buffer = cameraUniformBuffers[i];
            cameraBufferInfo.offset = 0;
            cameraBufferInfo.range = sizeof(CameraUBO);
            
            auto lightBufferInfo = vk::DescriptorBufferInfo();
            lightBufferInfo.buffer = lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUBO);
            
            auto materialBufferInfo = vk::DescriptorBufferInfo();
            materialBufferInfo.buffer = materialUniformBuffers[i];
            materialBufferInfo.offset = 0;
            materialBufferInfo.range = sizeof(PBRMaterial);
            
            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};
            
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].pBufferInfo = &cameraBufferInfo;
            
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[1].pBufferInfo = &lightBufferInfo;
            
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[2].pBufferInfo = &materialBufferInfo;
            
            device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
        
        return true;
    }
    
    bool createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        auto allocInfo = vk::CommandBufferAllocateInfo();
        allocInfo.commandPool = commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
        
        auto result = device.allocateCommandBuffers(&allocInfo, commandBuffers.data());
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to allocate command buffers\n";
            return false;
        }
        
        return true;
    }
    
    bool createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        
        auto semaphoreInfo = vk::SemaphoreCreateInfo();
        
        auto fenceInfo = vk::FenceCreateInfo();
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto imageResult = device.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
            auto renderResult = device.createSemaphore(&semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
            auto fenceResult = device.createFence(&fenceInfo, nullptr, &inFlightFences[i]);
            
            if (imageResult != vk::Result::eSuccess || renderResult != vk::Result::eSuccess || fenceResult != vk::Result::eSuccess) {
                std::cerr << "Failed to create sync objects\n";
                return false;
            }
        }
        
        return true;
    }
    
    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        // Update camera
        CameraUBO cameraUbo{};
        cameraUbo.view = glm::lookAt(glm::vec3(/*sinf(time)*4.0f, 0.0f, cosf(time)*4.0f*/0.0f,-0.0f,-5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cameraUbo.invView = glm::inverse(cameraUbo.view);
        cameraUbo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 1000.0f);
        cameraUbo.proj[1][1] *= -1; // Flip Y for Vulkan
        cameraUbo.viewPos = cameraUbo.view[3];
        
        void* data;
        vmaMapMemory(allocator, cameraUniformBufferAllocations[currentImage], &data);
        memcpy(data, &cameraUbo, sizeof(cameraUbo));
        vmaUnmapMemory(allocator, cameraUniformBufferAllocations[currentImage]);
        
        // Update lights
        LightUBO lightUbo{};
        lightUbo.lightPositions[0] = glm::vec4(sinf(time)*5.0f, 0.0f, cosf(time)*5.0f,0.0f);
        lightUbo.lightPositions[1] = glm::vec4(sinf(2.0944f+time)*5.0f, 0.0f, cosf(2.0944f+time)*5.0f, 0.0f);
        lightUbo.lightPositions[2] = glm::vec4(sinf(4.1888f+time)*5.0f, 0.0f, cosf(4.1888f+time)*5.0f,0.0f);
        lightUbo.lightPositions[3] = glm::vec4(sinf(time)*5.0f, -10.0f, cosf(time)*5.0f,0.0f);
        
        lightUbo.lightColors[0] = glm::vec4(30.0f, 0.0f, 0.0f,0.0f);   // CYAN
        lightUbo.lightColors[1] = glm::vec4(0.0f, 30.0f, 0.0f,0.0f);   // MAGENTA
        lightUbo.lightColors[2] = glm::vec4(0.0f, 0.0f, 30.0f,0.0f);   // YELLOW
        lightUbo.lightColors[3] = glm::vec4(0.0f, 0.0f, 0.0f,0.0f); // WHITE
        
        vmaMapMemory(allocator, lightUniformBufferAllocations[currentImage], &data);
        memcpy(data, &lightUbo, sizeof(lightUbo));
        vmaUnmapMemory(allocator, lightUniformBufferAllocations[currentImage]);
        
        // Update material
        PBRMaterial material{};
        material.albedo = glm::vec3(1.0f, 1.0f, 1.0f);
        material.metallic = 0.0f;
        material.roughness = 0.5f + (sinf(time/8.0f) / 2.0f);
        material.ao = 0.5f;
        
        vmaMapMemory(allocator, materialUniformBufferAllocations[currentImage], &data);
        memcpy(data, &material, sizeof(material));
        vmaUnmapMemory(allocator, materialUniformBufferAllocations[currentImage]);
    }
    
    void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {
        auto beginInfo = vk::CommandBufferBeginInfo();
        
        auto result = commandBuffer.begin(&beginInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
        
        auto clearColor = vk::ClearValue();
        clearColor.color = vk::ClearColorValue{std::array<float, 4>{{0.0f, 0.0f, 0.0f, 1.0f}}};
        
        auto renderPassInfo = vk::RenderPassBeginInfo();
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        
        commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
        
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        
        std::array<vk::Buffer, 1> vertexBuffers = {vertexBuffer};
        std::array<vk::DeviceSize, 1> offsets = {0};
        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());
        commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
        
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        
        commandBuffer.endRenderPass();
        
        result = commandBuffer.end();
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }
    
    void drawFrame() {
        device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        
        auto acquireResult = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
        
        if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
            // Recreate swap chain
            return;
        } else if (acquireResult.result != vk::Result::eSuccess && acquireResult.result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }
        
        uint32_t imageIndex = acquireResult.value;
        
        updateUniformBuffer(currentFrame);
        
        device.resetFences(1, &inFlightFences[currentFrame]);
        
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        
        std::array<vk::Semaphore, 1> waitSemaphores = {imageAvailableSemaphores[currentFrame]};
        std::array<vk::PipelineStageFlags, 1> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        std::array<vk::Semaphore, 1> signalSemaphores = {renderFinishedSemaphores[currentFrame]};
        
        auto submitInfo = vk::SubmitInfo();
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores.data();
        
        auto submitResult = graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]);
        if (submitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }
        
        std::array<vk::SwapchainKHR, 1> swapChains = {swapChain};
        
        auto presentInfo = vk::PresentInfoKHR();
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores.data();
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains.data();
        presentInfo.pImageIndices = &imageIndex;
        
        auto presentResult = graphicsQueue.presentKHR(&presentInfo);
        
        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR) {
            // Recreate swap chain
        } else if (presentResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image!");
        }
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main() {
    PBRRenderer renderer;
    
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer\n";
        return -1;
    }
    
    try {
        renderer.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        renderer.cleanup();
        return -1;
    }
    
    renderer.cleanup();

    
    return 0;
}
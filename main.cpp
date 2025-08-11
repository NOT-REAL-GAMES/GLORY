#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk/volk.h>

// VMA implementation - this must be included in exactly one source file
#define VMA_IMPLEMENTATION
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

// Vertex structure for PBR rendering
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
    
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
        
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        
        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Vertex, tangent);
        
        return attributeDescriptions;
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
};

// Lighting uniform buffer
struct LightUBO {
    alignas(16) glm::vec3 lightPositions[4];
    alignas(16) glm::vec3 lightColors[4];
};

class PBRRenderer {
private:
    SDL_Window* window;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    
    // Synchronization objects
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    
    // Vertex data
    VkBuffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;
    VkBuffer indexBuffer;
    VmaAllocation indexBufferAllocation;
    
    // Uniform buffers
    std::vector<VkBuffer> cameraUniformBuffers;
    std::vector<VmaAllocation> cameraUniformBufferAllocations;
    std::vector<VkBuffer> lightUniformBuffers;
    std::vector<VmaAllocation> lightUniformBufferAllocations;
    std::vector<VkBuffer> materialUniformBuffers;
    std::vector<VmaAllocation> materialUniformBufferAllocations;
    
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    
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
} camera;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragViewPos;

void main() {
    fragPos = inPosition;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    fragViewPos = camera.viewPos;
    
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

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

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
    float k = (r * r) / 8.0;
    
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
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(fragViewPos - fragPos);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.albedo, material.metallic);
    
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lights.lightPositions[i] - fragPos);
        vec3 H = normalize(V + L);
        float distance = length(lights.lightPositions[i] - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lights.lightColors[i] * attenuation;
        
        float NDF = DistributionGGX(N, H, material.roughness);
        float G = GeometrySmith(N, V, L, material.roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - material.metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * material.albedo / PI + specular) * radiance * NdotL;
    }
    
    vec3 ambient = vec3(0.03) * material.albedo * material.ao;
    vec3 color = ambient + Lo;
    
    // HDR tonemapping
    color = color / (color + vec3(1.0));
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
        
        vkDeviceWaitIdle(device);
    }
    
    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            
            // Cleanup sync objects
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
                vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
                vkDestroyFence(device, inFlightFences[i], nullptr);
            }
            
            // Cleanup buffers
            if (vertexBuffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferAllocation);
            }
            if (indexBuffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, indexBuffer, indexBufferAllocation);
            }
            
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vmaDestroyBuffer(allocator, cameraUniformBuffers[i], cameraUniformBufferAllocations[i]);
                vmaDestroyBuffer(allocator, lightUniformBuffers[i], lightUniformBufferAllocations[i]);
                vmaDestroyBuffer(allocator, materialUniformBuffers[i], materialUniformBufferAllocations[i]);
            }
            
            // Cleanup Vulkan objects
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            vkDestroyCommandPool(device, commandPool, nullptr);
            
            for (auto framebuffer : swapChainFramebuffers) {
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            }
            
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyRenderPass(device, renderPass, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            
            for (auto imageView : swapChainImageViews) {
                vkDestroyImageView(device, imageView, nullptr);
            }
            
            vkDestroySwapchainKHR(device, swapChain, nullptr);
            
            // Cleanup VMA
            if (allocator != VK_NULL_HANDLE) {
                vmaDestroyAllocator(allocator);
            }
            
            vkDestroyDevice(device, nullptr);
        }
        
        if (surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
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
        
        window = SDL_CreateWindow("PBR Vulkan Example", 800, 600, SDL_WINDOW_VULKAN);
        if (!window) {
            std::cerr << "Failed to create window\n";
            return false;
        }
        
        return true;
    }
    
    bool initVulkan() {
        if (volkInitialize() != VK_SUCCESS) {
            std::cerr << "Failed to initialize volk\n";
            return false;
        }
        
        // Create Vulkan instance
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "PBR Example";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        // Get required extensions from SDL
        uint32_t extensionCount = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;
        createInfo.enabledLayerCount = 0;
        
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance\n";
            return false;
        }
        
        volkLoadInstance(instance);
        
        // Create surface
        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
            std::cerr << "Failed to create surface\n";
            return false;
        }
        
        // Pick physical device and create logical device
        if (!pickPhysicalDevice() || !createLogicalDevice()) {
            return false;
        }
        
        return true;
    }
    
    bool pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            std::cerr << "No GPUs with Vulkan support\n";
            return false;
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        // Just pick the first suitable device
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            std::cerr << "No suitable GPU found\n";
            return false;
        }
        
        return true;
    }
    
    bool isDeviceSuitable(VkPhysicalDevice device) {
        // Find graphics queue family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                if (presentSupport) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    uint32_t findGraphicsQueueFamily() {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
                if (presentSupport) {
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
        
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = graphicsFamily;
        queueCreateInfo.queueCount = 1;
        
        float queuePriority = 1.0f;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        VkPhysicalDeviceFeatures deviceFeatures{};
        
        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
        
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.enabledLayerCount = 0;
        
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            std::cerr << "Failed to create logical device\n";
            return false;
        }
        
        volkLoadDevice(device);
        vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
        
        return true;
    }
    
    bool createVMA() {
        VmaVulkanFunctions vulkanFunctions = {};
        vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_0;
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
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);
        
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());
        
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
        
        // Choose surface format
        VkSurfaceFormatKHR surfaceFormat = formats[0];
        for (const auto& availableFormat : formats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                surfaceFormat = availableFormat;
                break;
            }
        }
        
        // Choose present mode
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        for (const auto& availablePresentMode : presentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                presentMode = availablePresentMode;
                break;
            }
        }
        
        // Choose extent
        VkExtent2D extent;
        if (capabilities.currentExtent.width != UINT32_MAX) {
            extent = capabilities.currentExtent;
        } else {
            int width, height;
            SDL_GetWindowSizeInPixels(window, &width, &height);
            extent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
        }
        
        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }
        
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            std::cerr << "Failed to create swap chain\n";
            return false;
        }
        
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
        
        // Create image views
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo viewCreateInfo{};
            viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCreateInfo.image = swapChainImages[i];
            viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCreateInfo.format = swapChainImageFormat;
            viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewCreateInfo.subresourceRange.baseMipLevel = 0;
            viewCreateInfo.subresourceRange.levelCount = 1;
            viewCreateInfo.subresourceRange.baseArrayLayer = 0;
            viewCreateInfo.subresourceRange.layerCount = 1;
            
            if (vkCreateImageView(device, &viewCreateInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create image view\n";
                return false;
            }
        }
        
        return true;
    }
    
    bool createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            std::cerr << "Failed to create render pass\n";
            return false;
        }
        
        return true;
    }
    
    bool createDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        
        // Camera uniform buffer
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].pImmutableSamplers = nullptr;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        
        // Light uniform buffer
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].pImmutableSamplers = nullptr;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        
        // Material uniform buffer
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[2].pImmutableSamplers = nullptr;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
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
    
    VkShaderModule createShaderModule(const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }
        
        return shaderModule;
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
        
        VkShaderModule vertShaderModule = createShaderModule(vertSpirv);
        VkShaderModule fragShaderModule = createShaderModule(fragSpirv);
        
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        // Create pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout\n";
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            glslang::FinalizeProcess();
            return false;
        }
        
        // Vertex input
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
        
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
        
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline\n";
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            glslang::FinalizeProcess();
            return false;
        }
        
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        
        glslang::FinalizeProcess();
        
        return true;
    }
    
    bool createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };
            
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create framebuffer\n";
                return false;
            }
        }
        
        return true;
    }
    
    bool createCommandPool() {
        uint32_t graphicsFamily = findGraphicsQueueFamily();
        
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsFamily;
        
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
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
                vertex.tangent = glm::vec3(-sinPhi, 0, cosPhi);
                
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
        
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        
        if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &vertexBuffer, &vertexBufferAllocation, nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex buffer\n";
            return false;
        }
        
        void* data;
        vmaMapMemory(allocator, vertexBufferAllocation, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vmaUnmapMemory(allocator, vertexBufferAllocation);
        
        // Create index buffer
        VkDeviceSize indexBufferSize = sizeof(indices[0]) * indices.size();
        
        VkBufferCreateInfo indexBufferInfo{};
        indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indexBufferInfo.size = indexBufferSize;
        indexBufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vmaCreateBuffer(allocator, &indexBufferInfo, &allocInfo, &indexBuffer, &indexBufferAllocation, nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create index buffer\n";
            return false;
        }
        
        vmaMapMemory(allocator, indexBufferAllocation, &data);
        memcpy(data, indices.data(), (size_t) indexBufferSize);
        vmaUnmapMemory(allocator, indexBufferAllocation);
        
        return true;
    }
    
    bool createUniformBuffers() {
        VkDeviceSize cameraBufferSize = sizeof(CameraUBO);
        VkDeviceSize lightBufferSize = sizeof(LightUBO);
        VkDeviceSize materialBufferSize = sizeof(PBRMaterial);
        
        cameraUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        cameraUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        materialUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        materialUniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Camera uniform buffer
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = cameraBufferSize;
            bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            
            if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &cameraUniformBuffers[i], &cameraUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
            
            // Light uniform buffer
            bufferInfo.size = lightBufferSize;
            if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &lightUniformBuffers[i], &lightUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
            
            // Material uniform buffer
            bufferInfo.size = materialBufferSize;
            if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &materialUniformBuffers[i], &materialUniformBufferAllocations[i], nullptr) != VK_SUCCESS) {
                return false;
            }
        }
        
        return true;
    }
    
    bool createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 3);
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }
        
        return true;
    }
    
    bool createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor sets\n";
            return false;
        }
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            
            VkDescriptorBufferInfo cameraBufferInfo{};
            cameraBufferInfo.buffer = cameraUniformBuffers[i];
            cameraBufferInfo.offset = 0;
            cameraBufferInfo.range = sizeof(CameraUBO);
            
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &cameraBufferInfo;
            
            VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUBO);
            
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &lightBufferInfo;
            
            VkDescriptorBufferInfo materialBufferInfo{};
            materialBufferInfo.buffer = materialUniformBuffers[i];
            materialBufferInfo.offset = 0;
            materialBufferInfo.range = sizeof(PBRMaterial);
            
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &materialBufferInfo;
            
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
        
        return true;
    }
    
    bool createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
        
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            std::cerr << "Failed to allocate command buffers\n";
            return false;
        }
        
        return true;
    }
    
    bool createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
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
        cameraUbo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cameraUbo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        cameraUbo.proj[1][1] *= -1; // Flip Y for Vulkan
        cameraUbo.viewPos = glm::vec3(2.0f, 2.0f, 2.0f);
        
        void* data;
        vmaMapMemory(allocator, cameraUniformBufferAllocations[currentImage], &data);
        memcpy(data, &cameraUbo, sizeof(cameraUbo));
        vmaUnmapMemory(allocator, cameraUniformBufferAllocations[currentImage]);
        
        // Update lights
        LightUBO lightUbo{};
        lightUbo.lightPositions[0] = glm::vec3(-10.0f, 10.0f, 10.0f);
        lightUbo.lightPositions[1] = glm::vec3(10.0f, 10.0f, 10.0f);
        lightUbo.lightPositions[2] = glm::vec3(-10.0f, -10.0f, 10.0f);
        lightUbo.lightPositions[3] = glm::vec3(10.0f, -10.0f, 10.0f);
        
        lightUbo.lightColors[0] = glm::vec3(300.0f, 300.0f, 300.0f);
        lightUbo.lightColors[1] = glm::vec3(300.0f, 300.0f, 300.0f);
        lightUbo.lightColors[2] = glm::vec3(300.0f, 300.0f, 300.0f);
        lightUbo.lightColors[3] = glm::vec3(300.0f, 300.0f, 300.0f);
        
        vmaMapMemory(allocator, lightUniformBufferAllocations[currentImage], &data);
        memcpy(data, &lightUbo, sizeof(lightUbo));
        vmaUnmapMemory(allocator, lightUniformBufferAllocations[currentImage]);
        
        // Update material
        PBRMaterial material{};
        material.albedo = glm::vec3(1.0f, 1.0f, 1.0f);
        material.metallic = 0.5f;
        material.roughness = 0.2f;
        material.ao = 0.5f;
        
        vmaMapMemory(allocator, materialUniformBufferAllocations[currentImage], &data);
        memcpy(data, &material, sizeof(material));
        vmaUnmapMemory(allocator, materialUniformBufferAllocations[currentImage]);
    }
    
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
        
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        
        vkCmdEndRenderPass(commandBuffer);
        
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }
    
    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Recreate swap chain
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }
        
        updateUniformBuffer(currentFrame);
        
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }
        
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        
        result = vkQueuePresentKHR(graphicsQueue, &presentInfo);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // Recreate swap chain
        } else if (result != VK_SUCCESS) {
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
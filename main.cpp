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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <libgen.h>
#elif defined(__linux__)
#include <unistd.h>
#include <linux/limits.h>
#include <libgen.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

vk::detail::DispatchLoaderDynamic dl;

// Cross-platform file utilities
class FileUtils {
public:
    static bool fileExists(const std::string& path) {
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }
    
    static std::string getExecutableDirectory() {
#ifdef __APPLE__
        char path[1024];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            char* dirPath = dirname(path);
            return std::string(dirPath);
        }
#elif defined(__linux__)
        char path[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
        if (count != -1) {
            path[count] = '\0';
            char* dirPath = dirname(path);
            return std::string(dirPath);
        }
#elif defined(_WIN32)
        char path[MAX_PATH];
        if (GetModuleFileNameA(NULL, path, MAX_PATH) != 0) {
            char* lastSlash = strrchr(path, '\\');
            if (lastSlash) {
                *lastSlash = '\0';
                return std::string(path);
            }
        }
#endif
        // Fallback to current working directory
        char cwd[1024];
        //if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        //    return std::string(cwd);
        //}
        return "";
    }
    
    static std::string normalizePath(const std::string& path) {
        std::string normalized = path;
        // Convert any backslashes to forward slashes for consistency
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        return normalized;
    }
    
    static std::string joinPaths(const std::string& dir, const std::string& file) {
        std::string result = normalizePath(dir);
        std::string filename = normalizePath(file);
        
        // Remove trailing slash from directory if present
        if (!result.empty() && result.back() == '/') {
            result.pop_back();
        }
        
        // Remove leading slash from filename if present
        if (!filename.empty() && filename.front() == '/') {
            filename = filename.substr(1);
        }
        
        return result + "/" + filename;
    }
    
    static std::string getDirectory(const std::string& path) {
        std::string normalized = normalizePath(path);
        size_t pos = normalized.find_last_of('/');
        if (pos == std::string::npos) {
            return "."; // Current directory if no path separator found
        }
        return normalized.substr(0, pos);
    }
    
    static std::string resolveRelativePath(const std::string& relativePath) {
        std::string normalized = normalizePath(relativePath);
        
        // If it's already an absolute path, return as-is
        if (!normalized.empty() && normalized[0] == '/') {
            return normalized;
        }
        
        // For relative paths, resolve relative to executable directory
        std::string execDir = getExecutableDirectory();
        return joinPaths(execDir, normalized);
    }
};

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

// Texture structure
struct Texture {
    vk::Image image;
    VmaAllocation allocation;
    vk::ImageView imageView;
    vk::Sampler sampler;
    uint32_t width, height;
    uint32_t mipLevels = 1;
};

// Mesh structure
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::Buffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;
    vk::Buffer indexBuffer;
    VmaAllocation indexBufferAllocation;
    uint32_t materialIndex = 0;
    glm::mat4 transform = glm::mat4(1.0f);
};

// PBR Material properties
struct PBRMaterial {
    alignas(16) glm::vec3 albedo{1.0f, 1.0f, 1.0f};
    alignas(4) float metallic{0.0f};
    alignas(4) float roughness{0.5f};
    alignas(4) float ao{1.0f};
    alignas(4) int hasAlbedoTexture{0};
    alignas(4) int hasNormalTexture{0};
    alignas(4) int hasMetallicRoughnessTexture{0};
    alignas(4) int hasAOTexture{0};
};
// Material with texture indices
struct Material {
    PBRMaterial properties;
    int albedoTextureIndex = -1;
    int normalTextureIndex = -1;
    int metallicRoughnessTextureIndex = -1;
    std::string name;
};

// Model structure
struct Model {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::string directory;
};



// Camera/View uniform buffer
struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 viewPos;
    alignas(16) glm::mat4 invView;
};

// Free camera class
class FreeCamera {
public:
    glm::vec3 position{0.0f, 2.0f, 5.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    glm::vec3 right{1.0f, 0.0f, 0.0f};
    
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 5.0f;
    float sensitivity = 0.1f;
    float zoom = 45.0f;
    
    bool firstMouse = true;
    float lastX = 400.0f;
    float lastY = 300.0f;
    
    void updateVectors() {
        glm::vec3 newFront;
        newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        newFront.y = sin(glm::radians(pitch));
        newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(newFront);
        
        right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, front));
    }
    
    glm::mat4 getViewMatrix() {
        return glm::lookAt(position, position + front, up);
    }
    
    void processKeyboard(const bool* keyState, float deltaTime) {
        float velocity = speed * deltaTime;
        
        if (keyState[SDL_SCANCODE_W])
            position += front * velocity;
        if (keyState[SDL_SCANCODE_S])
            position -= front * velocity;
        if (keyState[SDL_SCANCODE_A])
            position -= right * velocity;
        if (keyState[SDL_SCANCODE_D])
            position += right * velocity;
        if (keyState[SDL_SCANCODE_SPACE])
            position += up * velocity;
        if (keyState[SDL_SCANCODE_LSHIFT])
            position -= up * velocity;
    }
    
    void processMouseMovement(float xpos, float ypos) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
        lastX = xpos;
        lastY = ypos;
        
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        
        yaw += xoffset;
        pitch += yoffset;
        
        // Constrain pitch
        //if (pitch > 179.0f)
        //    pitch = 179.0f;
        //if (pitch < -179.0f)
        //    pitch = -179.0f;
        
        updateVectors();
    }
    
    void processRelativeMouseMovement(float xrel, float yrel) {
        float xoffset = xrel * sensitivity;
        float yoffset = -yrel * sensitivity; // Inverted for natural camera movement
        
        yaw += xoffset;
        pitch += yoffset;
        
        // Constrain pitch but allow unlimited yaw rotation
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
        
        updateVectors();
    }
    
    void processMouseScroll(float yoffset) {
        zoom -= yoffset;
        if (zoom < 1.0f)
            zoom = 1.0f;
        if (zoom > 45.0f)
            zoom = 45.0f;
    }
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
    vk::Format textureFormat;
    
    // Shared samplers to reduce sampler count for MoltenVK compatibility
    vk::Sampler sharedTextureSampler;    // For textures with mipmaps
    vk::Sampler sharedDefaultSampler;    // For 1x1 default textures
    
    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    
    // Depth buffer
    vk::Image depthImage;
    VmaAllocation depthImageAllocation;
    vk::ImageView depthImageView;
    
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
    
    // Textures
    Texture defaultAlbedoTexture;
    Texture defaultNormalTexture;
    Texture defaultMetallicRoughnessTexture;
    std::vector<Texture> loadedTextures;
    
    // Model loading
    Model loadedModel;
    bool useModel = false;
    
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    
    VmaAllocator allocator;
    uint32_t currentFrame = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 2;
    
    // Camera and input
    FreeCamera camera;
    std::chrono::steady_clock::time_point lastFrameTime;
    bool mouseCaptured = false;
    
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
layout(location = 5) out vec3 fragTangent;

void main() {
    // Transform position to view space
    vec4 viewPos = camera.view * vec4(inPosition, 1.0);
    fragPos = viewPos.xyz;
    
    // Transform normal to view space (inverse transpose of view matrix for normals)
    mat3 normalMatrix = transpose(inverse(mat3(camera.view)));
    fragNormal = normalize(normalMatrix * inNormal);
    
    // Transform tangent to view space (same as normal transformation)
    fragTangent = normalize(normalMatrix * inTangent);
    
    fragTexCoord = inTexCoord;
    fragViewPos = camera.viewPos;
    fragInvViewPos = camera.invViewPos;
    
    gl_Position = camera.proj * viewPos;
}
)";
    
    const std::string fragShaderSource = R"(
#version 450

layout(binding = 1) uniform LightUBO {
    vec3 lightPositions[4];
    vec3 lightColors[4];
} lights;

layout(push_constant) uniform PBRMaterial {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
    int hasAlbedoTexture;
    int hasNormalTexture;
    int hasMetallicRoughnessTexture;
    int hasAOTexture;
} material;

layout(binding = 2) uniform sampler2D albedoTexture;
layout(binding = 3) uniform sampler2D normalTexture;
layout(binding = 4) uniform sampler2D metallicRoughnessTexture;

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragViewPos;
layout(location = 4) in vec3 fragInvViewPos;
layout(location = 5) in vec3 fragTangent;

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

vec3 getNormalFromMap() {
    if(false){
        // Sample normal map
        vec3 normalColor = texture(normalTexture, fragTexCoord).rgb;
        vec3 tangentNormal = normalize(normalColor * 2.0 - 1.0);
        
        // Use proper tangent vectors from vertex shader
        vec3 N = normalize(fragNormal);
        vec3 T = normalize(fragTangent);
        
        // Re-orthogonalize T with respect to N (Gram-Schmidt process)
        T = normalize(T - dot(T, N) * N);
        
        // Calculate bitangent
        vec3 B = cross(N, T);
        
        // Create TBN matrix
        mat3 TBN = mat3(T, B, N);
        
        return normalize(TBN * tangentNormal);
    }
    
    return normalize(fragNormal);
}

void main() {
    // DEBUG: Test texture sampling directly
    if (true) { // Set to true for debug
        // Show albedo texture directly with enhanced debugging
        if (material.hasAlbedoTexture != 0) {
            vec4 texSample = texture(albedoTexture, fragTexCoord);
            // Show individual channels to debug channel issues
            // R=red, G=green, B=blue, mix=normal
            float debugMode = fract(fragTexCoord.x * 4.0);
            if (debugMode < 0.25) {
                outColor = vec4(texSample.r, 0.0, 0.0, 1.0); // Red channel only
            } else if (debugMode < 0.5) {
                outColor = vec4(0.0, texSample.g, 0.0, 1.0); // Green channel only
            } else if (debugMode < 0.75) {
                outColor = vec4(0.0, 0.0, texSample.b, 1.0); // Blue channel only
            } else {
                outColor = vec4(texSample.rgb, 1.0); // Normal RGB
            }
            return;
        } else {
            outColor = vec4(1.0, 0.0, 1.0, 1.0); // Magenta if no texture
            return;
        }
    }
    
    // Sample textures
    vec3 albedo = material.hasAlbedoTexture != 0 ? 
        texture(albedoTexture, fragTexCoord).rgb * material.albedo : 
        material.albedo;
    
    vec3 metallicRoughness = material.hasMetallicRoughnessTexture != 0 ?
        texture(metallicRoughnessTexture, fragTexCoord).rgb :
        vec3(0.0, material.roughness, material.metallic);
    
    float metallic = material.hasMetallicRoughnessTexture != 0 ? 
        metallicRoughness.b : material.metallic;
    float roughness = material.hasMetallicRoughnessTexture != 0 ? 
        metallicRoughness.g : material.roughness;
    float ao = material.ao;
    
    vec3 N = getNormalFromMap();
    vec3 V = normalize(-fragPos); // In view space, camera is at origin
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    vec3 ambient = vec3(0.0);

    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lights.lightPositions[i] - fragPos);
        vec3 H = normalize((V + L));
        float distance = -length( fragPos - lights.lightPositions[i]);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lights.lightColors[i] * attenuation;
        
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);

        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float VdotL = max(dot(V, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);

        float wrap = 0.001;

        vec3 F = fresnelSchlick(max(dot(H, V), 0.0)*2.0, F0);
            
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        vec3 specular = vec3(0.0);

        if (NdotL > 0.0 && NdotV > 0.0) {


            //G = min(1.0, min(2.0 * NdotH * NdotV / VdotH, 2.0 * NdotH * NdotL / VdotH));


            
            vec3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
            specular = numerator / denominator;
        }

        vec3 diffuse = kD * albedo / PI * NdotL;

            // For rough surfaces, use a softer falloff
            float softNdotL = smoothstep(-roughness, 1.0, NdotL);
            // Standard BRDF for smooth surfaces
            Lo += (diffuse * radiance) + (specular * radiance * softNdotL);
            
            float viewRim = 1.0 - max(6.0 * softNdotL * NdotH * NdotV / -VdotH*8.0, 0.0);
            viewRim = pow(viewRim, 2.5); // Tighten the rim

            // Modulate rim by light contribution
            float lightInfluence = smoothstep(-0.01, viewRim, softNdotL); // How much this light affects the rim
            float rim = viewRim * lightInfluence;

            // Make rim stronger on smooth surfaces
            rim *= (1.0 - roughness * 0.5);

            vec3 rimColor = lights.lightColors[i] * rim * 0.002; // Use actual light color
            Lo += rimColor * radiance;


    }
    
    ambient += vec3(0.003) * albedo * ao;
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
        if (!createDepthResources()) return false;
        if (!createDescriptorSetLayout()) return false;
        if (!createGraphicsPipeline()) return false;
        if (!createFramebuffers()) return false;
        if (!createCommandPool()) return false;
        if (!createVertexBuffer()) return false;
        if (!createUniformBuffers()) return false;
        if (!createDefaultTextures()) return false;
        if (!createDescriptorPool()) return false;
        if (!createDescriptorSets()) return false;
        if (!createCommandBuffers()) return false;
        if (!createSyncObjects()) return false;
        
        return true;
    }
    
    void loadSponzaIfAvailable() {
        std::cout << "Executable directory: " << FileUtils::getExecutableDirectory() << std::endl;
        std::string sponzaPath = FileUtils::resolveRelativePath("assets/models/sponza/Sponza.gltf");
        std::cout << "Looking for Sponza model at: " << sponzaPath << std::endl;
        
        if (FileUtils::fileExists(sponzaPath)) {
            try {
                std::cout << "Loading Sponza model..." << std::endl;
                loadedModel = loadModel(sponzaPath);
                useModel = true;
                std::cout << "Sponza loaded successfully! Meshes: " << loadedModel.meshes.size() 
                         << ", Materials: " << loadedModel.materials.size() << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load Sponza: " << e.what() << std::endl;
                useModel = false;
            }
        } else {
            std::cout << "Sponza model not found at " << sponzaPath << ", using default sphere." << std::endl;
        }
    }
    
    void run() {
        // Try to load Sponza after initialization
        loadSponzaIfAvailable();
        
        // Initialize timing
        lastFrameTime = std::chrono::steady_clock::now();
        
        bool running = true;
        SDL_Event event;
        
        // Enable relative mouse mode for freecam
        SDL_SetWindowRelativeMouseMode(window,true);
        mouseCaptured = true;
        
        while (running) {
            // Calculate delta time
            auto currentFrameTime = std::chrono::steady_clock::now();
            float deltaTime = std::chrono::duration<float>(currentFrameTime - lastFrameTime).count();
            lastFrameTime = currentFrameTime;
            
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) {
                    running = false;
                }
                if (event.type == SDL_EVENT_KEY_DOWN) {
                    if (event.key.key == SDLK_R) {
                        // Reload model on R key press
                        loadSponzaIfAvailable();
                    }
                    if (event.key.key == SDLK_ESCAPE) {
                        // Toggle mouse capture
                        mouseCaptured = !mouseCaptured;
                        SDL_SetWindowRelativeMouseMode(window,mouseCaptured);
                    }
                }
                if (event.type == SDL_EVENT_MOUSE_MOTION && mouseCaptured) {
                    // Use relative mouse movement for free rotation
                    camera.processRelativeMouseMovement(static_cast<float>(event.motion.xrel), static_cast<float>(event.motion.yrel));
                }
                if (event.type == SDL_EVENT_MOUSE_WHEEL && mouseCaptured) {
                    camera.processMouseScroll(static_cast<float>(event.wheel.y));
                }
            }
            
            // Process keyboard input for camera movement
            if (mouseCaptured) {
                const bool* keyState = SDL_GetKeyboardState(nullptr);
                camera.processKeyboard(keyState, deltaTime);
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
            
            // Cleanup shared samplers first
            if (sharedTextureSampler) device.destroySampler(sharedTextureSampler);
            if (sharedDefaultSampler) device.destroySampler(sharedDefaultSampler);
            
            // Cleanup textures (don't destroy samplers since they're shared)
            auto cleanupTexture = [this](Texture& texture) {
                // Note: Don't destroy sampler since it's shared
                if (texture.imageView) device.destroyImageView(texture.imageView);
                if (texture.image) vmaDestroyImage(allocator, texture.image, texture.allocation);
            };
            
            cleanupTexture(defaultAlbedoTexture);
            cleanupTexture(defaultNormalTexture);
            cleanupTexture(defaultMetallicRoughnessTexture);
            
            for (auto& texture : loadedTextures) {
                cleanupTexture(texture);
            }
            
            // Cleanup depth buffer
            if (depthImageView) device.destroyImageView(depthImageView);
            if (depthImage) vmaDestroyImage(allocator, depthImage, depthImageAllocation);
            
            // Cleanup model buffers
            for (auto& mesh : loadedModel.meshes) {
                if (mesh.vertexBuffer) {
                    vmaDestroyBuffer(allocator, mesh.vertexBuffer, mesh.vertexBufferAllocation);
                }
                if (mesh.indexBuffer) {
                    vmaDestroyBuffer(allocator, mesh.indexBuffer, mesh.indexBufferAllocation);
                }
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
        
        // Initialize texture format after device creation
        textureFormat = findSupportedTextureFormat();
        
        // Create shared samplers to reduce sampler count for MoltenVK
        createSharedSamplers();
        
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
    
    vk::Format findDepthFormat() {
        std::vector<vk::Format> candidates = {
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint
        };
        
        for (vk::Format format : candidates) {
            auto props = physicalDevice.getFormatProperties(format);
            if ((props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) == vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
                return format;
            }
        }
        
        throw std::runtime_error("Failed to find supported depth format!");
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
        
        auto depthAttachment = vk::AttachmentDescription();
            depthAttachment.format = findDepthFormat();
            depthAttachment.samples = vk::SampleCountFlagBits::e1;
            depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
            depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
            depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
            depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        
        auto colorAttachmentRef = vk::AttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
            
        auto depthAttachmentRef = vk::AttachmentReference();
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        
        auto subpass = vk::SubpassDescription();
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;
        
        std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        auto renderPassInfo = vk::RenderPassCreateInfo();
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
        
        auto result = device.createRenderPass(&renderPassInfo, nullptr, &renderPass);
        if (result != vk::Result::eSuccess) {
            std::cerr << "Failed to create render pass\n";
            return false;
        }
        
        return true;
    }
    
    bool createDepthResources() {
        vk::Format depthFormat = findDepthFormat();
        
        createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eDepthStencilAttachment, depthImage, depthImageAllocation);
        
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        
        return true;
    }
    
    bool createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
        
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
        
        // Albedo texture (now binding 2)
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bindings[2].pImmutableSamplers = nullptr;
        bindings[2].stageFlags = vk::ShaderStageFlagBits::eFragment;
        
        // Normal texture (now binding 3)
        bindings[3].binding = 3;
        bindings[3].descriptorCount = 1;
        bindings[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bindings[3].pImmutableSamplers = nullptr;
        bindings[3].stageFlags = vk::ShaderStageFlagBits::eFragment;
        
        // Metallic/Roughness texture (now binding 4)
        bindings[4].binding = 4;
        bindings[4].descriptorCount = 1;
        bindings[4].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bindings[4].pImmutableSamplers = nullptr;
        bindings[4].stageFlags = vk::ShaderStageFlagBits::eFragment;
        
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
    
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        auto memProperties = physicalDevice.getMemoryProperties();
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    
    bool needsChannelSwizzle(vk::Format format) {
        return format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eB8G8R8A8Unorm;
    }
    
    void swizzleRGBAToBGRA(unsigned char* pixels, int width, int height) {
        for (int i = 0; i < width * height * 4; i += 4) {
            std::swap(pixels[i], pixels[i + 2]); // Swap R and B channels
        }
    }
    
    vk::Format findSupportedTextureFormat() {
        std::vector<vk::Format> candidates = {
            vk::Format::eR8G8B8A8Unorm,   // Try linear format first on Mac
            vk::Format::eR8G8B8A8Srgb,    // Preferred sRGB format
            vk::Format::eB8G8R8A8Unorm,   // Alternative linear
            vk::Format::eB8G8R8A8Srgb     // Alternative sRGB (common on Mac)
        };
        
        for (vk::Format format : candidates) {
            auto props = physicalDevice.getFormatProperties(format);
            vk::FormatFeatureFlags required = vk::FormatFeatureFlagBits::eSampledImage | 
                                              vk::FormatFeatureFlagBits::eTransferDst |
                                              vk::FormatFeatureFlagBits::eTransferSrc;
            
            if ((props.optimalTilingFeatures & required) == required) {
                std::cout << "Selected texture format: " << vk::to_string(format) << std::endl;
                return format;
            }
        }
        
        throw std::runtime_error("Failed to find supported texture format!");
    }
    
    vk::Sampler createCompatibleSampler(uint32_t mipLevels) {
        auto samplerInfo = vk::SamplerCreateInfo();
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        
        // Check if anisotropic filtering is supported
        auto deviceFeatures = physicalDevice.getFeatures();
        if (deviceFeatures.samplerAnisotropy) {
            samplerInfo.anisotropyEnable = VK_TRUE;
            auto properties = physicalDevice.getProperties();
            // Clamp to reasonable values for Mac compatibility
            samplerInfo.maxAnisotropy = std::min(properties.limits.maxSamplerAnisotropy, 4.0f);
        } else {
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.maxAnisotropy = 1.0f;
        }
        
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.mipLodBias = 0.0f; // No LOD bias for Mac compatibility
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        
        return device.createSampler(samplerInfo).value;
    }
    
    void createSharedSamplers() {
        std::cout << "Creating shared samplers for MoltenVK compatibility" << std::endl;
        
        // Create shared sampler for regular textures (with mipmaps)
        auto textureSamplerInfo = vk::SamplerCreateInfo();
        textureSamplerInfo.magFilter = vk::Filter::eLinear;
        textureSamplerInfo.minFilter = vk::Filter::eLinear;
        textureSamplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        textureSamplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        textureSamplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        
        // Check if anisotropic filtering is supported
        auto deviceFeatures = physicalDevice.getFeatures();
        if (deviceFeatures.samplerAnisotropy) {
            textureSamplerInfo.anisotropyEnable = VK_TRUE;
            auto properties = physicalDevice.getProperties();
            // Clamp to reasonable values for Mac compatibility
            textureSamplerInfo.maxAnisotropy = std::min(properties.limits.maxSamplerAnisotropy, 4.0f);
        } else {
            textureSamplerInfo.anisotropyEnable = VK_FALSE;
            textureSamplerInfo.maxAnisotropy = 1.0f;
        }
        
        textureSamplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        textureSamplerInfo.unnormalizedCoordinates = VK_FALSE;
        textureSamplerInfo.compareEnable = VK_FALSE;
        textureSamplerInfo.compareOp = vk::CompareOp::eAlways;
        textureSamplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        textureSamplerInfo.mipLodBias = 0.0f;
        textureSamplerInfo.minLod = 0.0f;
        textureSamplerInfo.maxLod = VK_LOD_CLAMP_NONE; // Allow all mip levels
        
        sharedTextureSampler = device.createSampler(textureSamplerInfo).value;
        
        // Create shared sampler for default textures (no mipmaps)
        auto defaultSamplerInfo = textureSamplerInfo; // Copy base settings
        defaultSamplerInfo.anisotropyEnable = VK_FALSE; // No anisotropy for 1x1 textures
        defaultSamplerInfo.maxAnisotropy = 1.0f;
        defaultSamplerInfo.maxLod = 0.0f; // Single mip level
        
        sharedDefaultSampler = device.createSampler(defaultSamplerInfo).value;
        
        std::cout << "Created 2 shared samplers (texture + default)" << std::endl;
    }
    
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, 
                     vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::Image& image, VmaAllocation& allocation) {
        auto imageInfo = vk::ImageCreateInfo();
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageInfo.usage = usage;
        imageInfo.samples = vk::SampleCountFlagBits::e1;
        imageInfo.sharingMode = vk::SharingMode::eExclusive;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        
        if (vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfo,
                          reinterpret_cast<VkImage*>(&image), &allocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image!");
        }
    }
    
    vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
        auto viewInfo = vk::ImageViewCreateInfo();
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        return device.createImageView(viewInfo).value;
    }
    
    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
        auto commandBuffer = beginSingleTimeCommands();
        
        auto barrier = vk::ImageMemoryBarrier();
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
        
        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("Unsupported layout transition!");
        }
        
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);
        
        endSingleTimeCommands(commandBuffer);
    }
    
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        auto commandBuffer = beginSingleTimeCommands();
        
        auto region = vk::BufferImageCopy();
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{width, height, 1};
        
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
        
        endSingleTimeCommands(commandBuffer);
    }
    
    vk::CommandBuffer beginSingleTimeCommands() {
        auto allocInfo = vk::CommandBufferAllocateInfo();
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
        
        auto commandBuffer = device.allocateCommandBuffers(allocInfo).value[0];
        
        auto beginInfo = vk::CommandBufferBeginInfo();
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        
        commandBuffer.begin(beginInfo);
        
        return commandBuffer;
    }
    
    void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
        commandBuffer.end();
        
        auto submitInfo = vk::SubmitInfo();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        graphicsQueue.submit(1, &submitInfo, nullptr);
        graphicsQueue.waitIdle();
        
        device.freeCommandBuffers(commandPool, 1, &commandBuffer);
    }
    
    Texture loadTexture(const std::string& path) {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;
        
        if (!pixels) {
            throw std::runtime_error("Failed to load texture image: " + path);
        }
        
        // Handle channel swizzling for BGR formats
        if (needsChannelSwizzle(textureFormat)) {
            std::cout << "Swizzling texture channels for BGR format: " << path << std::endl;
            swizzleRGBAToBGRA(pixels, texWidth, texHeight);
        }
        
        std::cout << "Loading texture " << path << " (" << texWidth << "x" << texHeight << ", " << texChannels << " channels)" << std::endl;
        
        // Debug: Check first few pixels to see if data is correct
        std::cout << "  First pixel RGBA: " << (int)pixels[0] << ", " << (int)pixels[1] << ", " << (int)pixels[2] << ", " << (int)pixels[3] << std::endl;
        if (texWidth > 1 || texHeight > 1) {
            int midPixel = (texHeight / 2 * texWidth + texWidth / 2) * 4;
            std::cout << "  Middle pixel RGBA: " << (int)pixels[midPixel] << ", " << (int)pixels[midPixel+1] << ", " << (int)pixels[midPixel+2] << ", " << (int)pixels[midPixel+3] << std::endl;
        }
        
        uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
        
        // Create staging buffer
        vk::Buffer stagingBuffer;
        VmaAllocation stagingAllocation;
        
        auto bufferInfo = vk::BufferCreateInfo();
        bufferInfo.size = imageSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        
        vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo,
                       reinterpret_cast<VkBuffer*>(&stagingBuffer), &stagingAllocation, nullptr);
        
        void* data;
        vmaMapMemory(allocator, stagingAllocation, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        
        // Debug: Verify data after copy
        unsigned char* copiedData = static_cast<unsigned char*>(data);
        std::cout << "  After staging copy - First pixel RGBA: " << (int)copiedData[0] << ", " << (int)copiedData[1] << ", " << (int)copiedData[2] << ", " << (int)copiedData[3] << std::endl;
        
        vmaUnmapMemory(allocator, stagingAllocation);
        
        stbi_image_free(pixels);
        
        Texture texture;
        texture.width = texWidth;
        texture.height = texHeight;
        texture.mipLevels = mipLevels;
        
        createImage(texWidth, texHeight, mipLevels, textureFormat, vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                   texture.image, texture.allocation);
        
        transitionImageLayout(texture.image, textureFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
        copyBufferToImage(stagingBuffer, texture.image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        
        generateMipmaps(texture.image, textureFormat, texWidth, texHeight, mipLevels);
        
        texture.imageView = createImageView(texture.image, textureFormat, vk::ImageAspectFlagBits::eColor, mipLevels);
        
        // Use shared sampler for MoltenVK compatibility
        texture.sampler = sharedTextureSampler;
        
        return texture;
    }
    
    void generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
        // Check if image format supports linear blitting
        auto formatProperties = physicalDevice.getFormatProperties(imageFormat);
        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            std::cout << "Warning: Linear blitting not supported for format " << vk::to_string(imageFormat) 
                      << ", skipping mipmap generation" << std::endl;
            // Transition to shader read only layout for level 0
            transitionImageLayout(image, imageFormat, vk::ImageLayout::eTransferDstOptimal, 
                                vk::ImageLayout::eShaderReadOnlyOptimal, 1);
            return;
        }
        auto commandBuffer = beginSingleTimeCommands();
        
        auto barrier = vk::ImageMemoryBarrier();
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;
        
        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
            
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
                                        0, nullptr, 0, nullptr, 1, &barrier);
            
            auto blit = vk::ImageBlit();
            blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.srcOffsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
            blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            
            commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                                  image, vk::ImageLayout::eTransferDstOptimal,
                                  1, &blit, vk::Filter::eLinear);
            
            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
                                        0, nullptr, 0, nullptr, 1, &barrier);
            
            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }
        
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
                                    0, nullptr, 0, nullptr, 1, &barrier);
        
        endSingleTimeCommands(commandBuffer);
    }
    
    bool createDefaultTextures() {
        // Create 1x1 default textures
        uint8_t whitePixel[4] = {255, 255, 255, 255};
        uint8_t normalPixel[4] = {128, 128, 255, 255}; // Default normal (0, 0, 1) in tangent space
        uint8_t metallicRoughnessPixel[4] = {0, 128, 0, 255}; // No metallic, 0.5 roughness
        
        // Handle channel swizzling for BGR formats
        if (needsChannelSwizzle(textureFormat)) {
            std::cout << "Swizzling default texture channels for BGR format" << std::endl;
            std::swap(whitePixel[0], whitePixel[2]);
            std::swap(normalPixel[0], normalPixel[2]);
            std::swap(metallicRoughnessPixel[0], metallicRoughnessPixel[2]);
        }
        
        defaultAlbedoTexture = createDefaultTexture(whitePixel);
        defaultNormalTexture = createDefaultTexture(normalPixel);
        defaultMetallicRoughnessTexture = createDefaultTexture(metallicRoughnessPixel);
        
        return true;
    }
    
    Texture createDefaultTexture(uint8_t* pixelData) {
        vk::DeviceSize imageSize = 4; // 1x1 RGBA
        
        // Create staging buffer
        vk::Buffer stagingBuffer;
        VmaAllocation stagingAllocation;
        
        auto bufferInfo = vk::BufferCreateInfo();
        bufferInfo.size = imageSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        
        vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo,
                       reinterpret_cast<VkBuffer*>(&stagingBuffer), &stagingAllocation, nullptr);
        
        void* data;
        vmaMapMemory(allocator, stagingAllocation, &data);
        memcpy(data, pixelData, static_cast<size_t>(imageSize));
        vmaUnmapMemory(allocator, stagingAllocation);
        
        Texture texture;
        texture.width = 1;
        texture.height = 1;
        texture.mipLevels = 1;
        
        createImage(1, 1, 1, textureFormat, vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                   texture.image, texture.allocation);
        
        transitionImageLayout(texture.image, textureFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
        copyBufferToImage(stagingBuffer, texture.image, 1, 1);
        transitionImageLayout(texture.image, textureFormat, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);
        
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        
        texture.imageView = createImageView(texture.image, textureFormat, vk::ImageAspectFlagBits::eColor, 1);
        
        // Use shared default sampler for MoltenVK compatibility
        texture.sampler = sharedDefaultSampler;
        
        return texture;
    }
    
    void updateMaterialDescriptors(uint32_t frameIndex, uint32_t materialIndex) {
        if (materialIndex >= loadedModel.materials.size()) {
            return; // Use default material/textures
        }
        
        const Material& material = loadedModel.materials[materialIndex];
        
        std::cout << "Updating material " << materialIndex << " descriptors:" << std::endl;
        std::cout << "  Albedo texture index: " << material.albedoTextureIndex << " (loaded textures: " << loadedTextures.size() << ")" << std::endl;
        
        // Determine which textures to use
        const Texture* albedoTex = (material.albedoTextureIndex >= 0 && material.albedoTextureIndex < loadedTextures.size()) 
            ? &loadedTextures[material.albedoTextureIndex] : &defaultAlbedoTexture;
        const Texture* normalTex = (material.normalTextureIndex >= 0 && material.normalTextureIndex < loadedTextures.size()) 
            ? &loadedTextures[material.normalTextureIndex] : &defaultNormalTexture;
        const Texture* metallicRoughnessTex = (material.metallicRoughnessTextureIndex >= 0 && material.metallicRoughnessTextureIndex < loadedTextures.size()) 
            ? &loadedTextures[material.metallicRoughnessTextureIndex] : &defaultMetallicRoughnessTexture;
        
        std::cout << "  Using textures: albedo=" << (albedoTex == &defaultAlbedoTexture ? "default" : "loaded") 
                  << ", normal=" << (normalTex == &defaultNormalTexture ? "default" : "loaded") 
                  << ", metallic=" << (metallicRoughnessTex == &defaultMetallicRoughnessTexture ? "default" : "loaded") << std::endl;
        
        // Validate texture objects
        if (!albedoTex->imageView || !albedoTex->sampler) {
            std::cerr << "ERROR: Invalid albedo texture!" << std::endl;
        }
        if (!normalTex->imageView || !normalTex->sampler) {
            std::cerr << "ERROR: Invalid normal texture!" << std::endl;
        }
        if (!metallicRoughnessTex->imageView || !metallicRoughnessTex->sampler) {
            std::cerr << "ERROR: Invalid metallic/roughness texture!" << std::endl;
        }
        
        // Update texture descriptors
        std::array<vk::WriteDescriptorSet, 3> textureWrites{};
        
        vk::DescriptorImageInfo albedoImageInfo{};
        albedoImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        albedoImageInfo.imageView = albedoTex->imageView;
        albedoImageInfo.sampler = albedoTex->sampler;
        
        vk::DescriptorImageInfo normalImageInfo{};
        normalImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        normalImageInfo.imageView = normalTex->imageView;
        normalImageInfo.sampler = normalTex->sampler;
        
        vk::DescriptorImageInfo metallicRoughnessImageInfo{};
        metallicRoughnessImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        metallicRoughnessImageInfo.imageView = metallicRoughnessTex->imageView;
        metallicRoughnessImageInfo.sampler = metallicRoughnessTex->sampler;
        
        // Albedo texture
        textureWrites[0].dstSet = descriptorSets[frameIndex];
        textureWrites[0].dstBinding = 2;
        textureWrites[0].dstArrayElement = 0;
        textureWrites[0].descriptorCount = 1;
        textureWrites[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        textureWrites[0].pImageInfo = &albedoImageInfo;
        
        // Normal texture
        textureWrites[1].dstSet = descriptorSets[frameIndex];
        textureWrites[1].dstBinding = 3;
        textureWrites[1].dstArrayElement = 0;
        textureWrites[1].descriptorCount = 1;
        textureWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        textureWrites[1].pImageInfo = &normalImageInfo;
        
        // Metallic/Roughness texture
        textureWrites[2].dstSet = descriptorSets[frameIndex];
        textureWrites[2].dstBinding = 4;
        textureWrites[2].dstArrayElement = 0;
        textureWrites[2].descriptorCount = 1;
        textureWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        textureWrites[2].pImageInfo = &metallicRoughnessImageInfo;
        
        device.updateDescriptorSets(static_cast<uint32_t>(textureWrites.size()), textureWrites.data(), 0, nullptr);
    }
    
    PBRMaterial getMaterialProperties(uint32_t materialIndex) {
        if (materialIndex >= loadedModel.materials.size()) {
            // Return default material
            PBRMaterial defaultMat{};
            defaultMat.albedo = glm::vec3(1.0f);
            defaultMat.metallic = 0.0f;
            defaultMat.roughness = 0.5f;
            defaultMat.ao = 1.0f;
            defaultMat.hasAlbedoTexture = 0;
            defaultMat.hasNormalTexture = 0;
            defaultMat.hasMetallicRoughnessTexture = 0;
            defaultMat.hasAOTexture = 0;
            return defaultMat;
        }
        return loadedModel.materials[materialIndex].properties;
    }
    
    Model loadModel(const std::string& path) {
        // Resolve to absolute path to ensure texture loading works correctly
        std::string absolutePath = FileUtils::resolveRelativePath(path);
        std::cout << "Loading model from absolute path: " << absolutePath << std::endl;
        
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, 
            aiProcess_Triangulate | 
            aiProcess_FlipUVs | 
            aiProcess_CalcTangentSpace |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_OptimizeMeshes);

        if (!scene) {
            scene = importer.ReadFile(absolutePath, 
            aiProcess_Triangulate | 
            aiProcess_FlipUVs | 
            aiProcess_CalcTangentSpace |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_OptimizeMeshes);
        }
        
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            throw std::runtime_error("Failed to load model: " + absolutePath + " - " + importer.GetErrorString());
        }
        
        Model model;
        model.directory = FileUtils::getDirectory(path);
        
        // Load materials
        loadMaterials(scene, model);
        
        // Process nodes recursively with 10% scale
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
        processNode(scene->mRootNode, scene, model, scaleMatrix);
        
        // Create GPU buffers for all meshes
        for (auto& mesh : model.meshes) {
            createMeshBuffers(mesh);
        }
        
        return model;
    }
    
    void loadMaterials(const aiScene* scene, Model& model) {
        for (unsigned int i = 0; i < scene->mNumMaterials; i++) {
            const aiMaterial* mat = scene->mMaterials[i];
            Material material;
            
            // Load material properties
            aiColor3D color(0.0f, 0.0f, 0.0f);
            float value;
            
            if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
                material.properties.albedo = glm::vec3(color.r, color.g, color.b);
            }
            
            if (mat->Get(AI_MATKEY_METALLIC_FACTOR, value) == AI_SUCCESS) {
                material.properties.metallic = value;
            }
            
            if (mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, value) == AI_SUCCESS) {
                material.properties.roughness = value;
            }
            
            // Load textures - try multiple texture types as different exporters use different conventions
            loadMaterialTextures(mat, aiTextureType_DIFFUSE, model.directory, material.albedoTextureIndex);
            if (material.albedoTextureIndex < 0) {
                loadMaterialTextures(mat, aiTextureType_BASE_COLOR, model.directory, material.albedoTextureIndex);
            }
            
            loadMaterialTextures(mat, aiTextureType_NORMALS, model.directory, material.normalTextureIndex);
            if (material.normalTextureIndex < 0) {
                loadMaterialTextures(mat, aiTextureType_NORMAL_CAMERA, model.directory, material.normalTextureIndex);
            }
            
            // Try multiple sources for metallic/roughness
            loadMaterialTextures(mat, aiTextureType_METALNESS, model.directory, material.metallicRoughnessTextureIndex);
            if (material.metallicRoughnessTextureIndex < 0) {
                loadMaterialTextures(mat, aiTextureType_DIFFUSE_ROUGHNESS, model.directory, material.metallicRoughnessTextureIndex);
            }
            if (material.metallicRoughnessTextureIndex < 0) {
                loadMaterialTextures(mat, aiTextureType_UNKNOWN, model.directory, material.metallicRoughnessTextureIndex);
            }
            
            // Set material flags
            material.properties.hasAlbedoTexture = (material.albedoTextureIndex >= 0) ? 1 : 0;
            material.properties.hasNormalTexture = (material.normalTextureIndex >= 0) ? 1 : 0;
            material.properties.hasMetallicRoughnessTexture = (material.metallicRoughnessTextureIndex >= 0) ? 1 : 0;
            
            aiString name;
            if (mat->Get(AI_MATKEY_NAME, name) == AI_SUCCESS) {
                material.name = name.C_Str();
            }
            
            std::cout << "Material " << i << " (" << material.name << "):" << std::endl;
            std::cout << "  Albedo: " << material.properties.albedo.x << ", " << material.properties.albedo.y << ", " << material.properties.albedo.z << std::endl;
            std::cout << "  Metallic: " << material.properties.metallic << ", Roughness: " << material.properties.roughness << std::endl;
            std::cout << "  Textures: Albedo=" << material.albedoTextureIndex << ", Normal=" << material.normalTextureIndex << ", MetallicRoughness=" << material.metallicRoughnessTextureIndex << std::endl;
            std::cout << "  Flags: hasAlbedo=" << material.properties.hasAlbedoTexture << ", hasNormal=" << material.properties.hasNormalTexture << ", hasMetallicRoughness=" << material.properties.hasMetallicRoughnessTexture << std::endl;
            
            model.materials.push_back(material);
        }
    }
    
    void loadMaterialTextures(const aiMaterial* mat, aiTextureType type, const std::string& directory, int& textureIndex) {
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
            aiString str;
            mat->GetTexture(type, i, &str);
            std::string texturePath = FileUtils::joinPaths(directory, str.C_Str());
            
            // Check if texture already loaded
            bool skip = false;
            for (size_t j = 0; j < loadedTextures.size(); j++) {
                // We should store texture paths to avoid duplicates
                // For now, just load each texture
            }
            
            if (!skip) {
                // Check if file exists before attempting to load
                if (!FileUtils::fileExists(texturePath)) {
                    std::cerr << "Warning: Texture file not found: " << texturePath << std::endl;
                    textureIndex = -1;
                } else {
                    try {
                        std::cout << "Loading texture: " << texturePath << std::endl;
                        Texture texture = loadTexture(texturePath);
                        loadedTextures.push_back(texture);
                        textureIndex = static_cast<int>(loadedTextures.size() - 1);
                        std::cout << "Successfully loaded texture " << texturePath << " at index " << textureIndex << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to load texture " << texturePath << ": " << e.what() << std::endl;
                        textureIndex = -1;
                    }
                }
            }
        }
    }
    
    void processNode(aiNode* node, const aiScene* scene, Model& model, const glm::mat4& parentTransform) {
        // Convert aiMatrix4x4 to glm::mat4
        aiMatrix4x4 aiTrans = node->mTransformation;
        glm::mat4 nodeTransform = glm::mat4(
            aiTrans.a1, aiTrans.b1, aiTrans.c1, aiTrans.d1,
            aiTrans.a2, aiTrans.b2, aiTrans.c2, aiTrans.d2,
            aiTrans.a3, aiTrans.b3, aiTrans.c3, aiTrans.d3,
            aiTrans.a4, aiTrans.b4, aiTrans.c4, aiTrans.d4
        );
        
        glm::mat4 transform = parentTransform * nodeTransform;
        
        // Process all the node's meshes
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            model.meshes.push_back(processMesh(mesh, scene, transform));
        }
        
        // Process children
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene, model, transform);
        }
    }
    
    Mesh processMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& transform) {
        Mesh result;
        result.transform = transform;
        result.materialIndex = mesh->mMaterialIndex;
        
        // Process vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;
            
            // Position - apply transform matrix
            glm::vec4 worldPos = transform * glm::vec4(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z, 1.0f);
            vertex.pos.x = worldPos.x;
            vertex.pos.y = worldPos.y;
            vertex.pos.z = worldPos.z;
            
            // Normal - transform with inverse transpose
            if (mesh->HasNormals()) {
                glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));
                glm::vec3 worldNormal = normalMatrix * glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
                vertex.normal = glm::normalize(worldNormal);
            }
            
            // Texture coordinates
            if (mesh->mTextureCoords[0]) {
                vertex.texCoord.x = mesh->mTextureCoords[0][i].x;
                vertex.texCoord.y = mesh->mTextureCoords[0][i].y;
            } else {
                vertex.texCoord = glm::vec2(0.0f, 0.0f);
            }
            
            // Tangent - transform like normal and store handedness in w component
            if (mesh->HasTangentsAndBitangents()) {
                glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));
                glm::vec3 worldTangent = normalMatrix * glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
                glm::vec3 worldBitangent = normalMatrix * glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);
                
                // Calculate handedness for proper bitangent reconstruction
                glm::vec3 worldNormal = normalMatrix * glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
                float handedness = glm::dot(glm::cross(worldNormal, worldTangent), worldBitangent) < 0.0f ? -1.0f : 1.0f;
                
                vertex.tangent = glm::normalize(worldTangent);
                // We'll store handedness in a separate way since tangent is vec3
            }
            
            result.vertices.push_back(vertex);
        }
        
        // Process indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                result.indices.push_back(face.mIndices[j]);
            }
        }
        
        return result;
    }
    
    void createMeshBuffers(Mesh& mesh) {
        vk::DeviceSize vertexBufferSize = sizeof(mesh.vertices[0]) * mesh.vertices.size();
        vk::DeviceSize indexBufferSize = sizeof(mesh.indices[0]) * mesh.indices.size();
        
        // Create vertex buffer
        auto bufferInfo = vk::BufferCreateInfo();
        bufferInfo.size = vertexBufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        
        if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                           reinterpret_cast<VkBuffer*>(&mesh.vertexBuffer), &mesh.vertexBufferAllocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create vertex buffer for mesh");
        }
        
        void* data;
        vmaMapMemory(allocator, mesh.vertexBufferAllocation, &data);
        memcpy(data, mesh.vertices.data(), static_cast<size_t>(vertexBufferSize));
        vmaUnmapMemory(allocator, mesh.vertexBufferAllocation);
        
        // Create index buffer
        bufferInfo.size = indexBufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
        
        if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, 
                           reinterpret_cast<VkBuffer*>(&mesh.indexBuffer), &mesh.indexBufferAllocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create index buffer for mesh");
        }
        
        vmaMapMemory(allocator, mesh.indexBufferAllocation, &data);
        memcpy(data, mesh.indices.data(), static_cast<size_t>(indexBufferSize));
        vmaUnmapMemory(allocator, mesh.indexBufferAllocation);
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
        
        // Create pipeline layout with push constants for material data
        vk::PushConstantRange pushConstantRange;
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PBRMaterial);
        
        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        
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
        
        auto depthStencil = vk::PipelineDepthStencilStateCreateInfo();
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = vk::CompareOp::eLess;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;
        
        auto pipelineInfo = vk::GraphicsPipelineCreateInfo();
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
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
            std::array<vk::ImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
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
        std::array<vk::DescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2); // Camera and Light only
        poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 3);
        
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
            
            auto albedoImageInfo = vk::DescriptorImageInfo();
            albedoImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            albedoImageInfo.imageView = defaultAlbedoTexture.imageView;
            albedoImageInfo.sampler = defaultAlbedoTexture.sampler;
            
            auto normalImageInfo = vk::DescriptorImageInfo();
            normalImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            normalImageInfo.imageView = defaultNormalTexture.imageView;
            normalImageInfo.sampler = defaultNormalTexture.sampler;
            
            auto metallicRoughnessImageInfo = vk::DescriptorImageInfo();
            metallicRoughnessImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            metallicRoughnessImageInfo.imageView = defaultMetallicRoughnessTexture.imageView;
            metallicRoughnessImageInfo.sampler = defaultMetallicRoughnessTexture.sampler;
            
            std::array<vk::WriteDescriptorSet, 5> descriptorWrites{};
            
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
            descriptorWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[2].pImageInfo = &albedoImageInfo;
            
            descriptorWrites[3].dstSet = descriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[3].pImageInfo = &normalImageInfo;
            
            descriptorWrites[4].dstSet = descriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[4].pImageInfo = &metallicRoughnessImageInfo;
            
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
        cameraUbo.view = camera.getViewMatrix();
        cameraUbo.invView = glm::inverse(cameraUbo.view);
        cameraUbo.proj = glm::perspective(glm::radians(camera.zoom), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10000.0f);
        cameraUbo.proj[1][1] *= -1; // Flip Y for Vulkan
        cameraUbo.viewPos = camera.position;
        
        void* data;
        vmaMapMemory(allocator, cameraUniformBufferAllocations[currentImage], &data);
        memcpy(data, &cameraUbo, sizeof(cameraUbo));
        vmaUnmapMemory(allocator, cameraUniformBufferAllocations[currentImage]);
        
        // Update lights - transform to view space
        LightUBO lightUbo{};
        glm::vec4 worldLightPos[4] = {
            glm::vec4(sinf(time)*5.0f, 0.0f, cosf(time)*5.0f, 1.0f),
            glm::vec4(sinf(2.0944f+time)*5.0f, 0.0f, cosf(2.0944f+time)*5.0f, 1.0f),
            glm::vec4(sinf(4.1888f+time)*5.0f, 0.0f, cosf(4.1888f+time)*5.0f, 1.0f),
            glm::vec4(camera.position.x, camera.position.y, camera.position.z, 1.0f)
        };
        
        // Transform light positions to view space
        for(int i = 0; i < 4; ++i) {
            glm::vec4 viewSpacePos = cameraUbo.view * worldLightPos[i];
            lightUbo.lightPositions[i] = glm::vec4((glm::vec3)viewSpacePos, 0.0f);
        }
        
        lightUbo.lightColors[0] = glm::vec4(3.0f, 0.0f, 0.0f,0.0f);   // CYAN
        lightUbo.lightColors[1] = glm::vec4(0.0f, 3.0f, 0.0f,0.0f);   // MAGENTA
        lightUbo.lightColors[2] = glm::vec4(0.0f, 0.0f, 3.0f,0.0f);   // YELLOW
        lightUbo.lightColors[3] = glm::vec4(0.0f, 0.0f, 0.0f,0.0f); // WHITE
        
        vmaMapMemory(allocator, lightUniformBufferAllocations[currentImage], &data);
        memcpy(data, &lightUbo, sizeof(lightUbo));
        vmaUnmapMemory(allocator, lightUniformBufferAllocations[currentImage]);
        
        // Update material
        PBRMaterial material{};
        material.albedo = glm::vec3(1.0f, 1.0f, 1.0f);
        material.metallic = 1.0f;
        material.roughness = 0.1f;
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
        
        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color = vk::ClearColorValue{std::array<float, 4>{{0.0f, 0.0f, 0.0f, 1.0f}}};
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        
        auto renderPassInfo = vk::RenderPassBeginInfo();
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        
        commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
        
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        
        if (useModel && !loadedModel.meshes.empty()) {
            // Render all meshes in the loaded model
            for (size_t meshIdx = 0; meshIdx < loadedModel.meshes.size(); meshIdx++) {
                const auto& mesh = loadedModel.meshes[meshIdx];
                
                // Update descriptor set for this mesh's material BEFORE binding
                updateMaterialDescriptors(currentFrame, mesh.materialIndex);
                
                // Push material constants
                PBRMaterial materialProps = getMaterialProperties(mesh.materialIndex);
                commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PBRMaterial), &materialProps);
                
                std::array<vk::Buffer, 1> vertexBuffers = {mesh.vertexBuffer};
                std::array<vk::DeviceSize, 1> offsets = {0};
                commandBuffer.bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());
                commandBuffer.bindIndexBuffer(mesh.indexBuffer, 0, vk::IndexType::eUint32);
                
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
                
                commandBuffer.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
            }
        } else {
            // Render the default sphere
            std::array<vk::Buffer, 1> vertexBuffers = {vertexBuffer};
            std::array<vk::DeviceSize, 1> offsets = {0};
            commandBuffer.bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());
            commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
            
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            
            commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        }
        
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
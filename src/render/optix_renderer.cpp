#include "render/optix_renderer.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <sutil/vec_math.h>

#include "convert/gsplat_bounds.h"
#include "convert/voxel_grid.h"
#include "io/gsplat_asset.h"
#include "render/gl_ext.h"
#include "render/kernels/pathtracer_params.h"
#include "render/optix_check.h"
#include "render_config.h"

namespace italy {
namespace {

float3 toFloat3(const glm::vec3 &v) { return make_float3(v.x, v.y, v.z); }

template <typename T> struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};
using RayGenRecord = SbtRecord<RayGenData>;
using MissRecord = SbtRecord<MissData>;
using HitGroupRecord = SbtRecord<HitGroupData>;

std::vector<char> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error(std::string("cannot open ") + path);
  const auto size = f.tellg();
  std::vector<char> data(static_cast<size_t>(size));
  f.seekg(0);
  f.read(data.data(), size);
  return data;
}

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *) {
  std::fprintf(stderr, "[optix][%u][%s] %s\n", level, tag, message);
}

enum class GeometryKind { Triangle, Sphere, SolidSphere, Voxel, Sdf, Gsplat, Nvdb };

struct SceneObject {
  OptixTraversableHandle gas = 0;
  CUdeviceptr gasBuffer = 0;
  HitGroupData material{};
  GeometryKind kind = GeometryKind::Triangle;
};

} // namespace

struct OptixRenderer::Impl {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixModule sphereModule = nullptr;
  OptixPipelineCompileOptions pipelineCompileOptions{};

  OptixProgramGroup raygenPG = nullptr;
  OptixProgramGroup missRadiancePG = nullptr;
  OptixProgramGroup missOcclusionPG = nullptr;
  OptixProgramGroup hitTrianglePG = nullptr;
  OptixProgramGroup hitSpherePG = nullptr;
  OptixProgramGroup hitSolidSpherePG = nullptr;
  OptixProgramGroup hitVoxelPG = nullptr;
  OptixProgramGroup hitSdfPG = nullptr;
  OptixProgramGroup hitGsplatPG = nullptr;
  OptixProgramGroup hitNvdbPG = nullptr;

  OptixProgramGroup lightSubpathRaygenPG = nullptr;
  OptixProgramGroup lightSubpathMissPG = nullptr;
  OptixProgramGroup lightSubpathHitTrianglePG = nullptr;
  OptixProgramGroup lightSubpathHitSpherePG = nullptr;
  OptixProgramGroup lightSubpathHitSolidSpherePG = nullptr;
  OptixProgramGroup lightSubpathHitSdfPG = nullptr;

  OptixProgramGroup mergeHitPG = nullptr;
  OptixProgramGroup mergeMissPG = nullptr;

  OptixProgramGroup tonemapRaygenPG = nullptr;
  OptixShaderBindingTable tonemapSbt{};

  OptixProgramGroup reservoirBuildRaygenPG = nullptr;
  OptixShaderBindingTable reservoirBuildSbt{};

  OptixProgramGroup causticAabbRaygenPG = nullptr;
  OptixShaderBindingTable causticAabbSbt{};
  OptixDenoiser denoiser = nullptr;
  CUdeviceptr denoiserStateBuffer = 0;
  size_t denoiserStateSize = 0;
  CUdeviceptr denoiserScratchBuffer = 0;
  size_t denoiserScratchSize = 0;
  CUdeviceptr denoisedBuffer[2] = {0, 0};
  CUdeviceptr denoiserInternalGuideBuffer[2] = {0, 0};
  size_t denoiserInternalGuideLayerPixelSizeInBytes = 0;
  bool denoiserTemporalActive = false;
  bool denoiserBuilt = false;

  OptixPipeline pipeline = nullptr;

  std::vector<SceneObject> objects;
  OptixTraversableHandle iasHandle = 0;
  CUdeviceptr iasBuffer = 0;

  OptixShaderBindingTable sbt{};

  CUstream stream = nullptr;
  CUdeviceptr accumBuffer[2] = {0, 0};
  CUdeviceptr accumAlbedoBuffer[2] = {0, 0};
  CUdeviceptr accumNormalBuffer[2] = {0, 0};
  CUdeviceptr motionVectorBuffer = 0;
  CUdeviceptr denoiserFlowBuffer = 0;
  int nextWriteIdx = 0;
  bool hasPrevCamera = false;
  float3 prevEyeUsed{};
  float3 prevUUsed{};
  float3 prevVUsed{};
  float3 prevWUsed{};
  CUdeviceptr reservoirBuffer[2] = {0, 0};
  CUdeviceptr paramsBuffer = 0;

  unsigned int pbo = 0;
  cudaGraphicsResource *cudaPbo = nullptr;

  QuadLight light{};
  glm::vec3 boundsCenter{0.0f};
  float boundsRadius = 3.0f;

  CUdeviceptr meshNormals = 0;
  CUdeviceptr meshUvs = 0;
  CUdeviceptr meshTangents = 0;
  CUdeviceptr meshTriangleMaterial = 0;
  CUdeviceptr meshMaterials = 0;
  std::vector<cudaArray_t> textureArrays;
  std::vector<cudaTextureObject_t> textureObjects;

  CUdeviceptr voxelAabbBuffer = 0;
  CUdeviceptr voxelColorBuffer = 0;

  std::vector<CUdeviceptr> sdfDistanceBuffers;
  std::vector<CUdeviceptr> sdfBaseColorBuffers;
  std::vector<CUdeviceptr> sdfMetallicBuffers;
  std::vector<CUdeviceptr> sdfRoughnessBuffers;
  std::vector<CUdeviceptr> sdfBranchBuffers;
  std::vector<CUdeviceptr> sdfPaletteBuffers;

  CUdeviceptr splatPositionBuffer = 0;
  CUdeviceptr splatScaleBuffer = 0;
  CUdeviceptr splatRotationBuffer = 0;
  CUdeviceptr splatOpacityBuffer = 0;
  CUdeviceptr splatColorBuffer = 0;

  CUdeviceptr nvdbGridBuffer = 0;
  bool hasVolume = false;
  float3 nvdbBoundsMin{};
  float3 nvdbBoundsMax{};
  float3 nvdbSigmaT{};
  float3 nvdbScatterAlbedo{};
  float nvdbG = 0.0f;
  float nvdbDensityScale = 1.0f;
  float nvdbMajorant = 0.0f;

  bool hasEnvironment = false;
  bool wantGroundPlane = true;
  float groundOffset = 0.0f;
  cudaArray_t envArray = nullptr;
  cudaTextureObject_t envTexObj = 0;
  CUdeviceptr envMarginalCdfBuffer = 0;
  CUdeviceptr envConditionalCdfBuffer = 0;
  int envWidth = 0;
  int envHeight = 0;

  bool enableLightSubpaths = false;
  static constexpr unsigned int kLightSubpathBatchSize = 262144;
  static constexpr unsigned int kLightVertexCapacity = 524288;
  OptixShaderBindingTable lightSubpathSbt{};
  CUdeviceptr lightVertexBuffer = 0;
  CUdeviceptr lightVertexCounterBuffer = 0;
  CUdeviceptr causticAabbBuffer = 0;
  unsigned int lightSubpathPassIndex = 0;
  unsigned int totalLightPathsEmitted = 0;

  float mergeRadius = -1.0f;

  CUdeviceptr mergeGasOutputBuffer = 0;
  CUdeviceptr mergeGasTempBuffer = 0;
  size_t mergeGasOutputCapacityBytes = 0;
  size_t mergeGasTempCapacityBytes = 0;
  OptixTraversableHandle mergeGasHandle = 0;
  unsigned int mergeHitSbtOffset = 0;

  static constexpr float kPpmAlpha = 0.7f;
  static constexpr float kInitialMergeRadiusFraction = 0.02f;

  ~Impl() { destroy(); }

  void destroy() {
    for (cudaTextureObject_t obj : textureObjects)
      if (obj)
        cudaDestroyTextureObject(obj);
    for (cudaArray_t arr : textureArrays)
      if (arr)
        cudaFreeArray(arr);
    if (meshNormals)
      cudaFree(reinterpret_cast<void *>(meshNormals));
    if (meshUvs)
      cudaFree(reinterpret_cast<void *>(meshUvs));
    if (meshTangents)
      cudaFree(reinterpret_cast<void *>(meshTangents));
    if (meshTriangleMaterial)
      cudaFree(reinterpret_cast<void *>(meshTriangleMaterial));
    if (meshMaterials)
      cudaFree(reinterpret_cast<void *>(meshMaterials));
    if (voxelAabbBuffer)
      cudaFree(reinterpret_cast<void *>(voxelAabbBuffer));
    if (voxelColorBuffer)
      cudaFree(reinterpret_cast<void *>(voxelColorBuffer));
    for (const std::vector<CUdeviceptr> *bufs : {&sdfDistanceBuffers, &sdfBaseColorBuffers, &sdfMetallicBuffers,
                                                  &sdfRoughnessBuffers, &sdfBranchBuffers, &sdfPaletteBuffers})
      for (CUdeviceptr p : *bufs)
        if (p)
          cudaFree(reinterpret_cast<void *>(p));
    if (splatPositionBuffer)
      cudaFree(reinterpret_cast<void *>(splatPositionBuffer));
    if (splatScaleBuffer)
      cudaFree(reinterpret_cast<void *>(splatScaleBuffer));
    if (splatRotationBuffer)
      cudaFree(reinterpret_cast<void *>(splatRotationBuffer));
    if (splatOpacityBuffer)
      cudaFree(reinterpret_cast<void *>(splatOpacityBuffer));
    if (splatColorBuffer)
      cudaFree(reinterpret_cast<void *>(splatColorBuffer));
    if (nvdbGridBuffer)
      cudaFree(reinterpret_cast<void *>(nvdbGridBuffer));
    if (envTexObj)
      cudaDestroyTextureObject(envTexObj);
    if (envArray)
      cudaFreeArray(envArray);
    if (envMarginalCdfBuffer)
      cudaFree(reinterpret_cast<void *>(envMarginalCdfBuffer));
    if (envConditionalCdfBuffer)
      cudaFree(reinterpret_cast<void *>(envConditionalCdfBuffer));
    if (lightVertexBuffer)
      cudaFree(reinterpret_cast<void *>(lightVertexBuffer));
    if (lightVertexCounterBuffer)
      cudaFree(reinterpret_cast<void *>(lightVertexCounterBuffer));
    if (causticAabbBuffer)
      cudaFree(reinterpret_cast<void *>(causticAabbBuffer));
    if (mergeGasOutputBuffer)
      cudaFree(reinterpret_cast<void *>(mergeGasOutputBuffer));
    if (mergeGasTempBuffer)
      cudaFree(reinterpret_cast<void *>(mergeGasTempBuffer));
    if (lightSubpathSbt.raygenRecord)
      cudaFree(reinterpret_cast<void *>(lightSubpathSbt.raygenRecord));
    if (lightSubpathSbt.missRecordBase)
      cudaFree(reinterpret_cast<void *>(lightSubpathSbt.missRecordBase));
    if (lightSubpathSbt.hitgroupRecordBase)
      cudaFree(reinterpret_cast<void *>(lightSubpathSbt.hitgroupRecordBase));
    if (tonemapSbt.raygenRecord)
      cudaFree(reinterpret_cast<void *>(tonemapSbt.raygenRecord));
    if (tonemapSbt.missRecordBase)
      cudaFree(reinterpret_cast<void *>(tonemapSbt.missRecordBase));
    if (reservoirBuildSbt.raygenRecord)
      cudaFree(reinterpret_cast<void *>(reservoirBuildSbt.raygenRecord));
    if (causticAabbSbt.raygenRecord)
      cudaFree(reinterpret_cast<void *>(causticAabbSbt.raygenRecord));
    if (causticAabbSbt.missRecordBase)
      cudaFree(reinterpret_cast<void *>(causticAabbSbt.missRecordBase));
    freeDenoiser();
    if (paramsBuffer)
      cudaFree(reinterpret_cast<void *>(paramsBuffer));
    for (int i = 0; i < 2; ++i) {
      if (accumBuffer[i])
        cudaFree(reinterpret_cast<void *>(accumBuffer[i]));
      if (accumAlbedoBuffer[i])
        cudaFree(reinterpret_cast<void *>(accumAlbedoBuffer[i]));
      if (accumNormalBuffer[i])
        cudaFree(reinterpret_cast<void *>(accumNormalBuffer[i]));
    }
    if (motionVectorBuffer)
      cudaFree(reinterpret_cast<void *>(motionVectorBuffer));
    if (denoiserFlowBuffer)
      cudaFree(reinterpret_cast<void *>(denoiserFlowBuffer));
    for (int i = 0; i < 2; ++i) {
      if (reservoirBuffer[i])
        cudaFree(reinterpret_cast<void *>(reservoirBuffer[i]));
    }
    if (sbt.raygenRecord)
      cudaFree(reinterpret_cast<void *>(sbt.raygenRecord));
    if (sbt.missRecordBase)
      cudaFree(reinterpret_cast<void *>(sbt.missRecordBase));
    if (sbt.hitgroupRecordBase)
      cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase));
    if (iasBuffer)
      cudaFree(reinterpret_cast<void *>(iasBuffer));
    for (auto &obj : objects)
      if (obj.gasBuffer)
        cudaFree(reinterpret_cast<void *>(obj.gasBuffer));
    if (cudaPbo)
      cudaGraphicsUnregisterResource(cudaPbo);
    if (pbo)
      GLBufferFns::get().glDeleteBuffers(1, &pbo);
    if (pipeline)
      optixPipelineDestroy(pipeline);
    if (causticAabbRaygenPG)
      optixProgramGroupDestroy(causticAabbRaygenPG);
    if (tonemapRaygenPG)
      optixProgramGroupDestroy(tonemapRaygenPG);
    if (mergeMissPG)
      optixProgramGroupDestroy(mergeMissPG);
    if (mergeHitPG)
      optixProgramGroupDestroy(mergeHitPG);
    if (lightSubpathHitSdfPG)
      optixProgramGroupDestroy(lightSubpathHitSdfPG);
    if (lightSubpathHitSolidSpherePG)
      optixProgramGroupDestroy(lightSubpathHitSolidSpherePG);
    if (lightSubpathHitSpherePG)
      optixProgramGroupDestroy(lightSubpathHitSpherePG);
    if (lightSubpathHitTrianglePG)
      optixProgramGroupDestroy(lightSubpathHitTrianglePG);
    if (lightSubpathMissPG)
      optixProgramGroupDestroy(lightSubpathMissPG);
    if (lightSubpathRaygenPG)
      optixProgramGroupDestroy(lightSubpathRaygenPG);
    if (hitNvdbPG)
      optixProgramGroupDestroy(hitNvdbPG);
    if (hitGsplatPG)
      optixProgramGroupDestroy(hitGsplatPG);
    if (hitSdfPG)
      optixProgramGroupDestroy(hitSdfPG);
    if (hitVoxelPG)
      optixProgramGroupDestroy(hitVoxelPG);
    if (hitSolidSpherePG)
      optixProgramGroupDestroy(hitSolidSpherePG);
    if (hitSpherePG)
      optixProgramGroupDestroy(hitSpherePG);
    if (hitTrianglePG)
      optixProgramGroupDestroy(hitTrianglePG);
    if (missOcclusionPG)
      optixProgramGroupDestroy(missOcclusionPG);
    if (missRadiancePG)
      optixProgramGroupDestroy(missRadiancePG);
    if (raygenPG)
      optixProgramGroupDestroy(raygenPG);
    if (sphereModule)
      optixModuleDestroy(sphereModule);
    if (module)
      optixModuleDestroy(module);
    if (context)
      optixDeviceContextDestroy(context);
  }

  void initContext() {
    CUDA_CHECK(cudaFree(nullptr));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 3;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  void buildModule() {
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 25;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    const std::vector<char> ir = readFile(kPathTracerOptixIRPath);
    OPTIX_CHECK_LOG(optixModuleCreate(context, &moduleOptions, &pipelineCompileOptions, ir.data(), ir.size(), LOG,
                                       &LOG_SIZE, &module));

    OptixBuiltinISOptions sphereOpts{};
    sphereOpts.usesMotionBlur = false;
    sphereOpts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG(
        optixBuiltinISModuleGet(context, &moduleOptions, &pipelineCompileOptions, &sphereOpts, &sphereModule));
  }

  void buildProgramGroups() {
    OptixProgramGroupOptions pgOptions{};

    OptixProgramGroupDesc raygenDesc{};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygenDesc, 1, &pgOptions, LOG, &LOG_SIZE, &raygenPG));

    OptixProgramGroupDesc missRadDesc{};
    missRadDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missRadDesc.miss.module = module;
    missRadDesc.miss.entryFunctionName = "__miss__radiance";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &missRadDesc, 1, &pgOptions, LOG, &LOG_SIZE, &missRadiancePG));

    OptixProgramGroupDesc missOccDesc{};
    missOccDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missOccDesc.miss.module = module;
    missOccDesc.miss.entryFunctionName = "__miss__occlusion";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &missOccDesc, 1, &pgOptions, LOG, &LOG_SIZE, &missOcclusionPG));

    OptixProgramGroupDesc hitTriDesc{};
    hitTriDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitTriDesc.hitgroup.moduleCH = module;
    hitTriDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitTriDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitTrianglePG));

    OptixProgramGroupDesc hitSphereDesc{};
    hitSphereDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitSphereDesc.hitgroup.moduleCH = module;
    hitSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitSphereDesc.hitgroup.moduleIS = sphereModule;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitSphereDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitSpherePG));

    OptixProgramGroupDesc hitSolidSphereDesc{};
    hitSolidSphereDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitSolidSphereDesc.hitgroup.moduleCH = module;
    hitSolidSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitSolidSphereDesc.hitgroup.moduleIS = module;
    hitSolidSphereDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere_solid";
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(context, &hitSolidSphereDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitSolidSpherePG));

    OptixProgramGroupDesc hitVoxelDesc{};
    hitVoxelDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitVoxelDesc.hitgroup.moduleCH = module;
    hitVoxelDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitVoxelDesc.hitgroup.moduleIS = module;
    hitVoxelDesc.hitgroup.entryFunctionNameIS = "__intersection__voxel";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitVoxelDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitVoxelPG));

    OptixProgramGroupDesc hitSdfDesc{};
    hitSdfDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitSdfDesc.hitgroup.moduleCH = module;
    hitSdfDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitSdfDesc.hitgroup.moduleIS = module;
    hitSdfDesc.hitgroup.entryFunctionNameIS = "__intersection__sdf";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitSdfDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitSdfPG));

    OptixProgramGroupDesc hitGsplatDesc{};
    hitGsplatDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitGsplatDesc.hitgroup.moduleCH = module;
    hitGsplatDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitGsplatDesc.hitgroup.moduleAH = module;
    hitGsplatDesc.hitgroup.entryFunctionNameAH = "__anyhit__gsplat";
    hitGsplatDesc.hitgroup.moduleIS = module;
    hitGsplatDesc.hitgroup.entryFunctionNameIS = "__intersection__gsplat";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitGsplatDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitGsplatPG));

    OptixProgramGroupDesc hitNvdbDesc{};
    hitNvdbDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitNvdbDesc.hitgroup.moduleCH = module;
    hitNvdbDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitNvdbDesc.hitgroup.moduleIS = module;
    hitNvdbDesc.hitgroup.entryFunctionNameIS = "__intersection__nvdb";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitNvdbDesc, 1, &pgOptions, LOG, &LOG_SIZE, &hitNvdbPG));

    OptixProgramGroupDesc lightSubpathRaygenDesc{};
    lightSubpathRaygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    lightSubpathRaygenDesc.raygen.module = module;
    lightSubpathRaygenDesc.raygen.entryFunctionName = "__raygen__lightSubpath";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &lightSubpathRaygenDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &lightSubpathRaygenPG));

    OptixProgramGroupDesc lightSubpathMissDesc{};
    lightSubpathMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    lightSubpathMissDesc.miss.module = module;
    lightSubpathMissDesc.miss.entryFunctionName = "__miss__lightSubpath";
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(context, &lightSubpathMissDesc, 1, &pgOptions, LOG, &LOG_SIZE, &lightSubpathMissPG));

    OptixProgramGroupDesc lightSubpathHitTriDesc{};
    lightSubpathHitTriDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    lightSubpathHitTriDesc.hitgroup.moduleCH = module;
    lightSubpathHitTriDesc.hitgroup.entryFunctionNameCH = "__closesthit__lightSubpath";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &lightSubpathHitTriDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &lightSubpathHitTrianglePG));

    OptixProgramGroupDesc lightSubpathHitSphereDesc{};
    lightSubpathHitSphereDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    lightSubpathHitSphereDesc.hitgroup.moduleCH = module;
    lightSubpathHitSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__lightSubpath";
    lightSubpathHitSphereDesc.hitgroup.moduleIS = sphereModule;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &lightSubpathHitSphereDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &lightSubpathHitSpherePG));

    OptixProgramGroupDesc lightSubpathHitSolidSphereDesc{};
    lightSubpathHitSolidSphereDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    lightSubpathHitSolidSphereDesc.hitgroup.moduleCH = module;
    lightSubpathHitSolidSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__lightSubpath";
    lightSubpathHitSolidSphereDesc.hitgroup.moduleIS = module;
    lightSubpathHitSolidSphereDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere_solid";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &lightSubpathHitSolidSphereDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &lightSubpathHitSolidSpherePG));

    OptixProgramGroupDesc lightSubpathHitSdfDesc{};
    lightSubpathHitSdfDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    lightSubpathHitSdfDesc.hitgroup.moduleCH = module;
    lightSubpathHitSdfDesc.hitgroup.entryFunctionNameCH = "__closesthit__lightSubpath";
    lightSubpathHitSdfDesc.hitgroup.moduleIS = module;
    lightSubpathHitSdfDesc.hitgroup.entryFunctionNameIS = "__intersection__sdf";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &lightSubpathHitSdfDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &lightSubpathHitSdfPG));

    OptixProgramGroupDesc mergeHitDesc{};
    mergeHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    mergeHitDesc.hitgroup.moduleAH = module;
    mergeHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__merge";
    mergeHitDesc.hitgroup.moduleIS = module;
    mergeHitDesc.hitgroup.entryFunctionNameIS = "__intersection__causticSplat";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &mergeHitDesc, 1, &pgOptions, LOG, &LOG_SIZE, &mergeHitPG));

    OptixProgramGroupDesc mergeMissDesc{};
    mergeMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    mergeMissDesc.miss.module = module;
    mergeMissDesc.miss.entryFunctionName = "__miss__merge";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &mergeMissDesc, 1, &pgOptions, LOG, &LOG_SIZE, &mergeMissPG));

    OptixProgramGroupDesc tonemapRaygenDesc{};
    tonemapRaygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    tonemapRaygenDesc.raygen.module = module;
    tonemapRaygenDesc.raygen.entryFunctionName = "__raygen__tonemap";
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(context, &tonemapRaygenDesc, 1, &pgOptions, LOG, &LOG_SIZE, &tonemapRaygenPG));

    OptixProgramGroupDesc causticAabbRaygenDesc{};
    causticAabbRaygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    causticAabbRaygenDesc.raygen.module = module;
    causticAabbRaygenDesc.raygen.entryFunctionName = "__raygen__causticAabb";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &causticAabbRaygenDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &causticAabbRaygenPG));

    OptixProgramGroupDesc reservoirBuildRaygenDesc{};
    reservoirBuildRaygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    reservoirBuildRaygenDesc.raygen.module = module;
    reservoirBuildRaygenDesc.raygen.entryFunctionName = "__raygen__reservoirBuild";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &reservoirBuildRaygenDesc, 1, &pgOptions, LOG, &LOG_SIZE,
                                             &reservoirBuildRaygenPG));
  }

  void buildPipeline() {
    OptixProgramGroup groups[] = {raygenPG,
                                   missRadiancePG,
                                   missOcclusionPG,
                                   hitTrianglePG,
                                   hitSpherePG,
                                   hitSolidSpherePG,
                                   hitVoxelPG,
                                   hitSdfPG,
                                   hitGsplatPG,
                                   hitNvdbPG,
                                   lightSubpathRaygenPG,
                                   lightSubpathMissPG,
                                   lightSubpathHitTrianglePG,
                                   lightSubpathHitSpherePG,
                                   lightSubpathHitSolidSpherePG,
                                   lightSubpathHitSdfPG,
                                   mergeHitPG,
                                   mergeMissPG,
                                   tonemapRaygenPG,
                                   causticAabbRaygenPG,
                                   reservoirBuildRaygenPG};
    OptixPipelineLinkOptions linkOptions{};
    const uint32_t maxTraceDepth = 2;
    linkOptions.maxTraceDepth = maxTraceDepth;
    OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipelineCompileOptions, &linkOptions, groups,
                                         sizeof(groups) / sizeof(groups[0]), LOG, &LOG_SIZE, &pipeline));

    OptixStackSizes stackSizes{};
    for (auto *g : groups)
      OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stackSizes, pipeline));
    uint32_t dcFromTraversal, dcFromState, contStack;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, maxTraceDepth, 0, 0, &dcFromTraversal, &dcFromState,
                                            &contStack));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dcFromTraversal, dcFromState, contStack, 2));
  }

  OptixTraversableHandle buildAccel(const OptixBuildInput &input, CUdeviceptr &outBuffer,
                                     unsigned int buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE) {
    OptixAccelBuildOptions accelOptions{};
    accelOptions.buildFlags = buildFlags;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &input, 1, &sizes));

    CUdeviceptr tempBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&tempBuffer), sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&outBuffer), sizes.outputSizeInBytes));

    OptixTraversableHandle handle = 0;
    OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &input, 1, tempBuffer, sizes.tempSizeInBytes,
                                 outBuffer, sizes.outputSizeInBytes, &handle, nullptr, 0));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(reinterpret_cast<void *>(tempBuffer));
    return handle;
  }

  static constexpr unsigned int kTracedGeometryFlags =
      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

  CUdeviceptr uploadTriangles(const std::vector<float3> &verts) {
    CUdeviceptr d;
    const size_t bytes = verts.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d), verts.data(), bytes, cudaMemcpyHostToDevice));
    return d;
  }

  SceneObject buildTriangleObject(const std::vector<float3> &verts, HitGroupData material) {
    SceneObject obj;
    obj.kind = GeometryKind::Triangle;
    obj.material = material;

    CUdeviceptr vertexBuffer = uploadTriangles(verts);

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.numVertices = static_cast<unsigned int>(verts.size());
    input.triangleArray.vertexBuffers = &vertexBuffer;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer, kTracedGeometryFlags);
    cudaFree(reinterpret_cast<void *>(vertexBuffer));
    return obj;
  }

  SceneObject buildSphereObject(float3 center, float radius, HitGroupData material) {
    SceneObject obj;
    obj.kind = GeometryKind::Sphere;
    obj.material = material;

    CUdeviceptr vertexBuffer, radiusBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&vertexBuffer), sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(vertexBuffer), &center, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&radiusBuffer), sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(radiusBuffer), &radius, sizeof(float), cudaMemcpyHostToDevice));

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    input.sphereArray.vertexBuffers = &vertexBuffer;
    input.sphereArray.numVertices = 1;
    input.sphereArray.radiusBuffers = &radiusBuffer;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.sphereArray.flags = flags;
    input.sphereArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer, kTracedGeometryFlags);
    cudaFree(reinterpret_cast<void *>(vertexBuffer));
    cudaFree(reinterpret_cast<void *>(radiusBuffer));
    return obj;
  }

  SceneObject buildSolidSphereObject(float3 center, float radius, HitGroupData material) {
    SceneObject obj;
    obj.kind = GeometryKind::SolidSphere;
    obj.material = material;
    obj.material.sphereCenter = center;
    obj.material.sphereRadius = radius;

    OptixAabb aabb{center.x - radius, center.y - radius, center.z - radius,
                   center.x + radius, center.y + radius, center.z + radius};
    CUdeviceptr aabbBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&aabbBuffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(aabbBuffer), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &aabbBuffer;
    input.customPrimitiveArray.numPrimitives = 1;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer);
    cudaFree(reinterpret_cast<void *>(aabbBuffer));
    return obj;
  }

  void addLightQuad(float3 corner, float3 v1, float3 v2, float3 emission) {
    if (hasEnvironment)
      return;
    light.corner = corner;
    light.v1 = v1;
    light.v2 = v2;
    light.normal = normalize(cross(v1, v2));
    light.emission = emission;

    HitGroupData lightMat{};
    lightMat.materialType = MATERIAL_LIGHT;
    lightMat.emission = emission;
    objects.push_back(buildTriangleObject(
        {corner, corner + v1, corner + v1 + v2, corner, corner + v1 + v2, corner + v2}, lightMat));
  }

  cudaTextureObject_t uploadTexture(const TextureAsset &tex, bool srgb) {
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t array = nullptr;
    CUDA_CHECK(cudaMallocArray(&array, &desc, tex.width, tex.height));
    CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, tex.pixelsRGBA.data(), tex.width * 4, tex.width * 4, tex.height,
                                    cudaMemcpyHostToDevice));
    textureArrays.push_back(array);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = srgb ? 1 : 0;
    cudaTextureObject_t obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&obj, &resDesc, &texDesc, nullptr));
    textureObjects.push_back(obj);
    return obj;
  }

  template <typename T> CUdeviceptr uploadVector(const std::vector<T> &v) {
    if (v.empty())
      return 0;
    CUdeviceptr d = 0;
    const size_t bytes = v.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d), v.data(), bytes, cudaMemcpyHostToDevice));
    return d;
  }

  SceneObject buildMeshObject(const MeshAsset &mesh) {
    SceneObject obj;
    obj.kind = GeometryKind::Triangle;
    obj.material.materialType = MATERIAL_TEXTURED_DIFFUSE;

    std::vector<float3> positions(mesh.positions.size());
    for (size_t i = 0; i < mesh.positions.size(); ++i)
      positions[i] = toFloat3(mesh.positions[i]);

    std::vector<float3> normals(mesh.normals.size());
    for (size_t i = 0; i < mesh.normals.size(); ++i)
      normals[i] = toFloat3(mesh.normals[i]);
    meshNormals = uploadVector(normals);
    obj.material.normals = reinterpret_cast<float3 *>(meshNormals);

    std::vector<float2> uvs(mesh.uvs.size());
    for (size_t i = 0; i < mesh.uvs.size(); ++i)
      uvs[i] = make_float2(mesh.uvs[i].x, mesh.uvs[i].y);
    meshUvs = uploadVector(uvs);
    obj.material.uvs = reinterpret_cast<float2 *>(meshUvs);

    std::vector<float4> tangents(mesh.tangents.size());
    for (size_t i = 0; i < mesh.tangents.size(); ++i)
      tangents[i] = make_float4(mesh.tangents[i].x, mesh.tangents[i].y, mesh.tangents[i].z, mesh.tangents[i].w);
    meshTangents = uploadVector(tangents);
    obj.material.tangents = reinterpret_cast<float4 *>(meshTangents);

    meshTriangleMaterial = uploadVector(mesh.triangleMaterial);
    obj.material.triangleMaterial = reinterpret_cast<unsigned int *>(meshTriangleMaterial);

    std::vector<cudaTextureObject_t> texObjects(mesh.textures.size(), 0);
    for (size_t i = 0; i < mesh.textures.size(); ++i)
      if (mesh.textures[i].width > 0 && mesh.textures[i].height > 0)
        texObjects[i] = uploadTexture(mesh.textures[i], mesh.textures[i].srgb);

    auto texOrZero = [&](int index) -> cudaTextureObject_t {
      return index >= 0 && index < static_cast<int>(texObjects.size()) ? texObjects[index] : 0;
    };
    std::vector<GpuMaterial> gpuMaterials(mesh.materials.size());
    for (size_t i = 0; i < mesh.materials.size(); ++i) {
      const MaterialAsset &m = mesh.materials[i];
      GpuMaterial &g = gpuMaterials[i];
      g.baseColorFactor = toFloat3(m.baseColorFactor);
      g.metallic = m.metallic;
      g.roughness = m.roughness;
      g.normalScale = m.normalScale;
      g.baseColorTex = texOrZero(m.baseColorTexture);
      g.metallicRoughnessTex = texOrZero(m.metallicRoughnessTexture);
      g.normalTex = texOrZero(m.normalTexture);
      g.transmission = m.transmission;
      g.ior = m.ior;
      g.attenuationColor = toFloat3(m.attenuationColor);
      g.attenuationDistance = m.attenuationDistance;
    }
    meshMaterials = uploadVector(gpuMaterials);
    obj.material.materials = reinterpret_cast<GpuMaterial *>(meshMaterials);

    CUdeviceptr vertexBuffer = uploadTriangles(positions);
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.numVertices = static_cast<unsigned int>(positions.size());
    input.triangleArray.vertexBuffers = &vertexBuffer;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer, kTracedGeometryFlags);
    cudaFree(reinterpret_cast<void *>(vertexBuffer));
    return obj;
  }

  SceneObject buildVoxelObject(const VoxelGrid &grid) {
    SceneObject obj;
    obj.kind = GeometryKind::Voxel;
    obj.material.materialType = MATERIAL_VOXEL;

    std::vector<OptixAabb> aabbs(grid.cells.size());
    for (size_t i = 0; i < grid.cells.size(); ++i) {
      const glm::vec3 mn = grid.cellMin(grid.cells[i]);
      const glm::vec3 mx = grid.cellMax(grid.cells[i]);
      aabbs[i] = OptixAabb{mn.x, mn.y, mn.z, mx.x, mx.y, mx.z};
    }
    const size_t aabbBytes = aabbs.size() * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&voxelAabbBuffer), aabbBytes));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void *>(voxelAabbBuffer), aabbs.data(), aabbBytes, cudaMemcpyHostToDevice));
    obj.material.voxelAabbs = reinterpret_cast<OptixAabb *>(voxelAabbBuffer);

    std::vector<float3> colors(grid.colors.size());
    for (size_t i = 0; i < grid.colors.size(); ++i)
      colors[i] = toFloat3(grid.colors[i]);
    const size_t colorBytes = colors.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&voxelColorBuffer), colorBytes));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void *>(voxelColorBuffer), colors.data(), colorBytes, cudaMemcpyHostToDevice));
    obj.material.voxelColors = reinterpret_cast<float3 *>(voxelColorBuffer);

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &voxelAabbBuffer;
    input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(aabbs.size());
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer);
    return obj;
  }

  SceneObject buildSdfObject(const SdfGrid &grid) {
    SceneObject obj;
    obj.kind = GeometryKind::Sdf;
    obj.material.materialType = MATERIAL_SDF;
    obj.material.sdfOrigin = toFloat3(grid.origin);
    obj.material.sdfVoxelSize = grid.voxelSize;
    obj.material.sdfNx = grid.nx;
    obj.material.sdfNy = grid.ny;
    obj.material.sdfNz = grid.nz;

    auto upload = [this](const void *src, size_t nbytes, std::vector<CUdeviceptr> &owner) -> CUdeviceptr {
      CUdeviceptr d = 0;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), nbytes));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d), src, nbytes, cudaMemcpyHostToDevice));
      owner.push_back(d);
      return d;
    };

    const size_t bytes = grid.distances.size() * sizeof(float);
    obj.material.sdfDistances =
        reinterpret_cast<float *>(upload(grid.distances.data(), bytes, sdfDistanceBuffers));

    if (!grid.baseColor.empty()) {
      std::vector<float3> baseColorHost(grid.baseColor.size());
      for (size_t i = 0; i < grid.baseColor.size(); ++i)
        baseColorHost[i] = toFloat3(grid.baseColor[i]);
      const size_t colorBytes = baseColorHost.size() * sizeof(float3);
      obj.material.sdfBaseColor =
          reinterpret_cast<float3 *>(upload(baseColorHost.data(), colorBytes, sdfBaseColorBuffers));

      const size_t scalarBytes = grid.metallic.size() * sizeof(float);
      obj.material.sdfMetallic =
          reinterpret_cast<float *>(upload(grid.metallic.data(), scalarBytes, sdfMetallicBuffers));
      obj.material.sdfRoughness =
          reinterpret_cast<float *>(upload(grid.roughness.data(), scalarBytes, sdfRoughnessBuffers));
    }

    if (!grid.branch.empty()) {
      const size_t branchBytes = grid.branch.size() * sizeof(uint8_t);
      obj.material.sdfBranch =
          reinterpret_cast<unsigned char *>(upload(grid.branch.data(), branchBytes, sdfBranchBuffers));

      std::vector<SdfGpuMaterial> paletteHost(grid.palette.size());
      for (size_t i = 0; i < grid.palette.size(); ++i) {
        const SdfPaletteMaterial &m = grid.palette[i];
        paletteHost[i] = SdfGpuMaterial{static_cast<unsigned int>(m.kind), toFloat3(m.baseColor), m.metallic,
                                         m.roughness, m.ior, toFloat3(m.extinction)};
      }
      const size_t paletteBytes = paletteHost.size() * sizeof(SdfGpuMaterial);
      obj.material.sdfPalette =
          reinterpret_cast<SdfGpuMaterial *>(upload(paletteHost.data(), paletteBytes, sdfPaletteBuffers));
      obj.material.sdfPaletteCount = static_cast<int>(grid.palette.size());
    }

    const glm::vec3 boundsMax = grid.boundsMax();
    OptixAabb aabb{grid.origin.x, grid.origin.y, grid.origin.z, boundsMax.x, boundsMax.y, boundsMax.z};
    CUdeviceptr aabbBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&aabbBuffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(aabbBuffer), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &aabbBuffer;
    input.customPrimitiveArray.numPrimitives = 1;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer);
    cudaFree(reinterpret_cast<void *>(aabbBuffer));
    return obj;
  }

  SceneObject buildSplatObject(const GsplatAsset &splats) {
    SceneObject obj;
    obj.kind = GeometryKind::Gsplat;
    obj.material.materialType = MATERIAL_GSPLAT;

    std::vector<OptixAabb> aabbs(splats.count());
    for (size_t i = 0; i < splats.count(); ++i) {
      const auto [lo, hi] = computeSplatAabb(splats.positions[i], splats.scales[i], splats.rotations[i]);
      aabbs[i] = OptixAabb{lo.x, lo.y, lo.z, hi.x, hi.y, hi.z};
    }
    CUdeviceptr aabbBuffer;
    const size_t aabbBytes = aabbs.size() * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&aabbBuffer), aabbBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(aabbBuffer), aabbs.data(), aabbBytes, cudaMemcpyHostToDevice));

    std::vector<float3> positions(splats.count());
    std::vector<float3> scales(splats.count());
    std::vector<float4> rotations(splats.count());
    std::vector<float3> colors(splats.count());
    for (size_t i = 0; i < splats.count(); ++i) {
      positions[i] = toFloat3(splats.positions[i]);
      scales[i] = toFloat3(splats.scales[i]);
      rotations[i] = make_float4(splats.rotations[i].x, splats.rotations[i].y, splats.rotations[i].z,
                                  splats.rotations[i].w);
      colors[i] = toFloat3(splats.colorDC[i]);
    }
    splatPositionBuffer = uploadVector(positions);
    obj.material.splatPositions = reinterpret_cast<float3 *>(splatPositionBuffer);
    splatScaleBuffer = uploadVector(scales);
    obj.material.splatScales = reinterpret_cast<float3 *>(splatScaleBuffer);
    splatRotationBuffer = uploadVector(rotations);
    obj.material.splatRotations = reinterpret_cast<float4 *>(splatRotationBuffer);
    splatOpacityBuffer = uploadVector(splats.opacity);
    obj.material.splatOpacity = reinterpret_cast<float *>(splatOpacityBuffer);
    splatColorBuffer = uploadVector(colors);
    obj.material.splatColors = reinterpret_cast<float3 *>(splatColorBuffer);

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &aabbBuffer;
    input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(aabbs.size());
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer);
    cudaFree(reinterpret_cast<void *>(aabbBuffer));
    return obj;
  }

  SceneObject buildNvdbObject(const NvdbVolume &volume) {
    SceneObject obj;
    obj.kind = GeometryKind::Nvdb;
    obj.material.materialType = MATERIAL_NVDB;

    const size_t blobBytes = volume.gridBlob.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&nvdbGridBuffer), blobBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(nvdbGridBuffer), volume.gridBlob.data(), blobBytes,
                           cudaMemcpyHostToDevice));

    const float sigmaScalar = (volume.sigmaT.x + volume.sigmaT.y + volume.sigmaT.z) / 3.0f;
    hasVolume = true;
    nvdbBoundsMin = toFloat3(volume.boundsMin);
    nvdbBoundsMax = toFloat3(volume.boundsMax);
    nvdbSigmaT = toFloat3(volume.sigmaT);
    nvdbScatterAlbedo = toFloat3(volume.scatterAlbedo);
    nvdbG = volume.g;
    nvdbDensityScale = volume.densityScale;
    nvdbMajorant = sigmaScalar * volume.maxDensity * volume.densityScale;

    obj.material.nvdbGrid = reinterpret_cast<void *>(nvdbGridBuffer);
    obj.material.nvdbBoundsMin = nvdbBoundsMin;
    obj.material.nvdbBoundsMax = nvdbBoundsMax;
    obj.material.nvdbSigmaT = nvdbSigmaT;
    obj.material.nvdbScatterAlbedo = nvdbScatterAlbedo;
    obj.material.nvdbG = nvdbG;
    obj.material.nvdbDensityScale = nvdbDensityScale;
    obj.material.nvdbMajorant = nvdbMajorant;

    OptixAabb aabb{nvdbBoundsMin.x, nvdbBoundsMin.y, nvdbBoundsMin.z,
                   nvdbBoundsMax.x, nvdbBoundsMax.y, nvdbBoundsMax.z};
    CUdeviceptr aabbBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&aabbBuffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(aabbBuffer), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &aabbBuffer;
    input.customPrimitiveArray.numPrimitives = 1;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    obj.gas = buildAccel(input, obj.gasBuffer);
    cudaFree(reinterpret_cast<void *>(aabbBuffer));
    return obj;
  }

  void buildSplatScene(const GsplatAsset &splats) {
    objects.push_back(buildSplatObject(splats));
    boundsCenter = splats.boundsCenter();
    boundsRadius = std::max(splats.boundsRadius(), 1e-3f);
    addGroundPlane(boundsCenter, boundsRadius, splats.boundsMin.y);
    addBoundsKeyLight(boundsCenter, boundsRadius);
  }

  void buildFixedTestScene() {
    HitGroupData groundMat{};
    groundMat.materialType = MATERIAL_DIFFUSE;
    groundMat.albedo = make_float3(0.72f, 0.72f, 0.7f);
    objects.push_back(buildTriangleObject(
        {
            make_float3(-3, -1, -3),
            make_float3(3, -1, -3),
            make_float3(3, -1, 3),
            make_float3(-3, -1, -3),
            make_float3(3, -1, 3),
            make_float3(-3, -1, 3),
        },
        groundMat));

    addLightQuad(make_float3(-0.6f, 2.9f, -0.6f), make_float3(1.2f, 0, 0), make_float3(0, 0, 1.2f),
                 make_float3(45.0f, 45.0f, 39.0f));

    HitGroupData diffuseMat{};
    diffuseMat.materialType = MATERIAL_DIFFUSE;
    diffuseMat.albedo = make_float3(0.75f, 0.22f, 0.2f);
    objects.push_back(buildSphereObject(make_float3(-1.2f, -0.3f, 0.0f), 0.7f, diffuseMat));

    HitGroupData mirrorMat{};
    mirrorMat.materialType = MATERIAL_MIRROR;
    mirrorMat.albedo = make_float3(0.9f, 0.9f, 0.92f);
    objects.push_back(buildSphereObject(make_float3(0.5f, -0.3f, 0.9f), 0.7f, mirrorMat));

    HitGroupData glassMat{};
    glassMat.materialType = MATERIAL_GLASS;
    glassMat.albedo = make_float3(1.0f, 1.0f, 1.0f);
    glassMat.ior = 1.5f;
    glassMat.extinction = make_float3(0.85f, 0.32f, 0.22f);
    objects.push_back(buildSolidSphereObject(make_float3(1.4f, 0.5f, -0.6f), 0.6f, glassMat));

    boundsCenter = glm::vec3(0.0f, 0.2f, 0.0f);
    boundsRadius = 3.0f;

    enableLightSubpaths = true;
  }

  void buildEmptyScene() {
    boundsCenter = glm::vec3(0.0f);
    boundsRadius = 1e-3f;
  }

  void addGroundPlane(const glm::vec3 &center, float radius, float floorY) {
    if (!wantGroundPlane)
      return;
    const float r = std::max(radius, 1.0f) * 1.0e4f;
    const float y = floorY - groundOffset - radius * 1e-3f;
    HitGroupData mat{};
    mat.materialType = MATERIAL_DIFFUSE;
    mat.albedo = make_float3(0.45f, 0.45f, 0.45f);
    const float3 c = toFloat3(center);
    objects.push_back(buildTriangleObject(
        {
            make_float3(c.x - r, y, c.z - r), make_float3(c.x + r, y, c.z - r), make_float3(c.x + r, y, c.z + r),
            make_float3(c.x - r, y, c.z - r), make_float3(c.x + r, y, c.z + r), make_float3(c.x - r, y, c.z + r),
        },
        mat));
  }

  void addBoundsKeyLight(const glm::vec3 &center, float radius) {
    const float3 c = toFloat3(center);
    const float r = radius;
    addLightQuad(make_float3(c.x - r, c.y + r * 2.2f, c.z - r), make_float3(r * 2.0f, 0, 0),
                 make_float3(0, 0, r * 2.0f), make_float3(20.0f, 20.0f, 18.0f));
  }

  void buildMeshScene(const MeshAsset &mesh) {
    objects.push_back(buildMeshObject(mesh));
    boundsCenter = mesh.boundsCenter();
    boundsRadius = std::max(mesh.boundsRadius(), 1e-3f);
    addGroundPlane(boundsCenter, boundsRadius, mesh.boundsMin.y);
    addBoundsKeyLight(boundsCenter, boundsRadius);

    for (const MaterialAsset &m : mesh.materials) {
      if (m.transmission > 0.0f || (m.metallic > kCausticMirrorMetallic && m.roughness < kCausticMirrorRoughness)) {
        enableLightSubpaths = true;
        break;
      }
    }
  }

  void buildVoxelScene(const VoxelGrid &grid) {
    objects.push_back(buildVoxelObject(grid));
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (const glm::ivec3 &c : grid.cells) {
      mn = glm::min(mn, grid.cellMin(c));
      mx = glm::max(mx, grid.cellMax(c));
    }
    boundsCenter = (mn + mx) * 0.5f;
    boundsRadius = std::max(glm::length(mx - mn) * 0.5f, 1e-3f);
    addGroundPlane(boundsCenter, boundsRadius, mn.y);
    addBoundsKeyLight(boundsCenter, boundsRadius);
  }

  static bool gridHasDielectric(const SdfGrid &grid) {
    for (const SdfPaletteMaterial &m : grid.palette)
      if (m.kind == SdfMaterialKind::Dielectric)
        return true;
    return false;
  }

  void buildSdfScene(const SdfGrid &grid) {
    objects.push_back(buildSdfObject(grid));
    const glm::vec3 mx = grid.boundsMax();
    boundsCenter = (grid.origin + mx) * 0.5f;
    boundsRadius = std::max(glm::length(mx - grid.origin) * 0.5f, 1e-3f);
    addGroundPlane(boundsCenter, boundsRadius, grid.origin.y);
    addBoundsKeyLight(boundsCenter, boundsRadius);
    if (gridHasDielectric(grid))
      enableLightSubpaths = true;
  }

  void buildEnvironment(const EnvironmentMap &env) {
    hasEnvironment = true;
    envWidth = env.width;
    envHeight = env.height;

    std::vector<float4> rgba(static_cast<size_t>(env.width) * env.height);
    for (size_t i = 0; i < rgba.size(); ++i)
      rgba[i] = make_float4(env.pixels[i * 3 + 0], env.pixels[i * 3 + 1], env.pixels[i * 3 + 2], 1.0f);

    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&envArray, &desc, env.width, env.height));
    CUDA_CHECK(cudaMemcpy2DToArray(envArray, 0, 0, rgba.data(), env.width * sizeof(float4), env.width * sizeof(float4),
                                    env.height, cudaMemcpyHostToDevice));
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envArray;
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    CUDA_CHECK(cudaCreateTextureObject(&envTexObj, &resDesc, &texDesc, nullptr));

    const size_t marginalBytes = env.marginalCdf.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&envMarginalCdfBuffer), marginalBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(envMarginalCdfBuffer), env.marginalCdf.data(), marginalBytes,
                           cudaMemcpyHostToDevice));

    const size_t conditionalBytes = env.conditionalCdf.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&envConditionalCdfBuffer), conditionalBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(envConditionalCdfBuffer), env.conditionalCdf.data(),
                           conditionalBytes, cudaMemcpyHostToDevice));
  }

  void buildScene(const SceneSource &source) {
    wantGroundPlane = source.groundPlane;
    groundOffset = source.groundOffset;
    if (source.environment)
      buildEnvironment(*source.environment);

    if (source.splats)
      buildSplatScene(*source.splats);
    else if (source.sdf)
      buildSdfScene(*source.sdf);
    else if (source.voxels)
      buildVoxelScene(*source.voxels);
    else if (source.mesh)
      buildMeshScene(*source.mesh);
    else if (source.emptyBase)
      buildEmptyScene();
    else
      buildFixedTestScene();

    if (!source.extraSdf.empty()) {
      glm::vec3 mn = boundsCenter - glm::vec3(boundsRadius);
      glm::vec3 mx = boundsCenter + glm::vec3(boundsRadius);
      for (const SdfGrid *grid : source.extraSdf) {
        objects.push_back(buildSdfObject(*grid));
        mn = glm::min(mn, grid->origin);
        mx = glm::max(mx, grid->boundsMax());
        if (gridHasDielectric(*grid))
          enableLightSubpaths = true;
      }
      boundsCenter = (mn + mx) * 0.5f;
      boundsRadius = std::max(glm::length(mx - mn) * 0.5f, 1e-3f);
    }

    if (source.emptyBase) {
      addGroundPlane(boundsCenter, boundsRadius, source.emptyBaseFloorY);
      addBoundsKeyLight(boundsCenter, boundsRadius);
    }

    if (source.volume) {
      objects.push_back(buildNvdbObject(*source.volume));
      const glm::vec3 volMin(nvdbBoundsMin.x, nvdbBoundsMin.y, nvdbBoundsMin.z);
      const glm::vec3 volMax(nvdbBoundsMax.x, nvdbBoundsMax.y, nvdbBoundsMax.z);
      boundsRadius = std::max({boundsRadius, glm::length(volMax - boundsCenter), glm::length(volMin - boundsCenter)});
    }

    if (enableLightSubpaths) {
      for (const SceneObject &o : objects) {
        if (o.kind == GeometryKind::Voxel || o.kind == GeometryKind::Gsplat || o.kind == GeometryKind::Nvdb) {
          std::fprintf(stderr,
                       "italy: light subpaths disabled — scene has a voxel/gsplat/nvdb object and "
                       "traceLightSubpathPass() has no hit-group for custom primitives (see buildScene()).\n");
          enableLightSubpaths = false;
          break;
        }
      }
    }

    buildIAS();

    if (enableLightSubpaths) {
      buildLightSubpathSbt();
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&lightVertexBuffer), sizeof(LightVertex) * kLightVertexCapacity));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lightVertexCounterBuffer), sizeof(unsigned int)));
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&causticAabbBuffer), sizeof(OptixAabb) * kLightVertexCapacity));
    }
  }

  void buildLightSubpathSbt() {
    RayGenRecord rgRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(lightSubpathRaygenPG, &rgRecord));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rgRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rgRecord, sizeof(rgRecord), cudaMemcpyHostToDevice));

    MissRecord missRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(lightSubpathMissPG, &missRecord));
    CUdeviceptr d_miss;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss), sizeof(missRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss), &missRecord, sizeof(missRecord), cudaMemcpyHostToDevice));

    std::vector<HitGroupRecord> hitRecords(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
      OptixProgramGroup pg = lightSubpathHitTrianglePG;
      if (objects[i].kind == GeometryKind::Sphere)
        pg = lightSubpathHitSpherePG;
      else if (objects[i].kind == GeometryKind::SolidSphere)
        pg = lightSubpathHitSolidSpherePG;
      else if (objects[i].kind == GeometryKind::Sdf)
        pg = lightSubpathHitSdfPG;
      OPTIX_CHECK(optixSbtRecordPackHeader(pg, &hitRecords[i]));
      hitRecords[i].data = objects[i].material;
    }
    CUdeviceptr d_hit;
    const size_t hitBytes = hitRecords.size() * sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit), hitBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hit), hitRecords.data(), hitBytes, cudaMemcpyHostToDevice));

    lightSubpathSbt.raygenRecord = d_rg;
    lightSubpathSbt.missRecordBase = d_miss;
    lightSubpathSbt.missRecordStrideInBytes = sizeof(MissRecord);
    lightSubpathSbt.missRecordCount = 1;
    lightSubpathSbt.hitgroupRecordBase = d_hit;
    lightSubpathSbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    lightSubpathSbt.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
  }

  unsigned int traceLightSubpathPass() {
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(lightVertexCounterBuffer), 0, sizeof(unsigned int), stream));
    OPTIX_CHECK(
        optixLaunch(pipeline, stream, paramsBuffer, sizeof(Params), &lightSubpathSbt, kLightSubpathBatchSize, 1, 1));

    unsigned int deposited = 0;
    CUDA_CHECK(cudaMemcpyAsync(&deposited, reinterpret_cast<void *>(lightVertexCounterBuffer), sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    deposited = std::min(deposited, kLightVertexCapacity);
    if (lightSubpathPassIndex < 3 || lightSubpathPassIndex % 64 == 0)
      std::fprintf(stderr, "italy: light-subpath pass %u: %u vertices deposited, mergeRadius %g, mergeGas %s\n",
                   lightSubpathPassIndex, deposited, static_cast<double>(mergeRadius),
                   mergeGasHandle ? "yes" : "no");

    if (mergeRadius < 0.0f)
      mergeRadius = boundsRadius * kInitialMergeRadiusFraction;
    else
      mergeRadius *= std::sqrt((lightSubpathPassIndex + kPpmAlpha) / (lightSubpathPassIndex + 1.0f));

    ++lightSubpathPassIndex;
    totalLightPathsEmitted += kLightSubpathBatchSize;
    return deposited;
  }

  void buildMergeGas(unsigned int vertexCount) {
    if (vertexCount == 0 || mergeRadius <= 0.0f) {
      mergeGasHandle = 0;
      return;
    }

    OPTIX_CHECK(optixLaunch(pipeline, stream, paramsBuffer, sizeof(Params), &causticAabbSbt, vertexCount, 1, 1));

    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &causticAabbBuffer;
    input.customPrimitiveArray.numPrimitives = vertexCount;
    static const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions{};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &input, 1, &sizes));
    if (sizes.outputSizeInBytes > mergeGasOutputCapacityBytes) {
      if (mergeGasOutputBuffer)
        cudaFree(reinterpret_cast<void *>(mergeGasOutputBuffer));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&mergeGasOutputBuffer), sizes.outputSizeInBytes));
      mergeGasOutputCapacityBytes = sizes.outputSizeInBytes;
    }
    if (sizes.tempSizeInBytes > mergeGasTempCapacityBytes) {
      if (mergeGasTempBuffer)
        cudaFree(reinterpret_cast<void *>(mergeGasTempBuffer));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&mergeGasTempBuffer), sizes.tempSizeInBytes));
      mergeGasTempCapacityBytes = sizes.tempSizeInBytes;
    }

    OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &input, 1, mergeGasTempBuffer,
                                 mergeGasTempCapacityBytes, mergeGasOutputBuffer, mergeGasOutputCapacityBytes,
                                 &mergeGasHandle, nullptr, 0));
  }

  void buildIAS() {
    std::vector<OptixInstance> instances(objects.size());
    static const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    for (size_t i = 0; i < objects.size(); ++i) {
      OptixInstance &inst = instances[i];
      std::memset(&inst, 0, sizeof(inst));
      std::memcpy(inst.transform, identity, sizeof(identity));
      inst.instanceId = static_cast<unsigned int>(i);
      inst.visibilityMask = 1;
      inst.sbtOffset = static_cast<unsigned int>(i);
      inst.flags = OPTIX_INSTANCE_FLAG_NONE;
      inst.traversableHandle = objects[i].gas;
    }

    CUdeviceptr instanceBuffer;
    const size_t instBytes = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&instanceBuffer), instBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(instanceBuffer), instances.data(), instBytes,
                           cudaMemcpyHostToDevice));

    OptixBuildInput iasInput{};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = instanceBuffer;
    iasInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    iasHandle = buildAccel(iasInput, iasBuffer);
    cudaFree(reinterpret_cast<void *>(instanceBuffer));
  }

  void buildSbt() {
    RayGenRecord rgRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rgRecord));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rgRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rgRecord, sizeof(rgRecord), cudaMemcpyHostToDevice));

    MissRecord missRecords[3]{};
    OPTIX_CHECK(optixSbtRecordPackHeader(missRadiancePG, &missRecords[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(missOcclusionPG, &missRecords[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(mergeMissPG, &missRecords[2]));
    CUdeviceptr d_miss;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss), sizeof(missRecords)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void *>(d_miss), missRecords, sizeof(missRecords), cudaMemcpyHostToDevice));

    mergeHitSbtOffset = static_cast<unsigned int>(objects.size());
    std::vector<HitGroupRecord> hitRecords(objects.size() + 1);
    for (size_t i = 0; i < objects.size(); ++i) {
      OptixProgramGroup pg;
      switch (objects[i].kind) {
      case GeometryKind::Sphere:
        pg = hitSpherePG;
        break;
      case GeometryKind::SolidSphere:
        pg = hitSolidSpherePG;
        break;
      case GeometryKind::Voxel:
        pg = hitVoxelPG;
        break;
      case GeometryKind::Sdf:
        pg = hitSdfPG;
        break;
      case GeometryKind::Gsplat:
        pg = hitGsplatPG;
        break;
      case GeometryKind::Nvdb:
        pg = hitNvdbPG;
        break;
      default:
        pg = hitTrianglePG;
        break;
      }
      OPTIX_CHECK(optixSbtRecordPackHeader(pg, &hitRecords[i]));
      hitRecords[i].data = objects[i].material;
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(mergeHitPG, &hitRecords[mergeHitSbtOffset]));
    CUdeviceptr d_hit;
    const size_t hitBytes = hitRecords.size() * sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit), hitBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hit), hitRecords.data(), hitBytes, cudaMemcpyHostToDevice));

    sbt.raygenRecord = d_rg;
    sbt.missRecordBase = d_miss;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 3;
    sbt.hitgroupRecordBase = d_hit;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
  }

  void buildReservoirBuildSbt() {
    RayGenRecord rgRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(reservoirBuildRaygenPG, &rgRecord));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rgRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rgRecord, sizeof(rgRecord), cudaMemcpyHostToDevice));

    reservoirBuildSbt.raygenRecord = d_rg;
    reservoirBuildSbt.missRecordBase = sbt.missRecordBase;
    reservoirBuildSbt.missRecordStrideInBytes = sbt.missRecordStrideInBytes;
    reservoirBuildSbt.missRecordCount = sbt.missRecordCount;
    reservoirBuildSbt.hitgroupRecordBase = sbt.hitgroupRecordBase;
    reservoirBuildSbt.hitgroupRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes;
    reservoirBuildSbt.hitgroupRecordCount = sbt.hitgroupRecordCount;
  }

  void freeDenoiser() {
    for (int i = 0; i < 2; ++i) {
      if (denoisedBuffer[i])
        cudaFree(reinterpret_cast<void *>(denoisedBuffer[i]));
      denoisedBuffer[i] = 0;
      if (denoiserInternalGuideBuffer[i])
        cudaFree(reinterpret_cast<void *>(denoiserInternalGuideBuffer[i]));
      denoiserInternalGuideBuffer[i] = 0;
    }
    if (denoiserStateBuffer)
      cudaFree(reinterpret_cast<void *>(denoiserStateBuffer));
    denoiserStateBuffer = 0;
    if (denoiserScratchBuffer)
      cudaFree(reinterpret_cast<void *>(denoiserScratchBuffer));
    denoiserScratchBuffer = 0;
    if (denoiser)
      optixDenoiserDestroy(denoiser);
    denoiser = nullptr;
    denoiserBuilt = false;
  }

  void buildDenoiser(int width, int height, bool temporal) {
    if (denoiserBuilt)
      freeDenoiser();

    OptixDenoiserOptions options{};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    OPTIX_CHECK(optixDenoiserCreate(
        context, temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV : OPTIX_DENOISER_MODEL_KIND_HDR, &options,
        &denoiser));
    denoiserTemporalActive = temporal;

    OptixDenoiserSizes sizes{};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, static_cast<unsigned int>(width),
                                                     static_cast<unsigned int>(height), &sizes));
    denoiserStateSize = sizes.stateSizeInBytes;
    denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;
    denoiserInternalGuideLayerPixelSizeInBytes = sizes.internalGuideLayerPixelSizeInBytes;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&denoiserStateBuffer), denoiserStateSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&denoiserScratchBuffer), denoiserScratchSize));
    OPTIX_CHECK(optixDenoiserSetup(denoiser, stream, static_cast<unsigned int>(width),
                                    static_cast<unsigned int>(height), denoiserStateBuffer, denoiserStateSize,
                                    denoiserScratchBuffer, denoiserScratchSize));

    for (int i = 0; i < 2; ++i)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&denoisedBuffer[i]),
                             static_cast<size_t>(width) * height * sizeof(float4)));

    if (temporal && sizes.internalGuideLayerPixelSizeInBytes > 0) {
      const size_t internalBytes =
          static_cast<size_t>(width) * height * sizes.internalGuideLayerPixelSizeInBytes;
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&denoiserInternalGuideBuffer[i]), internalBytes));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(denoiserInternalGuideBuffer[i]), 0, internalBytes,
                                    stream));
      }
    }

    denoiserBuilt = true;
  }

  void buildCausticAabbSbt() {
    RayGenRecord rgRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(causticAabbRaygenPG, &rgRecord));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rgRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rgRecord, sizeof(rgRecord), cudaMemcpyHostToDevice));
    causticAabbSbt.raygenRecord = d_rg;

    MissRecord missRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(missOcclusionPG, &missRecord));
    CUdeviceptr d_miss;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss), sizeof(missRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss), &missRecord, sizeof(missRecord), cudaMemcpyHostToDevice));
    causticAabbSbt.missRecordBase = d_miss;
    causticAabbSbt.missRecordStrideInBytes = sizeof(MissRecord);
    causticAabbSbt.missRecordCount = 1;
  }

  void buildTonemapSbt() {
    RayGenRecord rgRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(tonemapRaygenPG, &rgRecord));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rgRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rgRecord, sizeof(rgRecord), cudaMemcpyHostToDevice));
    tonemapSbt.raygenRecord = d_rg;

    MissRecord missRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(missOcclusionPG, &missRecord));
    CUdeviceptr d_miss;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss), sizeof(missRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss), &missRecord, sizeof(missRecord), cudaMemcpyHostToDevice));
    tonemapSbt.missRecordBase = d_miss;
    tonemapSbt.missRecordStrideInBytes = sizeof(MissRecord);
    tonemapSbt.missRecordCount = 1;
  }

  void initGLInterop(int width, int height) {
    const auto &gl = GLBufferFns::get();
    gl.glGenBuffers(1, &pbo);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<ptrdiff_t>(width) * height * 4, nullptr, GL_STREAM_DRAW);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&accumBuffer[i]),
                             static_cast<size_t>(width) * height * sizeof(float4)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&accumAlbedoBuffer[i]),
                             static_cast<size_t>(width) * height * sizeof(float4)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&accumNormalBuffer[i]),
                             static_cast<size_t>(width) * height * sizeof(float4)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&motionVectorBuffer),
                           static_cast<size_t>(width) * height * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&denoiserFlowBuffer),
                           static_cast<size_t>(width) * height * sizeof(float2)));
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&reservoirBuffer[i]),
                             static_cast<size_t>(width) * height * sizeof(Reservoir)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&paramsBuffer), sizeof(Params)));
  }
};

OptixRenderer::OptixRenderer(int width, int height, const SceneSource &source)
    : impl_(new Impl()), width_(width), height_(height) {
  impl_->initContext();
  impl_->buildModule();
  impl_->buildProgramGroups();
  impl_->buildPipeline();
  impl_->buildScene(source);
  impl_->buildSbt();
  impl_->buildReservoirBuildSbt();
  impl_->buildTonemapSbt();
  impl_->buildCausticAabbSbt();
  impl_->initGLInterop(width, height);
  impl_->buildDenoiser(width, height, true);

  sceneBoundsCenter_ = impl_->boundsCenter;
  sceneBoundsRadius_ = impl_->boundsRadius;

  glGenTextures(1, &glTexture_);
  glBindTexture(GL_TEXTURE_2D, glTexture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);
}

OptixRenderer::~OptixRenderer() {
  if (glTexture_)
    glDeleteTextures(1, &glTexture_);
  delete impl_;
}

void OptixRenderer::resetAccumulation() {
  subframeIndex_ = 0;
  cameraMovedPending_ = false;
}

void OptixRenderer::notifyCameraMoved() { cameraMovedPending_ = true; }

bool OptixRenderer::readAccumulationRgb(std::vector<float> &rgb) const {
  const CUdeviceptr lastWritten = impl_->accumBuffer[1 - impl_->nextWriteIdx];
  if (!lastWritten || subframeIndex_ == 0)
    return false;
  const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<float4> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), reinterpret_cast<const void *>(lastWritten), count * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  rgb.resize(count * 3);
  for (size_t i = 0; i < count; ++i) {
    rgb[i * 3 + 0] = host[i].x;
    rgb[i * 3 + 1] = host[i].y;
    rgb[i * 3 + 2] = host[i].z;
  }
  return true;
}

void OptixRenderer::render(const OrbitCamera &camera, const RenderSettings &settings) {
  const glm::vec3 eye = camera.position();
  const glm::vec3 target = camera.target();
  const glm::vec3 forward = glm::normalize(target - eye);
  const glm::vec3 worldUp(0, 1, 0);
  const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
  const glm::vec3 up = glm::cross(right, forward);
  const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
  const float tanHalfFov = std::tan(camera.fovYRadians * 0.5f);

  const int writeIdx = impl_->nextWriteIdx;
  const int readIdx = 1 - writeIdx;

  Params params{};
  params.subframeIndex = subframeIndex_;
  params.accumBuffer = reinterpret_cast<float4 *>(impl_->accumBuffer[writeIdx]);
  params.accumAlbedoBuffer = reinterpret_cast<float4 *>(impl_->accumAlbedoBuffer[writeIdx]);
  params.accumNormalBuffer = reinterpret_cast<float4 *>(impl_->accumNormalBuffer[writeIdx]);
  params.prevAccumBuffer = reinterpret_cast<float4 *>(impl_->accumBuffer[readIdx]);
  params.prevAccumAlbedoBuffer = reinterpret_cast<float4 *>(impl_->accumAlbedoBuffer[readIdx]);
  params.prevAccumNormalBuffer = reinterpret_cast<float4 *>(impl_->accumNormalBuffer[readIdx]);
  params.motionVectorBuffer = reinterpret_cast<float2 *>(impl_->motionVectorBuffer);
  params.denoiserFlowBuffer = reinterpret_cast<float2 *>(impl_->denoiserFlowBuffer);
  params.cameraMoved = cameraMovedPending_ ? 1u : 0u;
  params.width = static_cast<unsigned int>(width_);
  params.height = static_cast<unsigned int>(height_);
  params.samplesPerLaunch = settings.samplesPerLaunch;
  params.exposure = settings.exposure;
  params.tonemapOperator = static_cast<unsigned int>(settings.tonemap);
  params.fireflyClamp = settings.fireflyClamp;
  params.aperture = settings.aperture;
  params.focusDistance =
      settings.focusDistance > 0.0f ? settings.focusDistance : glm::length(target - eye);
  params.envRotation = settings.envRotation;
  params.backgroundColor = toFloat3(settings.backgroundColor);
  params.sun.enabled = settings.sunEnabled ? 1u : 0u;
  const glm::vec3 sunDir = glm::normalize(settings.sunDirection);
  const float sunCos = std::cos(settings.envRotation);
  const float sunSin = std::sin(settings.envRotation);
  params.sun.direction = toFloat3(glm::vec3(sunDir.x * sunCos - sunDir.z * sunSin, sunDir.y,
                                            sunDir.x * sunSin + sunDir.z * sunCos));
  params.sun.cosAngularRadius = std::cos(glm::radians(settings.sunAngularRadiusDeg));
  params.sun.radiance = toFloat3(settings.sunRadiance);
  params.volume.enabled = impl_->hasVolume ? 1u : 0u;
  if (impl_->hasVolume) {
    params.volume.grid = reinterpret_cast<void *>(impl_->nvdbGridBuffer);
    params.volume.boundsMin = impl_->nvdbBoundsMin;
    params.volume.boundsMax = impl_->nvdbBoundsMax;
    params.volume.sigmaT = impl_->nvdbSigmaT;
    params.volume.scatterAlbedo = impl_->nvdbScatterAlbedo;
    params.volume.g = impl_->nvdbG;
    params.volume.densityScale = impl_->nvdbDensityScale;
    params.volume.majorant = impl_->nvdbMajorant;
  }
  params.sceneBoundsCenter = toFloat3(impl_->boundsCenter);
  params.sceneBoundsRadius = impl_->boundsRadius;
  params.eye = toFloat3(eye);
  params.U = toFloat3(right * tanHalfFov * aspect);
  params.V = toFloat3(up * tanHalfFov);
  params.W = toFloat3(forward);
  if (!impl_->hasPrevCamera) {
    impl_->prevEyeUsed = params.eye;
    impl_->prevUUsed = params.U;
    impl_->prevVUsed = params.V;
    impl_->prevWUsed = params.W;
    impl_->hasPrevCamera = true;
  }
  params.prevEye = impl_->prevEyeUsed;
  params.prevU = impl_->prevUUsed;
  params.prevV = impl_->prevVUsed;
  params.prevW = impl_->prevWUsed;
  params.light = impl_->light;
  params.extraLightCount = 0;
  if (settings.extraTestLightCount > 0 && !impl_->hasEnvironment) {
    const unsigned int n = std::min(settings.extraTestLightCount, kMaxExtraLights);
    params.extraLightCount = n;
    const float3 tangent = normalize(impl_->light.v1);
    const float3 bitangent = normalize(impl_->light.v2);
    const float radius = impl_->boundsRadius * 0.6f;
    for (unsigned int i = 0; i < n; ++i) {
      QuadLight q = impl_->light;
      const float angle = 6.2831853f * static_cast<float>(i + 1) / static_cast<float>(n + 1);
      const float3 offset = tangent * (radius * std::cos(angle)) + bitangent * (radius * std::sin(angle));
      q.corner = impl_->light.corner + offset;
      params.extraLights[i] = q;
    }
  }
  params.handle = impl_->iasHandle;
  params.reservoirNEE = settings.reservoirNEE ? 1u : 0u;
  params.reservoirTemporal = settings.reservoirTemporal ? 1u : 0u;
  params.reservoirBuffer = reinterpret_cast<Reservoir *>(impl_->reservoirBuffer[writeIdx]);
  params.prevReservoirBuffer = reinterpret_cast<Reservoir *>(impl_->reservoirBuffer[readIdx]);
  if (impl_->hasEnvironment) {
    params.envTex = impl_->envTexObj;
    params.envMarginalCdf = reinterpret_cast<float *>(impl_->envMarginalCdfBuffer);
    params.envConditionalCdf = reinterpret_cast<float *>(impl_->envConditionalCdfBuffer);
    params.envWidth = impl_->envWidth;
    params.envHeight = impl_->envHeight;
  }
  if (impl_->enableLightSubpaths && settings.lightSubpaths) {
    params.lightVertices = reinterpret_cast<LightVertex *>(impl_->lightVertexBuffer);
    params.lightVertexCounter = reinterpret_cast<unsigned int *>(impl_->lightVertexCounterBuffer);
    params.lightVertexCapacity = Impl::kLightVertexCapacity;
    params.lightSubpathBatchSize = Impl::kLightSubpathBatchSize;
    params.totalLightPathsEmitted = impl_->totalLightPathsEmitted;
    params.causticAabbs = reinterpret_cast<OptixAabb *>(impl_->causticAabbBuffer);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(impl_->paramsBuffer), &params, sizeof(Params),
                               cudaMemcpyHostToDevice, impl_->stream));
    const unsigned int deposited = impl_->traceLightSubpathPass();
    params.lightVertexCount = deposited;
    params.totalLightPathsEmitted = impl_->totalLightPathsEmitted;
    params.mergeRadius = impl_->mergeRadius;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(impl_->paramsBuffer), &params, sizeof(Params),
                               cudaMemcpyHostToDevice, impl_->stream));
    impl_->buildMergeGas(deposited);
    params.vertexMergeHandle = impl_->mergeGasHandle;
    params.mergeHitSbtOffset = impl_->mergeHitSbtOffset;
    params.maxConnectionsPerVertex = settings.maxConnectionsPerVertex;
  }

  if (settings.reservoirNEE) {
    params.reservoirBuildPass = 1u;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(impl_->paramsBuffer), &params, sizeof(Params),
                               cudaMemcpyHostToDevice, impl_->stream));
    OPTIX_CHECK(optixLaunch(impl_->pipeline, impl_->stream, impl_->paramsBuffer, sizeof(Params),
                             &impl_->reservoirBuildSbt, width_, height_, 1));
    params.reservoirBuildPass = 0u;
  }

  CUDA_CHECK(cudaGraphicsMapResources(1, &impl_->cudaPbo, impl_->stream));
  size_t mappedSize = 0;
  void *devicePtr = nullptr;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&devicePtr, &mappedSize, impl_->cudaPbo));
  params.frameBuffer = reinterpret_cast<uchar4 *>(devicePtr);
  params.denoiserEnabled = settings.denoise ? 1u : 0u;
  if (settings.denoise && impl_->denoiserTemporalActive != settings.denoiseTemporal)
    impl_->buildDenoiser(width_, height_, settings.denoiseTemporal);
  params.denoisedBuffer = reinterpret_cast<float4 *>(impl_->denoisedBuffer[writeIdx]);

  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(impl_->paramsBuffer), &params, sizeof(Params),
                             cudaMemcpyHostToDevice, impl_->stream));
  OPTIX_CHECK(optixLaunch(impl_->pipeline, impl_->stream, impl_->paramsBuffer, sizeof(Params), &impl_->sbt, width_,
                           height_, 1));

  if (settings.denoise) {
    const bool temporal = impl_->denoiserTemporalActive;

    OptixDenoiserParams denoiserParams{};
    denoiserParams.temporalModeUsePreviousLayers = (temporal && subframeIndex_ != 0) ? 1u : 0u;

    OptixImage2D img{};
    img.width = static_cast<unsigned int>(width_);
    img.height = static_cast<unsigned int>(height_);
    img.rowStrideInBytes = static_cast<unsigned int>(width_) * sizeof(float4);
    img.pixelStrideInBytes = sizeof(float4);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D inputImg = img;
    inputImg.data = impl_->accumBuffer[writeIdx];
    OptixImage2D outputImg = img;
    outputImg.data = impl_->denoisedBuffer[writeIdx];

    OptixDenoiserLayer layer{};
    layer.input = inputImg;
    layer.output = outputImg;
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    if (temporal) {
      OptixImage2D prevOutputImg = img;
      prevOutputImg.data = impl_->denoisedBuffer[readIdx];
      layer.previousOutput = prevOutputImg;
    }

    OptixImage2D albedoImg = img;
    albedoImg.data = impl_->accumAlbedoBuffer[writeIdx];
    OptixImage2D normalImg = img;
    normalImg.data = impl_->accumNormalBuffer[writeIdx];

    OptixDenoiserGuideLayer guideLayer{};
    guideLayer.albedo = albedoImg;
    guideLayer.normal = normalImg;

    if (temporal) {
      OptixImage2D flowImg{};
      flowImg.data = impl_->denoiserFlowBuffer;
      flowImg.width = static_cast<unsigned int>(width_);
      flowImg.height = static_cast<unsigned int>(height_);
      flowImg.rowStrideInBytes = static_cast<unsigned int>(width_) * sizeof(float2);
      flowImg.pixelStrideInBytes = sizeof(float2);
      flowImg.format = OPTIX_PIXEL_FORMAT_FLOAT2;
      guideLayer.flow = flowImg;

      OptixImage2D prevInternalImg{};
      prevInternalImg.data = impl_->denoiserInternalGuideBuffer[readIdx];
      prevInternalImg.width = static_cast<unsigned int>(width_);
      prevInternalImg.height = static_cast<unsigned int>(height_);
      prevInternalImg.pixelStrideInBytes =
          static_cast<unsigned int>(impl_->denoiserInternalGuideLayerPixelSizeInBytes);
      prevInternalImg.rowStrideInBytes = prevInternalImg.pixelStrideInBytes * prevInternalImg.width;
      prevInternalImg.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
      guideLayer.previousOutputInternalGuideLayer = prevInternalImg;

      OptixImage2D outInternalImg = prevInternalImg;
      outInternalImg.data = impl_->denoiserInternalGuideBuffer[writeIdx];
      guideLayer.outputInternalGuideLayer = outInternalImg;
    }

    OPTIX_CHECK(optixDenoiserInvoke(impl_->denoiser, impl_->stream, &denoiserParams, impl_->denoiserStateBuffer,
                                     impl_->denoiserStateSize, &guideLayer, &layer, 1, 0, 0,
                                     impl_->denoiserScratchBuffer, impl_->denoiserScratchSize));

    OPTIX_CHECK(optixLaunch(impl_->pipeline, impl_->stream, impl_->paramsBuffer, sizeof(Params), &impl_->tonemapSbt,
                             width_, height_, 1));
  }

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &impl_->cudaPbo, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  const auto &gl = GLBufferFns::get();
  gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->pbo);
  glBindTexture(GL_TEXTURE_2D, glTexture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);
  gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  impl_->prevEyeUsed = params.eye;
  impl_->prevUUsed = params.U;
  impl_->prevVUsed = params.V;
  impl_->prevWUsed = params.W;
  impl_->nextWriteIdx = readIdx;
  cameraMovedPending_ = false;

  ++subframeIndex_;
}

} // namespace italy

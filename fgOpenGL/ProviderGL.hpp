/* fgOpenGL - OpenGL Provider for Feather GUI
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef FG__OPENGL_H
#define FG__OPENGL_H

#include "Context.hpp"
#include <vector>

#define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg __VA_OPT__(, ) __VA_ARGS__)

namespace GL {
  class Provider : public FG_GraphicsInterface
  {
  public:
    Provider(void* root, FG_Log log);
    ~Provider();
    template<typename... Args> void Log(FG_Level level, const char* file, int line, const char* msg, Args&&... args)
    {
      if constexpr(sizeof...(Args) == 0)
        _log(_logctx, level, file, line, msg, nullptr, 0, &FreeImpl);
      else
        fgLogImpl<&FreeImpl>(_log, _logctx, level, file, line, msg, std::tuple<Args...>(std::forward<Args>(args)...),
                             std::index_sequence_for<Args...>{});
    }
    FG_COMPILER_DLLEXPORT int LoadGL(GLADloadproc loader);
    static void FreeImpl(char* p) { free(p); }
    static FG_Caps GetCaps(FG_GraphicsInterface* self);
    static FG_Context* CreateContext(FG_GraphicsInterface* self, FG_Vec2i size, enum FG_PixelFormat backbuffer);
    static int ResizeContext(FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size);
    static FG_Shader CompileShader(FG_GraphicsInterface* self, FG_Context* context, enum FG_ShaderStage stage,
                                   const char* source);
    static int DestroyShader(FG_GraphicsInterface* self, FG_Context* context, FG_Shader shader);
    static FG_CommandList* CreateCommandList(FG_GraphicsInterface* self, FG_Context* context, bool bundle);
    static int DestroyCommandList(FG_GraphicsInterface* self, FG_Context* context, FG_CommandList* commands);
    static int Clear(FG_GraphicsInterface* self, FG_CommandList* commands, uint8_t clearbits, FG_Color16 RGBA,
                     uint8_t stencil, float depth, uint32_t num_rects, FG_Rect* rects);
    static int CopyResource(FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                            FG_Vec3i size, int mipmaplevel);
    static int CopySubresource(FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                               unsigned long srcoffset, unsigned long destoffset, unsigned long bytes);
    static int CopyResourceRegion(FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                                  int level, FG_Vec3i srcoffset, FG_Vec3i destoffset, FG_Vec3i size);
    static int DrawGL(FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t vertexcount, uint32_t instancecount,
                      uint32_t startvertex, uint32_t startinstance);
    static int DrawIndexed(FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t indexcount,
                           uint32_t instancecount, uint32_t startindex, int startvertex, uint32_t startinstance);
    static int DrawMesh(FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t first, uint32_t count);
    static int Dispatch(FG_GraphicsInterface* self, FG_CommandList* commands);
    static int SyncPoint(FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t barrier_flags);
    static int SetPipelineState(FG_GraphicsInterface* self, FG_CommandList* commands, uintptr_t state);
    static int SetViewports(FG_GraphicsInterface* self, FG_CommandList* commands, FG_Viewport* viewports, uint32_t count);
    static int SetScissors(FG_GraphicsInterface* self, FG_CommandList* commands, FG_Rect* rects, uint32_t count);
    static int SetShaderConstants(FG_GraphicsInterface* self, FG_CommandList* commands, const FG_ShaderParameter* uniforms,
                                  const FG_ShaderValue* values, uint32_t count);
    static int Execute(FG_GraphicsInterface* self, FG_Context* context, FG_CommandList* commands);
    static uintptr_t CreatePipelineState(FG_GraphicsInterface* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                         FG_Resource rendertarget, FG_Blend* blends, FG_Resource* vertexbuffer,
                                         GLsizei* strides, uint32_t n_buffers, FG_VertexParameter* attributes,
                                         uint32_t n_attributes, FG_Resource indexbuffer, uint8_t indexstride);
    static uintptr_t CreateComputePipeline(FG_GraphicsInterface* self, FG_Context* context, FG_Shader computeshader,
                                           FG_Vec3i workgroup, uint32_t flags);
    static int DestroyPipelineState(FG_GraphicsInterface* self, FG_Context* context, uintptr_t state);
    static FG_Resource CreateBuffer(FG_GraphicsInterface* self, FG_Context* context, void* data, uint32_t bytes,
                                    enum FG_Usage usage);
    static FG_Resource CreateTexture(FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size, enum FG_Usage usage,
                                     enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount);
    static FG_Resource CreateRenderTarget(FG_GraphicsInterface* self, FG_Context* context, FG_Resource depthstencil,
                                          FG_Resource* textures, uint32_t n_textures, int attachments);
    static int DestroyResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource);
    static void* MapResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, uint32_t offset,
                             uint32_t length, enum FG_Usage usage, uint32_t access);
    static int UnmapResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, enum FG_Usage usage);
    static int BeginDraw(FG_GraphicsInterface* self, FG_Context* context, FG_Rect* area);
    static int EndDraw(FG_GraphicsInterface* self, FG_Context* context);
    static int DestroyGL(FG_GraphicsInterface* self);

    static void ErrorCallback(int error, const char* description);
    static void JoystickCallback(int id, int connected);

    static constexpr FG_Resource NULL_RESOURCE = 0;
    static constexpr FG_Shader NULL_SHADER     = 0;
    static constexpr uintptr_t NULL_PIPELINE   = 0;
    static constexpr void* NULL_COMMANDLIST    = nullptr;

#ifdef AUTO_FRIEND
    friend AUTO_FRIEND;
#endif

  protected:
    FG_Log _log;
    bool _insidelist;
    void* _logctx;
  };
}

#endif

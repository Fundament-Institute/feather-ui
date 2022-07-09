/* fgOpenGL - OpenGL Backend for Feather GUI
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

#include "Window.hpp"
#include <vector>

struct FT_LibraryRec_;

#ifdef FG_COMPILER_MSC
  #define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg, __VA_ARGS__)
#else
  #define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg __VA_OPT__(,) __VA_ARGS__)
#endif

namespace GL {
  enum GL_Err
  {
    ERR_SUCCESS           = 0,
    ERR_UNKNOWN           = -1,
    ERR_NOT_IMPLEMENTED   = -2,
    ERR_MISSING_PARAMETER = -0xFFFD,
    ERR_UNKNOWN_COMMAND_CATEGORY,
    ERR_INVALID_KIND,
    ERR_CLIPBOARD_FAILURE,
    ERR_INVALID_CURSOR,
    ERR_INVALID_DISPLAY,
    ERR_INVALID_PARAMETER,
    ERR_INVALID_CALL,
    ERR_NULL,
  };

  template<class T> constexpr void* pack_ptr(T i) noexcept
  {
    static_assert(sizeof(T) <= sizeof(void*), "Cannot fit T into pointer");
    return reinterpret_cast<void*>(static_cast<size_t>(i));
  }

  template<class T> constexpr T unpack_ptr(void* p) noexcept
  {
    static_assert(sizeof(T) <= sizeof(void*), "Cannot fit T into pointer");
    return static_cast<T>(reinterpret_cast<size_t>(p));
  }

  // C++23 introduces uz but we don't have that yet.
  static inline constexpr std::size_t operator"" _uz(unsigned long long int x) noexcept { return x; }

  class Backend : public FG_Backend
  {
    template<class T> constexpr FG_LogValue log_value(const T& v)
    {
      using U = std::remove_cvref_t<T>;
      // We do not use is_same for anything other than bool, because types like "unsigned long long" and "uint64_t" might
      // not actually be the same type depending on the compiler
      if constexpr(std::is_same_v<U, bool>)
        return FG_LogValue{ .type = FG_LogType_Boolean, .bit = v };
      else if constexpr(std::is_same_v<T, const char*>)
        return FG_LogValue{ .type = FG_LogType_String, .string = v };
      else if constexpr(std::is_same_v<U, std::string>)
      {
        char* str = malloc(v.size() + 1);
        strcpy(str, v.data());
        return FG_LogValue{ .type = FG_LogType_String, .owned = str };
      }
      else if constexpr(std::is_same_v<U, std::unique_ptr<char>>)
        return FG_LogValue{ .type = FG_LogType_String, .owned = v.release() };
      else if constexpr(std::is_floating_point_v<U> && sizeof(U) == sizeof(float))
        return FG_LogValue{ .type = FG_LogType_F32, .f32 = v };
      else if constexpr(std::is_floating_point_v<U> && sizeof(U) == sizeof(double))
        return FG_LogValue{ .type = FG_LogType_F64, .f64 = v };
      else if constexpr(std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == sizeof(int32_t))
        return FG_LogValue{ .type = FG_LogType_I32, .i32 = v };
      else if constexpr(std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == sizeof(int32_t))
        return FG_LogValue{ .type = FG_LogType_U32, .u32 = v };
      else if constexpr(std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == sizeof(int64_t))
        return FG_LogValue{ .type = FG_LogType_I64, .i64 = v };
      else if constexpr(std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == sizeof(int64_t))
        return FG_LogValue{ .type = FG_LogType_I64, .i64 = v };
      else
        static_assert(std::is_same_v<U, bool>, "Invalid type for FG_LogValue");
    }
    template<typename T, std::size_t... I>
    void log_impl(FG_Level level, const char* file, int line, const char* msg, const T& args, std::index_sequence<I...>)
    {
      FG_LogValue values[sizeof...(I)] = { log_value(std::get<I>(args))... };

      _log(_root, level, file, line, msg, values, sizeof...(I), &FreeGL);
    }

  public:
    Backend(void* root, FG_Log log, FG_Behavior behavior);
    ~Backend();
    FG_Result Behavior(Context* data, const FG_Msg& msg);
    template<typename... Args> void Log(FG_Level level, const char* file, int line, const char* msg, Args&&... args)
    {
      if constexpr(sizeof...(Args) == 0)
        _log(_root, level, file, line, msg, nullptr, 0, &FreeGL);
      else
        log_impl(level, file, line, msg, std::tuple<Args...>(std::forward<Args>(args)...),
                 std::index_sequence_for<Args...>{});
    }

    static void FreeGL(char* p) { free(p); }
    static FG_Caps GetCaps(FG_Backend* self);
    static void* CompileShader(FG_Backend* self, FG_Context* context, enum FG_ShaderStage stage, const char* source);
    static int DestroyShader(FG_Backend* self, FG_Context* context, void* shader);
    static void* CreateCommandList(FG_Backend* self, FG_Context* context, bool bundle);
    static int DestroyCommandList(FG_Backend* self, void* commands);
    static int ClearDepthStencil(FG_Backend* self, void* commands, FG_Resource* rendertarget, char clear, uint8_t stencil,
                                 float depth, uint32_t num_rects, FG_Rect* rects);
    static int ClearRenderTarget(FG_Backend* self, void* commands, FG_Resource* rendertarget, FG_Color RGBA,
                                 uint32_t num_rects, FG_Rect* rects);
    static int CopyResource(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest);
    static int CopySubresource(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest,
                               unsigned long srcoffset, unsigned long destoffset, unsigned long bytes);
    static int CopyResourceRegion(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest, FG_Vec3i srcoffset,
                                  FG_Vec3i destoffset, FG_Vec3i size);
    static int DrawGL(FG_Backend* self, void* commands, uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex,
                      uint32_t startinstance);
    static int DrawIndexed(FG_Backend* self, void* commands, uint32_t indexcount, uint32_t instancecount,
                           uint32_t startindex, int startvertex, uint32_t startinstance);
    static int SetPipelineState(FG_Backend* self, void* commands, void* state);
    static int SetDepthStencil(FG_Backend* self, void* commands, bool Front, uint8_t StencilFailOp,
                               uint8_t StencilDepthFailOp, uint8_t StencilPassOp, uint8_t StencilFunc);
    static int SetViewports(FG_Backend* self, void* commands, FG_Viewport* viewports, uint32_t count);
    static int SetShaderConstants(FG_Backend* self, void* commands, const FG_ShaderParameter* uniforms,
                                  const FG_ShaderValue* values, uint32_t count);
    static int Execute(FG_Backend* self, FG_Context* context, void* commands);
    static void* CreatePipelineState(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                     FG_Resource** rendertargets, uint32_t n_targets, FG_Blend* blends,
                                     FG_Resource** vertexbuffer, GLsizei* strides, uint32_t n_buffers,
                                     FG_ShaderParameter* attributes, uint32_t n_attributes, FG_Resource* indexbuffer,
                                     uint8_t indexstride);
    static int DestroyPipelineState(FG_Backend* self, FG_Context* context, void* state);
    static FG_Resource* CreateBuffer(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes, enum FG_Type type);
    static FG_Resource* CreateTexture(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_Type type,
                                      enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount);
    static FG_Resource* CreateRenderTarget(FG_Backend* self, FG_Context* context, FG_Resource* texture, FG_Vec2i size);
    static int DestroyResource(FG_Backend* self, FG_Context* context, FG_Resource* resource);
    static FG_Window* CreateWindowGL(FG_Backend* self, FG_Element* element, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                                     const char* caption, uint64_t flags);
    static int SetWindow(FG_Backend* self, FG_Window* window, FG_Element* element, FG_Display* display, FG_Vec2* pos,
                         FG_Vec2* dim, const char* caption, uint64_t flags);
    static int DestroyWindow(FG_Backend* self, FG_Window* window);
    static int BeginDraw(FG_Backend* self, FG_Context* context, FG_Rect* area);
    static int EndDraw(FG_Backend* self, FG_Context* context);
    static int PutClipboard(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind, const char* data, uint32_t count);
    static uint32_t GetClipboard(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind, void* target, uint32_t count);
    static bool CheckClipboard(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind);
    static int ClearClipboard(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind);
    static int ProcessMessages(FG_Backend* self, FG_Window* window);
    static int GetMessageSyncObject(FG_Backend* self, FG_Window* window);
    static int SetCursorGL(FG_Backend* self, FG_Window* window, enum FG_Cursor cursor);
    static int GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out);
    static int GetDisplay(FG_Backend* self, void* handle, FG_Display* out);
    static int GetDisplayWindow(FG_Backend* self, FG_Window* window, FG_Display* out);
    static int CreateSystemControl(FG_Backend* self, FG_Context* context, const char* id, FG_Rect* area, ...);
    static int SetSystemControl(FG_Backend* self, FG_Context* context, void* control, FG_Rect* area, ...);
    static int DestroySystemControl(FG_Backend* self, FG_Context* context, void* control);
    static int DestroyGL(FG_Backend* self);

    static void ErrorCallback(int error, const char* description);
    static void JoystickCallback(int id, int connected);

    void* _root;
    Window* _windows;

    static int _lasterr;
    static int _refcount;
    static char _lasterrdesc[1024]; // 1024 is from GLFW internals
    static int _maxjoy;
    static const float BASE_DPI;
    static Backend* _singleton;
    static void* _library;

  protected:
    FG_Log _log;
    FG_Behavior _behavior;
    bool _insidelist;
  };
}

#endif

// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "feather/compiler.h"
#include "glad/glad.h"
#include "feather/graphics_interface.h"
#include "ShaderObject.hpp"
#include <math.h>
#include <vector>
#include <array>
#include <utility>
#include <memory>
#include <span>
#include <immintrin.h>

namespace GL {
  class Provider;
  struct ProgramObject;

  enum class GLCaps
  {
    GLCAP_GAMMA_EXT            = 1,
    GLCAP_INSTANCES_EXT        = 2,
    GLCAP_INSTANCED_ARRAYS_EXT = 4,
    GLCAP_ES2                  = 8,
    GLCAP_ES3                  = 16,
    GLCAP_GAMMA                = 32, // Proper gamma support
    GLCAP_INSTANCES            = 64,
    GLCAP_INSTANCED_ARRAYS     = 128,
    GLCAP_VAO                  = 256,
  };
  FG_FORCEINLINE int32_t FastTruncate(float f) noexcept
  {
#if defined(FG_PLATFORM_WIN32) && defined(BSS_CPU_x86_64)
    return _mm_cvtt_ss2si(_mm_load_ss(&f));
//#elif defined(BSS_CPU_x86_64)
    // GCC's libstdc++ is broken and doesn't have __builtin_ia32_loadss
    //return __builtin_ia32_cvttss2si(__builtin_ia32_loadss(&f));
#else
    return (int32_t)f;
#endif
  }

  // A context may or may not have an associated OS window, for use inside other 3D engines.
  struct Context
  {
    FG_COMPILER_DLLEXPORT explicit Context(FG_Vec2 dim);
    ~Context();
    FG_COMPILER_DLLEXPORT GLExpected<void> BeginDraw(const FG_Rect* area);
    FG_COMPILER_DLLEXPORT GLExpected<void> EndDraw();
    FG_COMPILER_DLLEXPORT GLExpected<void> Resize(FG_Vec2 dim);
    GLExpected<void> DrawArrays(uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex, uint32_t startinstance);
    GLExpected<void> DrawIndexed(uint32_t indexcount, uint32_t instancecount, uint32_t startindex, int startvertex,
                                 uint32_t startinstance);
    GLExpected<void> DrawMesh(uint32_t start, uint32_t count);
    GLExpected<void> Dispatch();
    GLExpected<void> Barrier(GLbitfield barrier_flags);
    GLExpected<void> SetShaderUniforms(const FG_ShaderParameter* uniforms, const FG_ShaderValue* values, uint32_t count);
    GLExpected<void> CopySubresource(FG_Resource src, FG_Resource dest, unsigned long srcoffset, unsigned long destoffset,
                                     unsigned long bytes);
    GLExpected<void> CopyResourceRegion(FG_Resource src, FG_Resource dest, int level, FG_Vec3i srcoffset,
                                        FG_Vec3i destoffset, FG_Vec3i size);
    GLExpected<void> ApplyBlendFactor(const std::array<float, 4>& factor);
    GLExpected<void> ApplyBlend(const FG_Blend& blend, bool force = false);
    GLExpected<void> ApplyFlags(uint16_t flags);
    GLExpected<void> ApplyFill(uint8_t fill);
    GLExpected<void> ApplyCull(uint8_t cull);
    GLExpected<void> SetViewports(std::span<FG_Viewport> viewports);
    GLExpected<void> SetScissors(std::span<FG_Rect> rects);
    GLExpected<void> Clear(uint8_t clearbits, FG_Color16 RGBA, uint8_t stencil, float depth, std::span<FG_Rect> rects);
    void ApplyWorkGroup(FG_Vec3i workgroup) { _workgroup = workgroup; }
    void ApplyDim(FG_Vec2 dim) { _dim = dim; }
    inline void ApplyIndextype(GLenum indextype) { _indextype = indextype; }
    inline void ApplyPrimitive(GLenum primitive) { _primitive = primitive; }
    GLExpected<void> ApplyProgram(const ProgramObject& program);
    void FlipFlag(int diff, int flags, int flag, int option);
    static inline void ColorFloats(const FG_Color8& c, std::array<float, 4>& colors, bool linearize)
    {
      colors[0] = c.r / 255.0f;
      colors[1] = c.g / 255.0f;
      colors[2] = c.b / 255.0f;
      colors[3] = c.a / 255.0f;

      if(linearize)
        for(int i = 0; i < 4; ++i)
          colors[i] = ToLinearRGB(colors[i]);
    }

    static inline void ColorFloats(const FG_Color16& c, std::array<float, 4>& colors, bool linearize)
    {
      colors[0] = c.r / 65535.0f;
      colors[1] = c.g / 65535.0f;
      colors[2] = c.b / 65535.0f;
      colors[3] = c.a / 65535.0f;

      if(linearize)
        for(int i = 0; i < 4; ++i)
          colors[i] = ToLinearRGB(colors[i]);
    }

    static int GetBytes(GLenum type);
    static inline int GetMatrixCount(int length, int width) { return length * (!width ? 1 : width); }
    static inline float ToLinearRGB(float sRGB)
    {
      return (sRGB <= 0.04045f) ? sRGB / 12.92f : powf((sRGB + 0.055f) / 1.055f, 2.4f);
    }
    static inline float ToSRGB(float linearRGB)
    {
      return linearRGB <= 0.0031308f ? linearRGB * 12.92f : 1.055f * powf(linearRGB, 1.0f / 2.4f) - 0.055f;
    }

    static inline GLExpected<void> CallWithRect(const FG_Rect& r, void (*func)(int, int, int, int), const char* callsite)
    {
      (*func)(FastTruncate(::floorf(r.left)), FastTruncate(::floorf(r.top)),
              FastTruncate(::ceilf(r.right - r.left)), FastTruncate(::ceilf(r.bottom - r.top)));
      GL_ERROR(callsite);
      return {};
    }

    static const FG_Blend Normal_Blend;      // For straight-alpha blending
    static const FG_Blend Premultiply_Blend; // For premultiplied blending (the default)
    static const FG_Blend Default_Blend;     // OpenGL default settings

    struct GLState
    {
      GLboolean blend;
      GLboolean tex2d;
      GLboolean framebuffer_srgb;
      GLboolean cullface;
      char blendmask;
      GLint frontface;
      GLint polymode[2];
      GLfloat blendcolor[4];
      GLint colorsrc;
      GLint colordest;
      GLint colorop;
      GLint alphasrc;
      GLint alphadest;
      GLint alphaop;
    } _statestore;
    const ProgramObject* _program;

  protected:
    template<class T> inline static void _buildPosUV(T (&v)[4], const FG_Rect& area, const FG_Rect& uv, float x, float y)
    {
      v[0].posUV[0] = area.left;
      v[0].posUV[1] = area.top;
      v[0].posUV[2] = uv.left / x;
      v[0].posUV[3] = uv.top / y;

      v[1].posUV[0] = area.right;
      v[1].posUV[1] = area.top;
      v[1].posUV[2] = uv.right / x;
      v[1].posUV[3] = uv.top / y;

      v[2].posUV[0] = area.left;
      v[2].posUV[1] = area.bottom;
      v[2].posUV[2] = uv.left / x;
      v[2].posUV[3] = uv.bottom / y;

      v[3].posUV[0] = area.right;
      v[3].posUV[1] = area.bottom;
      v[3].posUV[2] = uv.right / x;
      v[3].posUV[3] = uv.bottom / y;
    }

    GLenum _primitive;
    GLenum _indextype;
    FG_Blend _lastblend;
    std::array<float, 4> _lastfactor;
    uint16_t _lastflags;
    uint8_t _lastcull;
    uint8_t _lastfill;
    FG_Vec2 _dim;
    FG_Vec3i _workgroup;
    FG_Rect _lastscissor;
  };
}

#endif

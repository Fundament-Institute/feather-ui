// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "compiler.hpp"
#include "glad/glad.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "backend.h"
#include "ShaderObject.hpp"
#include <math.h>
#include <vector>
#include <array>
#include <utility>
#include <memory>

namespace GL {
  class Backend;
  struct ProgramObject;

  enum class GLCaps
  {
    GLCAP_GAMMA_EXT = 1,
    GLCAP_INSTANCES_EXT = 2,
    GLCAP_INSTANCED_ARRAYS_EXT = 4,
    GLCAP_ES2 = 8,
    GLCAP_ES3 = 16,
    GLCAP_GAMMA = 32, // Proper gamma support
    GLCAP_INSTANCES = 64,
    GLCAP_INSTANCED_ARRAYS = 128,
    GLCAP_VAO = 256,
  };

  // A context may or may not have an associated OS window, for use inside other 3D engines.
  struct Context : FG_Window
  {
    Context(Backend* backend, FG_Element* element, FG_Vec2 dim);
    virtual ~Context();
    GLExpected<void> BeginDraw(const FG_Rect* area);
    GLExpected<void> EndDraw();
    void Draw(const FG_Rect* area);
    GLExpected<void> DrawArrays(uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex, uint32_t startinstance);
    GLExpected<void> DrawIndexed(uint32_t indexcount, uint32_t instancecount, uint32_t startindex, int startvertex,
                                 uint32_t startinstance);
    GLFWwindow* GetWindow() const { return _window; }
    GLExpected<void> SetShaderUniforms(const FG_ShaderParameter* uniforms, const FG_ShaderValue* values, uint32_t count);
    GLExpected<void> ApplyBlend(const FG_Blend& blend, const std::array<float, 4>& factor, bool force = false);
    GLExpected<void> ApplyFlags(uint16_t flags, uint8_t cull, uint8_t fill);
    void ApplyDim(FG_Vec2 dim) { _dim = dim; }
    void ApplyPrimitiveShader(GLenum primitive, const ProgramObject& program, GLenum indextype)
    {
      _primitive = primitive;
      _program   = &program;
      _indextype = indextype;
    }
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

    static inline void ColorFloats(const FG_Color& c, std::array<float, 4>& colors, bool linearize)
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
    static inline int GetMultiCount(int length, int multi) { return length * (!multi ? 1 : multi); }
    static inline float ToLinearRGB(float sRGB)
    {
      return (sRGB <= 0.04045f) ? sRGB / 12.92f : powf((sRGB + 0.055f) / 1.055f, 2.4f);
    }
    static inline float ToSRGB(float linearRGB)
    {
      return linearRGB <= 0.0031308f ? linearRGB * 12.92f : 1.055f * powf(linearRGB, 1.0f / 2.4f) - 0.055f;
    }

    static const FG_Blend NORMAL_BLEND;      // For straight-alpha blending
    static const FG_Blend PREMULTIPLY_BLEND;     // For premultiplied blending (the default)
    static const FG_Blend DEFAULT_BLEND;     // OpenGL default settings

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

    GLFWwindow* _window;
    Backend* _backend;
    GLenum _primitive;
    GLenum _indextype;
    const ProgramObject* _program;
    FG_Blend _lastblend;
    std::array<float, 4> _lastfactor;
    uint16_t _lastflags;
    uint8_t _lastcull;
    uint8_t _lastfill;
    FG_Vec2 _dim;
  };
}

#endif

// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "compiler.h"
#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "khash.h"
#include "Layer.h"
#include "Shader.h"
#include "Asset.h"
#include "Signature.h"
#include <math.h>
#include <vector>
#include <utility>
#include <memory>

namespace GL {
  class Backend;
  struct Asset;
  struct Font;
  struct Glyph;

  typedef int FG_Err;
  typedef std::pair<const Signature*, GLuint> ShaderInput;

  KHASH_DECLARE(tex, const Asset*, GLuint);
  KHASH_DECLARE(shader, const Shader*, GLuint);
  KHASH_DECLARE(vao, ShaderInput, VAO*);
  KHASH_DECLARE(font, const Font*, uint64_t);
  KHASH_DECLARE(glyph, uint32_t, char);

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
    Context(Backend* backend, FG_MsgReceiver* element, FG_Vec* dim);
    virtual ~Context();
    void BeginDraw(const FG_Rect* area);
    void EndDraw();
    void Draw(const FG_Rect* area);
    FG_Err DrawTextureQuad(GLuint tex, ImageVertex* v, FG_Color color, mat4x4 transform, bool linearize);
    FG_Err DrawTextGL(FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color color, float blur, float rotate, float z,
                      bool linearize);
    FG_Err DrawAsset(FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color, float time, float rotate, float z,
                     bool linearize);
    FG_Err DrawRect(FG_Rect& area, FG_Rect& corners, FG_Color fillColor, float border, FG_Color borderColor, float blur,
                    FG_Asset* asset, float rotate, float z, bool linearize);
    FG_Err DrawCircle(FG_Rect& area, FG_Color fillColor, float border, FG_Color borderColor, float blur, float innerRadius,
                      float innerBorder, FG_Asset* asset, float z, bool linearize);
    FG_Err DrawArc(FG_Rect& area, FG_Vec angles, FG_Color fillColor, float border, FG_Color borderColor, float blur,
                   float innerRadius, FG_Asset* asset, float z, bool linearize);
    FG_Err DrawTriangle(FG_Rect& area, FG_Rect& corners, FG_Color fillColor, float border, FG_Color borderColor, float blur,
                        FG_Asset* asset, float rotate, float z, bool linearize);
    FG_Err DrawLines(FG_Vec* points, uint32_t count, FG_Color color, bool linearize);
    FG_Err DrawCurve(FG_Vec* anchors, uint32_t count, FG_Color fillColor, float stroke, FG_Color strokeColor,
                     bool linearize);
    FG_Err DrawShader(FG_Shader* fgshader, uint8_t primitive, Signature* input, GLsizei count, FG_ShaderValue* values);
    void PushClip(const FG_Rect& rect);
    void PopClip();
    Layer* CreateLayer(const FG_Vec* area, int flags);
    int PushLayer(Layer* layer, float* transform, float opacity, FG_BlendState* blend);
    int PopLayer();
    void CreateResources();
    void DestroyResources();
    GLFWwindow* GetWindow() const { return _window; }
    void Scissor(const FG_Rect& rect, float x, float y) const;
    inline void Viewport(float w, float h) const { Viewport(static_cast<int>(ceilf(w)), static_cast<int>(ceilf(h))); }
    void Viewport(int w, int h) const;
    void StandardViewport() const;
    void AppendBatch(const void* vertices, GLsizeiptr bytes, GLsizei count, GLuint bind);
    void AppendBatch(const void* vertices, GLsizeiptr bytes, GLsizei count);
    GLsizei FlushBatch();
    void SetDim(const FG_Vec& dim);
    GLuint LoadAsset(Asset* asset);
    GLuint LoadShader(Shader* shader);
    VAO* LoadSignature(Signature* input, GLuint shader);
    bool CheckGlyph(uint32_t g);
    void AddGlyph(uint32_t g);
    GLuint GetFontTexture(const Font* font);
    bool CheckFlush(GLintptr bytes) { return (_bufferoffset + bytes > BATCH_BYTES); }
    const FG_BlendState& ApplyBlend(const FG_BlendState* blend, bool force = false);
    void FlipFlag(int diff, int flags, int flag, int option);
    virtual void DirtyRect(const FG_Rect* rect) {}
    inline Backend* GetBackend() const { return _backend; }
    mat4x4& GetProjection() { return _layers.size() > 0 ? _layers.back()->proj : proj; }
    void SetDefaultState();

    inline void ColorFloats(const FG_Color& c, float (&colors)[4], bool linearize)
    {
      colors[0] = c.r / 255.0f;
      colors[1] = c.g / 255.0f;
      colors[2] = c.b / 255.0f;
      colors[3] = c.a / 255.0f;

      if(linearize)
        for(int i = 0; i < 4; ++i)
          colors[i] = ToLinearRGB(colors[i]);
    }

    static GLenum BlendOp(uint8_t op);
    static GLenum BlendValue(uint8_t value);
    static int GetBytes(GLenum type);
    static inline int GetMultiCount(int length, int multi) { return length * (!multi ? 1 : multi); }
    static mat4x4& GetRotationMatrix(mat4x4& m, float rotate, float z, mat4x4& proj);
    static void GenTransform(mat4x4 target, const FG_Rect& area, float rotate, float z);
    static GLenum GetShaderType(uint8_t type);
    static inline float ToLinearRGB(float sRGB)
    {
      return (sRGB <= 0.04045f) ? sRGB / 12.92f : powf((sRGB + 0.055f) / 1.055f, 2.4f);
    }
    static inline float ToSRGB(float linearRGB)
    {
      return linearRGB <= 0.0031308f ? linearRGB * 12.92f : 1.055f * powf(linearRGB, 1.0f / 2.4f) - 0.055f;
    }

    mat4x4 proj;
    FG_MsgReceiver* _element;
    GLuint _imageshader;
    GLuint _rectshader;
    GLuint _circleshader;
    GLuint _arcshader;
    GLuint _trishader;
    GLuint _lineshader;
    std::unique_ptr<VAO> _quadobject;
    GLuint _quadbuffer;
    std::unique_ptr<VAO> _imageobject;
    GLuint _imagebuffer;
    GLuint _imageindices;
    std::unique_ptr<VAO> _lineobject;
    GLuint _linebuffer;
    FG_BlendState _lastblend;

    static const size_t BATCH_BYTES = (1 << 14);
    static const size_t MAX_INDICES = BATCH_BYTES / sizeof(GLuint);
    static const FG_BlendState NORMAL_BLEND;      // For straight-alpha blending
    static const FG_BlendState PREMULTIPLY_BLEND; // For premultiplied blending (the default)
    static const FG_BlendState DEFAULT_BLEND;     // OpenGL default settings

    static const int SOIL_FLAG_LINEAR_RGB = 1024;

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
    GLuint _createBuffer(size_t stride, size_t count, const void* init);
    GLuint _genIndices(size_t num);
    void _flushbatchdraw(Font* font);
    void _drawStandard(GLuint shader, VAO* vao, mat4x4 proj, const FG_Rect& area, const FG_Rect& corners,
                       FG_Color fillColor, float border, FG_Color borderColor, float blur, float rotate, float z,
                       bool linearize);
    unsigned int _createTexture(const unsigned char* const data, int width, int height, int channels,
                                unsigned int reuse_texture_ID, unsigned int flags, unsigned int opengl_texture_type,
                                unsigned int opengl_texture_target, unsigned int texture_check_size_enum);

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
    std::vector<FG_Rect> _clipstack;
    std::vector<Layer*> _layers;
    GLintptr _bufferoffset;
    GLsizei _buffercount;
    kh_tex_s* _texhash;
    kh_font_s* _fonthash;   // Holds the initialized texture for this font on this context
    kh_glyph_s* _glyphhash; // The set of all glyphs that have been initialized
    kh_shader_s* _shaderhash;
    kh_vao_s* _vaohash;
    bool _initialized;
    bool _clipped;
  };
}

#endif

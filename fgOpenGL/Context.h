// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "compiler.h"
#include "khash.h"
#include "Layer.h"
#include "Shader.h"
#include "Asset.h"
#include <vector>
#include <utility>

namespace GL {
  class Backend;
  struct Asset;
  struct Font;
  struct Glyph;

  typedef std::pair<const Shader*, const Asset*> ShaderAsset;

  KHASH_DECLARE(tex, const Asset*, GLuint);
  KHASH_DECLARE(shader, const Shader*, GLuint);
  KHASH_DECLARE(vao, ShaderAsset, GLuint);
  KHASH_DECLARE(font, const Font*, uint64_t);
  KHASH_DECLARE(glyph, uint32_t, char);

  // A context may or may not have an associated OS window, for use inside other 3D engines.
  struct Context
  {
    Context(Backend* backend, FG_Element* element, FG_Vec* dim);
    virtual ~Context();
    void BeginDraw(const FG_Rect* area, bool clear);
    void EndDraw();
    void Draw(const FG_Rect* area);
    void PushClip(const FG_Rect& rect);
    void PopClip();
    void PushLayer(const FG_Rect& area, float* transform, float opacity, Layer* cache);
    Layer* PopLayer();
    void CreateResources();
    void DestroyResources();
    GLFWwindow* GetWindow() const { return _window; }
    void Scissor(const FG_Rect& rect, float x, float y) const;
    void Viewport(const FG_Rect& rect, float x, float y) const;
    void AppendBatch(const void* vertices, GLsizeiptr bytes, GLsizei count);
    GLsizei FlushBatch();
    void SetDim(const FG_Vec& dim);
    GLuint LoadAsset(Asset* asset);
    GLuint LoadShader(Shader* shader);
    GLuint LoadVAO(Shader* shader, Asset* asset);
    bool CheckGlyph(uint32_t g);
    void AddGlyph(uint32_t g);
    GLuint GetFontTexture(const Font* font);
    bool CheckFlush(GLintptr bytes) { return (_bufferoffset + bytes > BATCH_BYTES); }
    void ApplyBlend(const FG_BlendState* blend);

    static GLenum BlendOp(uint8_t op);
    static GLenum BlendValue(uint8_t value);
    static int GetBytes(GLenum type);
    static inline int GetMultiCount(int length, int multi) { return length * (!multi ? 1 : multi); }

    FG_Element* _element;
    GLuint _imageshader;
    GLuint _rectshader;
    GLuint _circleshader;
    GLuint _trishader;
    GLuint _lineshader;
    GLuint _quadobject;
    GLuint _quadbuffer;
    GLuint _imageobject;
    GLuint _imagebuffer;
    GLuint _imageindices;
    GLuint _lineobject;
    GLuint _linebuffer;
    float proj[4][4];
    FG_BlendState _lastblend;

    static const size_t BATCH_BYTES = (1 << 14);
    static const size_t MAX_INDICES = BATCH_BYTES / sizeof(GLuint);
    static const FG_BlendState DEFAULT_BLEND;

  protected:
    GLuint _createVAO(GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
                      size_t stride, GLuint indices);
    GLuint _createBuffer(size_t stride, size_t count, const void* init);
    GLuint _genIndices(size_t num);

    GLFWwindow* _window;
    Backend* _backend;
    std::vector<FG_Rect> _clipstack;
    std::vector<Layer*> _layers;
    GLintptr _bufferoffset;
    GLsizei _buffercount;
    bool _initialized;
    kh_tex_s* _texhash;
    kh_font_s* _fonthash;   // Holds the initialized texture for this font on this context
    kh_glyph_s* _glyphhash; // The set of all glyphs that have been initialized
    kh_shader_s* _shaderhash;
    kh_vao_s* _vaohash;
  };
}

#endif

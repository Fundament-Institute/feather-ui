// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "compiler.h"
#include "khash.h"
#include "Layer.h"
#include "Shader.h"
#include "Asset.h"
#include <vector>

namespace GL {
  class Backend;
  struct Asset;

  // A context may or may not have an associated OS window, for use inside other 3D engines.
  struct Context
  {
    Context(Backend* backend, FG_Element* element);
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
    void BeginBatch();
    void AppendBatch(const void* vertices, size_t bytes, int count);
    void FlushBatch(Shader* shader, unsigned int instance, int primitive, const Attribute* data, size_t n_data);

    FG_Element* _element;
    unsigned int _imageshader;
    unsigned int _rectshader;
    unsigned int _circleshader;
    unsigned int _trishader;

    static const size_t BATCH_BYTES = (1 << 14);

  protected:
    GLFWwindow* _window;
    Backend* _backend;
    std::vector<FG_Rect> _clipstack;
    std::vector<Layer*> _layers;
    unsigned int _batchbuffer;
    size_t _bufferoffset;
    int _buffercount;
    int _bufferstride;
    bool _initialized;
  };
}

#endif

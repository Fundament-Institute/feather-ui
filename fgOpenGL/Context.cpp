// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "Backend.h"
#include "linmath.h"
#include <algorithm>
#include <assert.h>

using namespace GL;

Context::Context(Backend* backend, FG_Element* element) :
  _backend(backend), _element(element), _window(nullptr), _initialized(false)
{}
Context::~Context()
{
  if(_initialized)
    DestroyResources();
}

static const struct
{
  float x, y;
  float r, g, b;
} global_vertices[3] = { { -0.6f, -0.4f, 1.f, 0.f, 0.f }, { 0.6f, -0.4f, 0.f, 1.f, 0.f }, { 0.f, 0.6f, 0.f, 0.f, 1.f } };

void Context::BeginDraw(const FG_Rect* area, bool clear)
{
  if(_window)
  {
    glfwMakeContextCurrent(_window);

    GLsizei w;
    GLsizei h;
    glfwGetFramebufferSize(_window, &w, &h);
    glViewport(0, 0, w, h);
    _backend->LogError("glViewport");
  }

  if(clear)
  {
    glClearColor(0, 0, 0, 1.0f);
    _backend->LogError("glClearColor");
    glClear(GL_COLOR_BUFFER_BIT);
    _backend->LogError("glClear");
  }
}
void Context::EndDraw()
{
  mat4x4 m, p, mvp;

  int width, height;
  glfwGetFramebufferSize(_window, &width, &height);
  float ratio = width / (float)height;

  mat4x4_identity(m);
  mat4x4_rotate_Z(m, m, (float)glfwGetTime());
  mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
  mat4x4_mul(mvp, p, m);

  ratio = width / (float)height;

  Attribute MVP("MVP", GL_FLOAT_MAT4, (float*)mvp);

  BeginBatch();
  AppendBatch(global_vertices, sizeof(global_vertices), 3);
  FlushBatch(&_backend->_imageshader, _imageshader, GL_TRIANGLES, &MVP, 1);

  if(_window)
    glfwSwapBuffers(_window);
}

void Context::Draw(const FG_Rect* area)
{
  FG_Msg msg    = { FG_Kind_DRAW };
  msg.draw.data = this;
  _backend->BeginDraw(_backend, this, &msg.draw.area, true);
  _backend->Behavior(this, msg);
  _backend->EndDraw(_backend, this);
}

void Context::PushClip(const FG_Rect& rect)
{
  if(_clipstack.empty())
    _clipstack.push_back(rect);
  else // Push intersection of previous clip rect with new clip rect
  {
    auto cur   = _clipstack.back();
    cur.left   = std::max(cur.left, rect.left);
    cur.top    = std::max(cur.top, rect.top);
    cur.right  = std::min(cur.right, rect.right);
    cur.bottom = std::min(cur.bottom, rect.bottom);
    _clipstack.push_back(cur);
  }
  Scissor(_clipstack.back(), 0, 0);
}
void Context::Scissor(const FG_Rect& rect, float x, float y) const
{
  int l = static_cast<int>(floorf(rect.left - x));
  int t = static_cast<int>(floorf(rect.top - y));
  int r = static_cast<int>(ceilf(rect.right - x));
  int b = static_cast<int>(ceilf(rect.bottom - y));
  glScissor(l, t, r - l, b - t);
  _backend->LogError("glScissor");
}
void Context::Viewport(const FG_Rect& rect, float x, float y) const
{
  int l = static_cast<int>(floorf(rect.left - x));
  int t = static_cast<int>(floorf(rect.top - y));
  int r = static_cast<int>(ceilf(rect.right - x));
  int b = static_cast<int>(ceilf(rect.bottom - y));
  glViewport(l, t, r - l, b - t);
  _backend->LogError("glViewport");
}
void Context::PopClip()
{
  _clipstack.pop_back();
  Scissor(_clipstack.back(), 0, 0);
}
void Context::PushLayer(const FG_Rect& area, float* transform, float opacity, Layer* layer)
{
  if(!layer)
    layer = new Layer(area, transform, opacity, _window);
  else
    layer->Update(area, transform, opacity, _window);

  _layers.push_back(layer);
  if(layer->dirty.left != NAN || layer->dirty.top != NAN || layer->dirty.right != NAN || layer->dirty.bottom != NAN)
    _clipstack.push_back(layer->dirty); // We override the clip stack here with our new clipping rect without intersection
  else
    _clipstack.push_back({ 0, 0, 0, 0 });

  auto& r = _clipstack.back();
  Scissor(_clipstack.back(), 0, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, layer->framebuffer);
  _backend->LogError("glBindFramebuffer");
  Viewport(layer->area, layer->area.left, layer->area.top);
}

void Context::BeginBatch()
{
  glBindBuffer(GL_ARRAY_BUFFER, _batchbuffer);
  _backend->LogError("glBindBuffer");
  _bufferoffset = 0;
  _buffercount  = 0;
  _bufferstride = 0;
}
void Context::AppendBatch(const void* vertices, size_t bytes, int count)
{
  glBufferSubData(GL_ARRAY_BUFFER, _bufferoffset, bytes, vertices);
  _backend->LogError("glBufferSubData");
  _bufferoffset += bytes;
  _buffercount += count;
  if(!_bufferstride)
    _bufferstride = bytes / count;
  else
    assert(_bufferstride == (bytes / count));
}

void Context::FlushBatch(Shader* shader, GLuint instance, int primitive, const Attribute* data, size_t n_data)
{
  _backend->DrawBoundBuffer(shader, instance, _bufferstride, _buffercount, data, n_data, primitive);
}

Layer* Context::PopLayer()
{
  auto p = _layers.back();
  _layers.pop_back();
  p->dirty = {
    NAN,
    NAN,
    NAN,
    NAN,
  };
  PopClip();
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  _backend->LogError("glBindFramebuffer");

  return p;
}

void Context::CreateResources()
{
  _imageshader = _backend->_imageshader.Create(_backend);
  //_rectshader   = _backend->_rectshader.Create(_backend);
  //_circleshader = _backend->_circleshader.Create(_backend);
  //_trishader    = _backend->_trishader.Create(_backend);

  glGenBuffers(1, &_batchbuffer);
  _backend->LogError("glGenBuffers");
  glBindBuffer(GL_ARRAY_BUFFER, _batchbuffer);
  _backend->LogError("glBindBuffer");
  glBufferData(GL_ARRAY_BUFFER, BATCH_BYTES, 0, GL_DYNAMIC_DRAW);
  _backend->LogError("glBufferData");
  _initialized = true;
}
void Context::DestroyResources()
{
  _backend->_imageshader.Destroy(_backend, _imageshader);
  //_backend->_imageshader.Destroy(_backend, _rectshader);
  //_backend->_imageshader.Destroy(_backend, _circleshader);
  //_backend->_imageshader.Destroy(_backend, _trishader);

  glDeleteBuffers(1, &_batchbuffer);
  _backend->LogError("glDeleteBuffers");
  _initialized = false;
}
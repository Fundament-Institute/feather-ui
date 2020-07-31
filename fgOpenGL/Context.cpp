// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "Backend.h"
#include "linmath.h"
#include "SOIL/SOIL.h"
#include "Font.h"
#include <algorithm>
#include <assert.h>

namespace GL {
  __KHASH_IMPL(tex, , const Asset*, GLuint, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(font, , const Font*, uint64_t, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(glyph, , uint32_t, char, 0, kh_int_hash_func2, kh_int_hash_equal);
}

using namespace GL;

Context::Context(Backend* backend, FG_Element* element, FG_Vec* dim) :
  _backend(backend),
  _element(element),
  _window(nullptr),
  _initialized(false),
  _buffercount(0),
  _bufferoffset(0),
  _texhash(kh_init_tex()),
  _fonthash(kh_init_font()),
  _glyphhash(kh_init_glyph())
{
  if(dim)
    SetDim(*dim);
}
Context::~Context()
{
  if(_initialized)
    DestroyResources();
  kh_destroy_tex(_texhash);
}

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
    glClearColor(0, 0, 0, 0);
    _backend->LogError("glClearColor");
    glClear(GL_COLOR_BUFFER_BIT);
    _backend->LogError("glClear");
  }
}
void Context::EndDraw()
{
  if(_window)
    glfwSwapBuffers(_window);
}
void Context::SetDim(const FG_Vec& dim) { mat4x4_ortho(proj, 0, dim.x, dim.y, 0, 1, -1); }
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

void Context::AppendBatch(const void* vertices, GLsizeiptr bytes, GLsizei count)
{
  glBufferSubData(GL_ARRAY_BUFFER, _bufferoffset, bytes, vertices);
  _backend->LogError("glBufferSubData");
  _bufferoffset += bytes;
  _buffercount += count;
}

GLsizei Context::FlushBatch()
{
  GLsizei count = _buffercount;
  _bufferoffset = 0;
  _buffercount  = 0;
  return count;
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

void Context::_createVAO(Shader& shader, GLuint instance, GLuint* object, GLuint* buffer, const void* init, size_t size,
                         size_t count, GLuint* indices, size_t num)
{
  glGenVertexArrays(1, object);
  _backend->LogError("glGenVertexArrays");
  glGenBuffers(1, buffer);
  _backend->LogError("glGenBuffers");
  glBindVertexArray(*object);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, *buffer);
  _backend->LogError("glBindBuffer");
  glBufferData(GL_ARRAY_BUFFER, size * count, init, !init ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  _backend->LogError("glBufferData");
  shader.SetVertices(_backend, instance, size);

  if(indices)
  {
    glGenBuffers(1, indices);
    _backend->LogError("glGenBuffers");
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *indices);
    _backend->LogError("glBindBuffer");
    std::unique_ptr<GLuint[]> buf(new GLuint[num]);

    for(size_t i = 5, k = 0; i < num; i += 6, k += 4)
    {
      buf[i - 5] = k + 0;
      buf[i - 4] = k + 1;
      buf[i - 3] = k + 2;
      buf[i - 2] = k + 1;
      buf[i - 1] = k + 2;
      buf[i - 0] = k + 3;
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, num * sizeof(GLuint), buf.get(), GL_STATIC_DRAW);
    _backend->LogError("glBufferData");
  }
  glBindVertexArray(0);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
}
void Context::CreateResources()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  _imageshader  = _backend->_imageshader.Create(_backend);
  _rectshader   = _backend->_rectshader.Create(_backend);
  _circleshader = _backend->_circleshader.Create(_backend);
  _trishader    = _backend->_trishader.Create(_backend);
  _lineshader   = _backend->_trishader.Create(_backend);

  QuadVertex rect[4] = {
    { 0, 0 },
    { 1, 0 },
    { 0, 1 },
    { 1, 1 },
  };
  CreateVAO(_backend->_rectshader, _rectshader, &_quadobject, &_quadbuffer, rect);
  _createVAO(_backend->_imageshader, _imageshader, &_imageobject, &_imagebuffer, nullptr, sizeof(ImageVertex),
             BATCH_BYTES / sizeof(ImageVertex), &_imageindices, BATCH_BYTES / sizeof(GLuint));

  _createVAO(_backend->_lineshader, _lineshader, &_lineobject, &_linebuffer, nullptr, sizeof(FG_Vec),
             BATCH_BYTES / sizeof(FG_Vec), nullptr, 0);

  for(auto& l : _layers)
    l->Create();

  _initialized = true;
}
GLuint Context::LoadAsset(Asset* asset)
{
  khiter_t iter = kh_get_tex(_texhash, asset);
  if(iter < kh_end(_texhash) && kh_exist(_texhash, iter))
    return kh_val(_texhash, iter);

  GLuint idx;
  /*if(!asset->count)
    idx = SOIL_load_OGL_texture((const char*)asset->data.data, SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);
  else
    idx = SOIL_load_OGL_texture_from_memory((const unsigned char*)asset->data.data, asset->count, SOIL_LOAD_AUTO,
                                            SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);*/
  return 0;
  int r;
  iter = kh_put_tex(_texhash, asset, &r);
  glBindTexture(GL_TEXTURE_2D, idx);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  if(r >= 0)
    kh_val(_texhash, iter) = idx;
  return idx;
}
void Context::DestroyResources()
{
  for(khiter_t i = 0; i < kh_end(_texhash); ++i)
  {
    if(kh_exist(_texhash, i))
    {
      GLuint idx = kh_val(_texhash, i);
      glDeleteTextures(1, &idx);
    }
  }
  kh_clear_tex(_texhash);

  for(khiter_t i = 0; i < kh_end(_fonthash); ++i)
  {
    if(kh_exist(_fonthash, i))
    {
      GLuint idx = static_cast<GLuint>(kh_val(_fonthash, i) & 0xFFFFFFFF);
      glDeleteTextures(1, &idx);
    }
  }
  kh_clear_font(_fonthash);

  kh_clear_glyph(_glyphhash);

  _backend->_imageshader.Destroy(_backend, _imageshader);
  _backend->_imageshader.Destroy(_backend, _rectshader);
  _backend->_imageshader.Destroy(_backend, _circleshader);
  _backend->_imageshader.Destroy(_backend, _trishader);
  _backend->_imageshader.Destroy(_backend, _lineshader);

  glDeleteVertexArrays(1, &_quadobject);
  _backend->LogError("glDeleteVertexArrays");
  glDeleteBuffers(1, &_quadbuffer);
  _backend->LogError("glDeleteBuffers");
  glDeleteVertexArrays(1, &_imageobject);
  _backend->LogError("glDeleteVertexArrays");
  glDeleteBuffers(1, &_imagebuffer);
  _backend->LogError("glDeleteBuffers");
  glDeleteBuffers(1, &_imageindices);
  _backend->LogError("glDeleteBuffers");
  glDeleteVertexArrays(1, &_lineobject);
  _backend->LogError("glDeleteVertexArrays");
  glDeleteBuffers(1, &_linebuffer);
  _backend->LogError("glDeleteBuffers");

  for(auto& l : _layers)
    l->Destroy();

  _initialized = false;
}

bool Context::CheckGlyph(uint32_t g)
{
  auto i = kh_get_glyph(_glyphhash, g);
  return i < kh_end(_glyphhash) && kh_exist(_glyphhash, i);
}
void Context::AddGlyph(uint32_t g)
{
  int r;
  kh_put_glyph(_glyphhash, g, &r);
}

GLuint Context::GetFontTexture(const Font* font)
{
  int r;
  auto i      = kh_put_font(_fonthash, font, &r);
  int powsize = font->GetSizePower();
  GLuint tex;
  if(!r)
  {
    auto pair = kh_val(_fonthash, i);
    tex       = uint32_t(pair & 0xFFFFFFFF);
    auto size = uint32_t(pair >> 32);
    if(powsize <= size)
      return tex;
    powsize = size;
  }
  else if(r < 0)
    return 0;
  else
  {
    glGenTextures(1, &tex);
    _backend->LogError("glGenTextures");
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  _backend->LogError("glBindTexture");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  _backend->LogError("glTexParameteri");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _backend->LogError("glTexParameteri");

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (1 << powsize), (1 << powsize), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  _backend->LogError("glTexImage2D");
  glBindTexture(GL_TEXTURE_2D, 0);
  _backend->LogError("glBindTexture");

  kh_val(_fonthash, i) = tex | (uint64_t(powsize) << 32);
  return tex;
}
// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include "linmath.h"
#include "SOIL.h"
#include "Font.h"
#include <algorithm>
#include <assert.h>

#define kh_pair_hash_func(key) \
  kh_int64_hash_func((static_cast<uint64_t>(kh_ptr_hash_func(key.first)) << 32) | kh_ptr_hash_func(key.first))

namespace GL {
  __KHASH_IMPL(tex, , const Asset*, GLuint, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(shader, , const Shader*, GLuint, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(vao, , ShaderAsset, GLuint, 1, kh_pair_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(font, , const Font*, uint64_t, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(glyph, , uint32_t, char, 0, kh_int_hash_func2, kh_int_hash_equal);
}

using namespace GL;

const FG_BlendState Context::DEFAULT_BLEND = {
  FG_BlendValue_SRC_ALPHA,
  FG_BlendValue_INV_SRC_ALPHA,
  FG_BlendOp_ADD,
  FG_BlendValue_SRC_ALPHA,
  FG_BlendValue_INV_SRC_ALPHA,
  FG_BlendOp_ADD,
  0b1111,
};

Context::Context(Backend* backend, FG_Element* element, FG_Vec* dim) :
  _backend(backend),
  _element(element),
  _window(nullptr),
  _initialized(false),
  _buffercount(0),
  _bufferoffset(0),
  _texhash(kh_init_tex()),
  _fonthash(kh_init_font()),
  _glyphhash(kh_init_glyph()),
  _vaohash(kh_init_vao()),
  _shaderhash(kh_init_shader()),
  _lastblend({
    FG_BlendValue_ONE,
    FG_BlendValue_ZERO,
    FG_BlendOp_ADD,
    FG_BlendValue_ONE,
    FG_BlendValue_ZERO,
    FG_BlendOp_ADD,
    0b1111,
  })
{
  if(dim)
    SetDim(*dim);
}
Context::~Context()
{
  if(_initialized)
    DestroyResources();
  kh_destroy_tex(_texhash);
  kh_destroy_font(_fonthash);
  kh_destroy_glyph(_glyphhash);
  kh_destroy_vao(_vaohash);
  kh_destroy_shader(_shaderhash);
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

void mat4x4_custom(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
  memset(M, 0, sizeof(mat4x4));

  M[0][0] = 2.0f / (r - l);
  M[0][1] = 0.f;
  M[0][2] = 0.f;
  M[0][3] = 0.f;

  M[1][0] = 0.f;
  M[1][1] = 2.0f / (t - b);
  M[1][2] = 0.f;
  M[1][3] = 0.f;

  M[2][0] = 0.f;
  M[2][1] = 0.f;
  M[2][2] = -((f + n) / (f - n));
  M[2][3] = -1.0f;

  M[3][0] = -(r + l) / (r - l);
  M[3][1] = -(t + b) / (t - b);
  M[3][2] = -((2.f * f * n) / (f - n));
  M[3][3] = 0.f;

  mat4x4_translate_in_place(M, 0, 0, -1.0f);
}

void Context::SetDim(const FG_Vec& dim)
{
  // mat4x4_ortho(proj, 0, dim.x, dim.y, 0, -1, 1);
  mat4x4_custom(proj, 0, dim.x, dim.y, 0, 0.2, 100);
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

int Context::GetBytes(GLenum type)
{
  switch(type)
  {
  case GL_BYTE:
  case GL_UNSIGNED_BYTE: return 1;
  case GL_SHORT:
  case GL_UNSIGNED_SHORT:
  case GL_HALF_FLOAT: return 2;
  case GL_INT:
  case GL_UNSIGNED_INT:
  case GL_FLOAT: return 4;
  case GL_DOUBLE: return 8;
  }
  assert(false);
  return 0;
}

GLuint Context::_createBuffer(size_t stride, size_t count, const void* init)
{
  GLuint buffer;
  glGenBuffers(1, &buffer);
  _backend->LogError("glGenBuffers");
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  _backend->LogError("glBindBuffer");
  glBufferData(GL_ARRAY_BUFFER, stride * count, init, !init ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  _backend->LogError("glBufferData");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
  return buffer;
}

GLuint Context::_genIndices(size_t num)
{
  GLuint indices;
  glGenBuffers(1, &indices);
  _backend->LogError("glGenBuffers");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
  _backend->LogError("glBindBuffer");
  std::unique_ptr<GLuint[]> buf(new GLuint[num]);

  GLuint k = 0;
  for(size_t i = 5; i < num; i += 6, k += 4)
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
  return indices;
}

GLuint Context::_createVAO(GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
                           size_t stride, GLuint indices)
{
  GLuint object;
  glGenVertexArrays(1, &object);
  _backend->LogError("glGenVertexArrays");
  glBindVertexArray(object);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  _backend->LogError("glBindBuffer");

  if(indices)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
    _backend->LogError("glBindBuffer");
  }

  GLuint offset = 0;
  for(size_t i = 0; i < n_parameters; ++i)
  {
    auto loc = glGetAttribLocation(shader, parameters[i].name);
    _backend->LogError("glGetAttribLocation");
    glEnableVertexAttribArray(loc);
    _backend->LogError("glEnableVertexAttribArray");
    size_t sz   = GetMultiCount(parameters[i].length, parameters[i].multi);
    GLenum type = 0;
    switch(parameters->type)
    {
    case FG_ShaderType_FLOAT: type = GL_FLOAT; break;
    case FG_ShaderType_INT: type = GL_INT; break;
    case FG_ShaderType_UINT: type = GL_UNSIGNED_INT; break;
    }

    glVertexAttribPointer(loc, sz, type, GL_FALSE, stride, (void*)offset);
    offset += GetBytes(type) * sz;
    _backend->LogError("glVertexAttribPointer");
  }

  glBindVertexArray(0);
  _backend->LogError("glBindVertexArray");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _backend->LogError("glBindBuffer");
  if(indices)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  return object;
}
void Context::CreateResources()
{
  glEnable(GL_BLEND);
  _backend->LogError("glEnable");
  glEnable(GL_TEXTURE_2D);
  _backend->LogError("glEnable");
  ApplyBlend(0);
  _backend->LogError("glBlendFunc");

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

  FG_ShaderParameter rectparams[1] = { { FG_ShaderType_FLOAT, 2, 0, "vPos" } };
  FG_ShaderParameter imgparams[2]  = { { FG_ShaderType_FLOAT, 4, 0, "vPosUV" }, { FG_ShaderType_FLOAT, 4, 0, "vColor" } };

  _quadbuffer = _createBuffer(sizeof(QuadVertex), 4, rect);
  _quadobject = _createVAO(_rectshader, rectparams, 1, _quadbuffer, sizeof(QuadVertex), 0);

  _imagebuffer  = _createBuffer(sizeof(ImageVertex), BATCH_BYTES / sizeof(ImageVertex), nullptr);
  _imageindices = _genIndices(BATCH_BYTES / sizeof(GLuint));
  _imageobject  = _createVAO(_imageshader, imgparams, 2, _imagebuffer, sizeof(ImageVertex), _imageindices);

  _linebuffer = _createBuffer(sizeof(FG_Vec), BATCH_BYTES / sizeof(FG_Vec), nullptr);
  _lineobject = _createVAO(_lineshader, rectparams, 1, _linebuffer, sizeof(FG_Vec), 0);

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
  if(asset->format == FG_Format_BUFFER)
  {
    glGenBuffers(1, &idx);
    _backend->LogError("glGenBuffers");

    GLenum kind = GL_ARRAY_BUFFER;
    switch(asset->primitive)
    {
    case FG_Primitive_INDEX_BYTE:
    case FG_Primitive_INDEX_SHORT:
    case FG_Primitive_INDEX_INT: kind = GL_ELEMENT_ARRAY_BUFFER; break;
    }

    glBindBuffer(kind, idx);
    _backend->LogError("glBindBuffer");
    glBufferData(kind, asset->stride * asset->count, asset->data.data, GL_STATIC_DRAW);
    _backend->LogError("glBufferData");
    glBindBuffer(kind, 0);
  }
  else
  {
    idx = SOIL_create_OGL_texture((const unsigned char*)asset->data.data, asset->size.x, asset->size.y, asset->channels,
                                  SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);

    if(!idx)
    {
      _backend->LogError("SOIL_create_OGL_texture");
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "%s failed (returned 0).", "SOIL_create_OGL_texture");
      return 0;
    }

    glBindTexture(GL_TEXTURE_2D, idx);
    _backend->LogError("glBindTexture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _backend->LogError("glTexParameteri");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    _backend->LogError("glTexParameteri");
    glBindTexture(GL_TEXTURE_2D, 0);
    _backend->LogError("glBindTexture");
  }

  int r;
  iter = kh_put_tex(_texhash, asset, &r);

  if(r >= 0)
    kh_val(_texhash, iter) = idx;
  return idx;
}
GLuint Context::LoadShader(Shader* shader)
{
  khiter_t iter = kh_get_shader(_shaderhash, shader);
  if(iter < kh_end(_shaderhash) && kh_exist(_shaderhash, iter))
    return kh_val(_shaderhash, iter);

  GLuint instance = shader->Create(_backend);
  if(!instance)
    return 0;

  int r;
  iter = kh_put_shader(_shaderhash, shader, &r);

  if(r >= 0)
    kh_val(_shaderhash, iter) = instance;
  return instance;
}

GLuint Context::LoadVAO(Shader* shader, Asset* asset)
{
  GLuint instance = LoadShader(shader);
  GLuint buffer   = LoadAsset(asset);
  if(!buffer || !instance || asset->format != FG_Format_BUFFER)
    return 0;

  ShaderAsset pair = { shader, asset };

  khiter_t iter = kh_get_vao(_vaohash, pair);
  if(iter < kh_end(_vaohash) && kh_exist(_vaohash, iter))
    return kh_val(_vaohash, iter);

  GLuint object = _createVAO(instance, asset->parameters, asset->n_parameters, buffer, asset->stride, 0);

  int r;
  iter = kh_put_vao(_vaohash, pair, &r);

  if(r >= 0)
    kh_val(_vaohash, iter) = object;
  return object;
}

void Context::DestroyResources()
{
  for(khiter_t i = 0; i < kh_end(_texhash); ++i)
  {
    if(kh_exist(_texhash, i))
    {
      GLuint idx = kh_val(_texhash, i);
      if(kh_key(_texhash, i)->format == FG_Format_BUFFER)
      {
        glDeleteBuffers(1, &idx);
        _backend->LogError("glDeleteBuffers");
      }
      else
      {
        glDeleteTextures(1, &idx);
        _backend->LogError("glDeleteTextures");
      }
    }
  }
  kh_clear_tex(_texhash);

  for(khiter_t i = 0; i < kh_end(_fonthash); ++i)
  {
    if(kh_exist(_fonthash, i))
    {
      GLuint idx = static_cast<GLuint>(kh_val(_fonthash, i) & 0xFFFFFFFF);
      glDeleteTextures(1, &idx);
      _backend->LogError("glDeleteTextures");
    }
  }
  kh_clear_font(_fonthash);
  kh_clear_glyph(_glyphhash);

  for(khiter_t i = 0; i < kh_end(_shaderhash); ++i)
  {
    if(kh_exist(_shaderhash, i))
      kh_key(_shaderhash, i)->Destroy(_backend, kh_val(_shaderhash, i));
  }
  kh_clear_shader(_shaderhash);

  for(khiter_t i = 0; i < kh_end(_vaohash); ++i)
  {
    if(kh_exist(_vaohash, i))
    {
      GLuint idx = static_cast<GLuint>(kh_val(_fonthash, i) & 0xFFFFFFFF);
      glDeleteVertexArrays(1, &idx);
      _backend->LogError("glDeleteVertexArrays");
    }
  }
  kh_clear_vao(_vaohash);

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
    int size  = uint32_t(pair >> 32);
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

GLenum Context::BlendValue(uint8_t value)
{
  switch(value)
  {
  case FG_BlendValue_DST_COLOR: return GL_DST_COLOR;
  case FG_BlendValue_INV_CONSTANT_ALPHA: return GL_ONE_MINUS_CONSTANT_ALPHA;
  case FG_BlendValue_CONSTANT_COLOR: return GL_CONSTANT_COLOR;
  case FG_BlendValue_SRC_ALPHA_SATURATE: return GL_SRC_ALPHA_SATURATE;
  case FG_BlendValue_CONSTANT_ALPHA: return GL_CONSTANT_ALPHA;
  case FG_BlendValue_INV_SRC_COLOR: return GL_ONE_MINUS_SRC_COLOR;
  case FG_BlendValue_ZERO: return GL_ZERO;
  case FG_BlendValue_INV_SRC_ALPHA: return GL_ONE_MINUS_SRC_ALPHA;
  case FG_BlendValue_INV_CONSTANT_COLOR: return GL_ONE_MINUS_CONSTANT_COLOR;
  case FG_BlendValue_ONE: return GL_ONE;
  case FG_BlendValue_SRC_ALPHA: return GL_SRC_ALPHA;
  case FG_BlendValue_SRC_COLOR: return GL_SRC_COLOR;
  case FG_BlendValue_INV_DST_ALPHA: return GL_ONE_MINUS_DST_ALPHA;
  case FG_BlendValue_DST_ALPHA: return GL_DST_ALPHA;
  case FG_BlendValue_INV_DST_COLOR: return GL_ONE_MINUS_DST_COLOR;
  }
  assert(false);
  return 0;
}

GLenum Context::BlendOp(uint8_t op)
{
  switch(op)
  {
  case FG_BlendOp_ADD: return GL_FUNC_ADD;
  case FG_BlendOp_SUBTRACT: return GL_FUNC_SUBTRACT;
  case FG_BlendOp_REV_SUBTRACT: return GL_FUNC_REVERSE_SUBTRACT;
  }
  assert(false);
  return 0;
}

void Context::ApplyBlend(const FG_BlendState* blend)
{
  if(!blend)
    blend = &DEFAULT_BLEND;

  // This comparison should get optimized out to a 64-bit integer equality check
  if(memcmp(blend, &_lastblend, sizeof(FG_BlendState)) != 0)
  {
    glBlendFuncSeparate(BlendValue(blend->srcBlend), BlendValue(blend->destBlend), BlendValue(blend->srcBlendAlpha),
                        BlendValue(blend->destBlendAlpha));
    glBlendEquationSeparate(BlendOp(blend->colorBlend), BlendOp(blend->alphaBlend));

    if(_lastblend.constant.v != blend->constant.v)
    {
      float colors[4];
      _backend->ColorFloats(blend->constant, colors);
      glBlendColor(colors[0], colors[1], colors[2], colors[3]);
    }

    if(_lastblend.mask != blend->mask)
      glColorMask(blend->mask & 0b0001, blend->mask & 0b0010, blend->mask & 0b0100, blend->mask & 0b1000);

    _lastblend = *blend;
  }
}

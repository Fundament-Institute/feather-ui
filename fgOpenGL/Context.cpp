// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include "linmath.h"
#include "SOIL.h"
#include "Font.h"
#include "VAO.h"
#include <algorithm>
#include <assert.h>

#define kh_pair_hash_func(key) \
  kh_int64_hash_func((static_cast<uint64_t>(kh_ptr_hash_func(key.first)) << 32) | kh_ptr_hash_func(key.first))

namespace GL {
  __KHASH_IMPL(tex, , const Asset*, GLuint, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(shader, , const Shader*, GLuint, 1, kh_ptr_hash_func, kh_int_hash_equal);
  __KHASH_IMPL(vao, , ShaderAsset, VAO*, 1, kh_pair_hash_func, kh_int_hash_equal);
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

Context::Context(Backend* backend, FG_MsgReceiver* element, FG_Vec* dim) :
  _backend(backend),
  _element(element),
  _window(nullptr),
  _buffercount(0),
  _bufferoffset(0),
  _texhash(kh_init_tex()),
  _fonthash(kh_init_font()),
  _glyphhash(kh_init_glyph()),
  _vaohash(kh_init_vao()),
  _shaderhash(kh_init_shader()),
  _initialized(false),
  _clipped(false),
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

void Context::BeginDraw(const FG_Rect* area)
{
  if(_window)
  {
    glfwMakeContextCurrent(_window);
    StandardViewport();
  }

  _clipped = area != nullptr;
  if(_clipped)
    PushClip(*area);
}
void Context::EndDraw()
{
  if(_clipped)
    PopClip();
  _clipped = false;
  if(_window)
    glfwSwapBuffers(_window);
}

void Context::SetDim(const FG_Vec& dim)
{
  // mat4x4_ortho(proj, 0, dim.x, dim.y, 0, -1, 1);
  Layer::mat4x4_proj(proj, 0, dim.x, dim.y, 0, Layer::NEARZ, Layer::FARZ);
  // mat4x4_custom(lproj, 0, dim.x, 0, dim.y, 0.2, 100);
}

void Context::Draw(const FG_Rect* area)
{
  FG_Msg msg    = { FG_Kind_DRAW };
  msg.draw.data = this;
  if(area)
    msg.draw.area = *area;
  else
  {
    GLsizei w;
    GLsizei h;
    glfwGetFramebufferSize(_window, &w, &h);
    msg.draw.area.right  = w;
    msg.draw.area.bottom = h;
  }

  _backend->BeginDraw(_backend, this, &msg.draw.area);
  _backend->Behavior(this, msg);
  _backend->EndDraw(_backend, this);
}

void Context::PushClip(const FG_Rect& rect)
{
  if(_clipstack.empty())
  {
    _clipstack.push_back(rect);
    glEnable(GL_SCISSOR_TEST);
  }
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

void Context::StandardViewport() const
{
  GLsizei w;
  GLsizei h;
  glfwGetFramebufferSize(_window, &w, &h);
  glViewport(0, 0, w, h);
  _backend->LogError("glViewport");
}
void Context::Viewport(float w, float h) const
{
  glViewport(0, 0, static_cast<int>(ceilf(w)), static_cast<int>(ceilf(h)));
  _backend->LogError("glViewport");
}

void Context::PopClip()
{
  _clipstack.pop_back();
  if(!_clipstack.empty())
    Scissor(_clipstack.back(), 0, 0);
  else
    glDisable(GL_SCISSOR_TEST);
}

Layer* Context::CreateLayer(const FG_Vec* psize)
{
  GLsizei w;
  GLsizei h;
  glfwGetFramebufferSize(_window, &w, &h);
  FG_Vec size = { w, h };
  return new Layer(!psize ? size : *psize, this);
}

int Context::PushLayer(Layer* layer, float* transform, float opacity, FG_BlendState* blend)
{
  if(!layer)
    return -1;

  layer->Update(transform, opacity, blend, this);
  _layers.push_back(layer);

  glBindFramebuffer(GL_FRAMEBUFFER, layer->framebuffer);
  _backend->LogError("glBindFramebuffer");
  Viewport(layer->size.x, layer->size.y);
  return 0;
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

int Context::PopLayer()
{
  Layer* p = _layers.back();
  _layers.pop_back();
  glBindFramebuffer(GL_FRAMEBUFFER, !_layers.size() ? 0 : _layers.back()->framebuffer);
  _backend->LogError("glBindFramebuffer");
  if(_layers.size() > 0)
    Viewport(_layers.back()->size.x, _layers.back()->size.y);
  else
    StandardViewport();

  ImageVertex v[4];

  v[0].posUV[0] = 0;
  v[0].posUV[1] = 0;
  v[0].posUV[2] = 0;
  v[0].posUV[3] = 1.0f;

  v[1].posUV[0] = p->size.x;
  v[1].posUV[1] = 0;
  v[1].posUV[2] = 1.0f;
  v[1].posUV[3] = 1.0f;

  v[2].posUV[0] = 0;
  v[2].posUV[1] = p->size.y;
  v[2].posUV[2] = 0;
  v[2].posUV[3] = 0;

  v[3].posUV[0] = p->size.x;
  v[3].posUV[1] = p->size.y;
  v[3].posUV[2] = 1.0f;
  v[3].posUV[3] = 0;

  mat4x4 mvp;
  mat4x4_mul(mvp, GetProjection(), (vec4*)p->transform);
  return _backend->DrawTextureQuad(this, p->data.index, v,
                                   FG_Color{ 0x00FFFFFF + ((unsigned int)roundf(0xFF * p->opacity) << 24) }, (float*)mvp,
                                   nullptr);
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
    buf[i - 1] = k + 3;
    buf[i - 0] = k + 2;
  }
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, num * sizeof(GLuint), buf.get(), GL_STATIC_DRAW);
  _backend->LogError("glBufferData");
  return indices;
}

void Context::CreateResources()
{
  glEnable(GL_BLEND);
  _backend->LogError("glEnable");
  glEnable(GL_TEXTURE_2D);
  _backend->LogError("glEnable");
  glFrontFace(GL_CW);
  _backend->LogError("glFrontFace");
  glEnable(GL_CULL_FACE);
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
  _quadobject = new VAO(_backend, _rectshader, rectparams, 1, _quadbuffer, sizeof(QuadVertex), 0);

  _imagebuffer  = _createBuffer(sizeof(ImageVertex), BATCH_BYTES / sizeof(ImageVertex), nullptr);
  _imageindices = _genIndices(BATCH_BYTES / sizeof(GLuint));
  _imageobject  = new VAO(_backend, _imageshader, imgparams, 2, _imagebuffer, sizeof(ImageVertex), _imageindices);

  _linebuffer = _createBuffer(sizeof(FG_Vec), BATCH_BYTES / sizeof(FG_Vec), nullptr);
  _lineobject = new VAO(_backend, _lineshader, rectparams, 1, _linebuffer, sizeof(FG_Vec), 0);

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

VAO* Context::LoadVAO(Shader* shader, Asset* asset)
{
  GLuint instance = LoadShader(shader);
  GLuint buffer   = LoadAsset(asset);
  if(!buffer || !instance || asset->format != FG_Format_BUFFER)
    return 0;

  ShaderAsset pair = { shader, asset };
  khiter_t iter    = kh_get_vao(_vaohash, pair);
  if(iter < kh_end(_vaohash) && kh_exist(_vaohash, iter))
    return kh_val(_vaohash, iter);

  VAO* object = new VAO(_backend, instance, asset->parameters, asset->n_parameters, buffer, asset->stride, 0);

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

  delete _quadobject;
  _backend->LogError("glDeleteVertexArrays");
  glDeleteBuffers(1, &_quadbuffer);
  _backend->LogError("glDeleteBuffers");
  delete _imageobject;
  glDeleteBuffers(1, &_imagebuffer);
  _backend->LogError("glDeleteBuffers");
  glDeleteBuffers(1, &_imageindices);
  _backend->LogError("glDeleteBuffers");
  delete _lineobject;
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

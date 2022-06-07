// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#include "BackendGL.h"
#include "linmath.h"
#include "VertexArrayObject.h"
#include "EnumMapping.h"
#include <algorithm>
#include <assert.h>
#include <float.h>

#define kh_pair_hash_func(key) \
  kh_int64_hash_func((static_cast<uint64_t>(kh_ptr_hash_func(key.first)) << 32) | kh_int_hash_func(key.second))

using namespace GL;

// Feather uses a premultiplied compositing pipeline:
// https://apoorvaj.io/alpha-compositing-opengl-blending-and-premultiplied-alpha/
const FG_Blend Context::PREMULTIPLY_BLEND = {
  FG_BLEND_ONE,
  FG_BLEND_INV_SRC_ALPHA,
  FG_BLEND_OP_ADD,
  FG_BLEND_ONE,
  FG_BLEND_INV_SRC_ALPHA,
  FG_BLEND_OP_ADD,
  0b1111,
};

const FG_Blend Context::NORMAL_BLEND = {
  FG_BLEND_SRC_ALPHA,
  FG_BLEND_INV_SRC_ALPHA,
  FG_BLEND_OP_ADD,
  FG_BLEND_ONE,
  FG_BLEND_INV_SRC_ALPHA,
  FG_BLEND_OP_ADD,
  0b1111,
};

const FG_Blend Context::DEFAULT_BLEND = { FG_BLEND_ONE,
                                          FG_BLEND_ZERO,
                                          FG_BLEND_OP_ADD,
                                          FG_BLEND_ONE,
                                          FG_BLEND_ZERO,
                                          FG_BLEND_OP_ADD,
                                          0b1111,
                                           };

Context::Context(Backend* backend, FG_Element* element, FG_Vec2* dim) :
  _backend(backend),
  _element(element),
  _window(nullptr),
  _buffercount(0),
  _bufferoffset(0),
  _initialized(false),
  _clipped(false),
  _lastblend(DEFAULT_BLEND)
{
  if(dim)
    SetDim(*dim);
}
Context::~Context()
{
}

GLExpected<void> Context::BeginDraw(const FG_Rect* area)
{
  if(_window)
  {
    glfwMakeContextCurrent(_window);
    StandardViewport();
  }
  else
  {
    // If we don't control the context, store OpenGL state so we don't stomp on it
    glGetBooleanv(GL_BLEND, &_statestore.blend);
    glGetBooleanv(GL_TEXTURE_2D, &_statestore.tex2d);
    glGetBooleanv(GL_FRAMEBUFFER_SRGB, &_statestore.framebuffer_srgb);
    glGetBooleanv(GL_CULL_FACE, &_statestore.cullface);
    glGetIntegerv(GL_FRONT_FACE, &_statestore.frontface);
    glGetIntegerv(GL_POLYGON_MODE, _statestore.polymode);

    GLboolean mask[4];
    glGetBooleanv(GL_COLOR_WRITEMASK, mask);
    _statestore.blendmask = 0;
    for(int i = 0; i < 4; ++i)
      _statestore.blendmask |= (mask[i] << i);

    glGetFloatv(GL_BLEND_COLOR, _statestore.blendcolor);
    glGetIntegerv(GL_BLEND_SRC, &_statestore.colorsrc);
    glGetIntegerv(GL_BLEND_DST, &_statestore.colordest);
    glGetIntegerv(GL_BLEND_EQUATION, &_statestore.colorop);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &_statestore.alphasrc);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &_statestore.alphadest);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &_statestore.alphaop);
  }

  return std::move(SetDefaultState());
}
GLExpected<void> Context::EndDraw()
{
  if(_window)
    glfwSwapBuffers(_window);
  else
  {
    // Restore saved OpenGL state
    FlipFlag(1, _statestore.blend, 1, GL_BLEND);
    GL_ERROR("glEnable");
    FlipFlag(1, _statestore.tex2d, 1, GL_TEXTURE_2D);
    GL_ERROR("glEnable");
    FlipFlag(1, _statestore.framebuffer_srgb, 1, GL_FRAMEBUFFER_SRGB);
    GL_ERROR("glEnable");
    FlipFlag(1, _statestore.cullface, 1, GL_CULL_FACE);
    GL_ERROR("glEnable");
    glFrontFace(_statestore.frontface);
    GL_ERROR("glFrontFace");
    glPolygonMode(GL_FRONT, _statestore.polymode[0]);
    GL_ERROR("glPolygonMode");
    glPolygonMode(GL_BACK, _statestore.polymode[1]);
    GL_ERROR("glPolygonMode");
    glColorMask(_statestore.blendmask & 0b0001, _statestore.blendmask & 0b0010, _statestore.blendmask & 0b0100,
                _statestore.blendmask & 0b1000);
    GL_ERROR("glColorMask");
    glBlendColor(_statestore.blendcolor[0], _statestore.blendcolor[1], _statestore.blendcolor[2],
                 _statestore.blendcolor[3]);
    GL_ERROR("glBlendColor");
    glBlendFuncSeparate(_statestore.colorsrc, _statestore.colordest, _statestore.alphasrc, _statestore.alphadest);
    GL_ERROR("glBlendFuncSeparate");
    glBlendEquationSeparate(_statestore.colorop, _statestore.alphaop);
    GL_ERROR("glBlendEquationSeparate");
  }
}

void Context::Draw(const FG_Rect* area)
{
  FG_Msg msg = { FG_Kind_DRAW };
  if(area)
    msg.draw.area = *area;
  else
  {
    auto vp              = GetViewportf();
    msg.draw.area.right  = vp.x;
    msg.draw.area.bottom = vp.y;
  }

  _backend->BeginDraw(_backend, this, &msg.draw.area);
  _backend->Behavior(this, msg);
  _backend->EndDraw(_backend, this);
}

FG_Err Context::DrawShader(FG_Shader* fgshader, uint8_t primitive, Signature* input, GLsizei count, FG_ShaderValue* values)
{
  auto shader   = static_cast<Shader*>(fgshader);
  auto instance = LoadShader(shader);
  auto vao      = LoadSignature(input, instance);

  glUseProgram(instance);
  GL_ERROR("glUseProgram");
  vao->Bind();

  for(uint32_t i = 0; i < shader->n_parameters; ++i)
  {
    auto type = Shader::GetType(shader->parameters[i]);
    switch(type)
    {
    case GL_DOUBLE:
    case GL_HALF_FLOAT: // we assume you pass in a proper float to fill this
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT: Shader::SetUniform(_backend, instance, shader->parameters[i].name, type, &values[i].f32); break;
    default:
      if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
      {
        GLuint idx = LoadAsset(static_cast<Asset*>(values[i].asset));
        Shader::SetUniform(_backend, instance, shader->parameters[i].name, type, (float*)&idx);
      }
      else
        Shader::SetUniform(_backend, instance, shader->parameters[i].name, type, values[i].pf32);
      break;
    }
  }

  GLenum kind = 0;
  switch(primitive)
  {
  case FG_Primitive_TRIANGLE: kind = GL_TRIANGLES; break;
  case FG_Primitive_TRIANGLE_STRIP: kind = GL_TRIANGLE_STRIP; break;
  case FG_Primitive_LINE: kind = GL_LINES; break;
  case FG_Primitive_LINE_STRIP: kind = GL_LINE_STRIP; break;
  case FG_Primitive_POINT: kind = GL_POINT; break;
  }

  if(!vao->ElementType())
  {
    glDrawArrays(kind, 0, count);
    GL_ERROR("glDrawArrays");
  }
  else
  {
    glDrawElements(kind, count, vao->ElementType(), nullptr);
    GL_ERROR("glDrawElements");
  }

  glBindVertexArray(0);
  return -1;
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
  GL_ERROR("glGenBuffers");
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GL_ERROR("glBindBuffer");
  glBufferData(GL_ARRAY_BUFFER, stride * count, init, !init ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  GL_ERROR("glBufferData");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  GL_ERROR("glBindBuffer");
  return buffer;
}

GLuint Context::_genIndices(size_t num)
{
  GLuint indices;
  glGenBuffers(1, &indices);
  GL_ERROR("glGenBuffers");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
  GL_ERROR("glBindBuffer");
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
  GL_ERROR("glBufferData");
  return indices;
}

GLExpected<void> Context::SetDefaultState()
{
  glEnable(GL_BLEND);
  GL_ERROR("glEnable");
  glEnable(GL_TEXTURE_2D);
  GL_ERROR("glEnable");
  glFrontFace(GL_CW);
  GL_ERROR("glFrontFace");
  // glEnable(GL_CULL_FACE);
  // GL_ERROR("glEnable");
  glEnable(GL_FRAMEBUFFER_SRGB);
  GL_ERROR("glEnable");

  ApplyBlend(0, true);
  GL_ERROR("glBlendFunc");
}

void Context::CreateResources()
{
  SetDefaultState();

  _imageshader  = _backend->_imageshader.Create(_backend);
  _rectshader   = _backend->_rectshader.Create(_backend);
  _circleshader = _backend->_circleshader.Create(_backend);
  _arcshader    = _backend->_arcshader.Create(_backend);
  _trishader    = _backend->_trishader.Create(_backend);
  _lineshader   = _backend->_lineshader.Create(_backend);

  QuadVertex rect[4] = {
    { 0, 0 },
    { 1, 0 },
    { 0, 1 },
    { 1, 1 },
  };

  FG_ShaderParameter rectparams[1] = { { FG_ShaderType_FLOAT, 2, 0, "vPos" } };
  FG_ShaderParameter imgparams[2]  = { { FG_ShaderType_FLOAT, 4, 0, "vPosUV" }, { FG_ShaderType_FLOAT, 4, 0, "vColor" } };
  GLsizei quadstride               = sizeof(QuadVertex);
  GLsizei imagestride              = sizeof(ImageVertex);
  GLsizei linestride               = sizeof(FG_Vec);

  _quadbuffer = _createBuffer(sizeof(QuadVertex), 4, rect);
  _quadobject = std::make_unique<VAO>(_backend, _rectshader, rectparams, &_quadbuffer, &quadstride, 1, 0, 0);

  _imagebuffer  = _createBuffer(sizeof(ImageVertex), BATCH_BYTES / sizeof(ImageVertex), nullptr);
  _imageindices = _genIndices(BATCH_BYTES / sizeof(GLuint));
  _imageobject  = std::make_unique<VAO>(_backend, _imageshader, imgparams, &_imagebuffer, &imagestride, 1, _imageindices,
                                       GL_UNSIGNED_INT);

  _linebuffer = _createBuffer(sizeof(FG_Vec), BATCH_BYTES / sizeof(FG_Vec), nullptr);
  _lineobject = std::make_unique<VAO>(_backend, _lineshader, rectparams, &_linebuffer, &linestride, 1, 0, 0);

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
  if(asset->format >= FG_Format_VERTEX_BUFFER && asset->format <= FG_Format_UNKNOWN_BUFFER)
  {
    glGenBuffers(1, &idx);
    GL_ERROR("glGenBuffers");

    GLenum kind = 0;
    switch(asset->format)
    {
    case FG_Format_VERTEX_BUFFER: kind = GL_ARRAY_BUFFER; break;
    case FG_Format_UNIFORM_BUFFER: kind = GL_UNIFORM_BUFFER; break;
    case FG_Format_ELEMENT_BUFFER: kind = GL_ELEMENT_ARRAY_BUFFER; break;
    }

    glBindBuffer(kind, idx);
    GL_ERROR("glBindBuffer");
    glBufferData(kind, asset->bytes, asset->data.data, GL_STATIC_DRAW);
    GL_ERROR("glBufferData");
    glBindBuffer(kind, 0);
  }
  else
  {
    unsigned int flags = SOIL_FLAG_MULTIPLY_ALPHA;
    if(!(asset->flags & FG_AssetFlags_NO_MIPMAP))
      flags |= SOIL_FLAG_MIPMAPS;

    idx = _createTexture((const unsigned char*)asset->data.data, asset->size.x, asset->size.y, asset->channels,
                         SOIL_CREATE_NEW_ID, flags, GL_TEXTURE_2D, GL_TEXTURE_2D, GL_MAX_TEXTURE_SIZE);

    if(!idx)
    {
      GL_ERROR("SOIL_create_OGL_texture");
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "%s failed (returned 0).", "SOIL_create_OGL_texture");
      return 0;
    }

    glBindTexture(GL_TEXTURE_2D, idx);
    GL_ERROR("glBindTexture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    GL_ERROR("glTexParameteri");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GL_ERROR("glTexParameteri");
    glBindTexture(GL_TEXTURE_2D, 0);
    GL_ERROR("glBindTexture");
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

void Context::DestroyResources()
{
  for(khiter_t i = 0; i < kh_end(_texhash); ++i)
  {
    if(kh_exist(_texhash, i))
    {
      GLuint idx = kh_val(_texhash, i);
      if(kh_key(_texhash, i)->format >= FG_Format_VERTEX_BUFFER && kh_key(_texhash, i)->format >= FG_Format_UNKNOWN_BUFFER)
      {
        glDeleteBuffers(1, &idx);
        GL_ERROR("glDeleteBuffers");
      }
      else
      {
        glDeleteTextures(1, &idx);
        GL_ERROR("glDeleteTextures");
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
      GL_ERROR("glDeleteTextures");
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
      delete kh_val(_vaohash, i);
    }
  }
  kh_clear_vao(_vaohash);

  _backend->_imageshader.Destroy(_backend, _imageshader);
  _backend->_imageshader.Destroy(_backend, _rectshader);
  _backend->_imageshader.Destroy(_backend, _circleshader);
  _backend->_imageshader.Destroy(_backend, _arcshader);
  _backend->_imageshader.Destroy(_backend, _trishader);
  _backend->_imageshader.Destroy(_backend, _lineshader);

  _quadobject.reset();
  GL_ERROR("glDeleteVertexArrays");
  glDeleteBuffers(1, &_quadbuffer);
  GL_ERROR("glDeleteBuffers");
  _imageobject.reset();
  glDeleteBuffers(1, &_imagebuffer);
  GL_ERROR("glDeleteBuffers");
  glDeleteBuffers(1, &_imageindices);
  GL_ERROR("glDeleteBuffers");
  _lineobject.reset();
  glDeleteBuffers(1, &_linebuffer);
  GL_ERROR("glDeleteBuffers");

  for(auto& l : _layers)
    l->Destroy();

  _initialized = false;
}

void Context::FlipFlag(int diff, int flags, int flag, int option)
{
  if(diff & flag)
  {
    if(flags & flag)
      glEnable(option);
    else
      glDisable(option);
  }
}

GLExpected<void> Context::ApplyBlend(const FG_Blend& blend, const std::array<float, 4>& factor, bool force)
{
  if(_lastfactor != factor)
  {
    glBlendColor(factor[0], factor[1], factor[2], factor[3]);
    GL_ERROR("glBlendColor");
    _lastfactor = factor;
  }

  if(force || memcmp(&blend, &_lastblend, sizeof(FG_Blend)) != 0)
  {
    glBlendFuncSeparate(BlendMapping[blend.SrcBlend], BlendMapping[blend.DestBlend], BlendMapping[blend.SrcBlendAlpha],
                        BlendMapping[blend.DestBlendAlpha]);
    GL_ERROR("glBlendFuncSeperate");
    glBlendEquationSeparate(BlendOpMapping[blend.BlendOp], BlendOpMapping[blend.BlendOpAlpha]);
    GL_ERROR("glBlendEquationSeparate");

    if(_lastblend.RenderTargetWriteMask != blend.RenderTargetWriteMask)
    {
      glColorMask(blend.RenderTargetWriteMask & 0b0001, blend.RenderTargetWriteMask & 0b0010,
                  blend.RenderTargetWriteMask & 0b0100, blend.RenderTargetWriteMask & 0b1000);
      GL_ERROR("glColorMask");
    }

    _lastblend = blend;
  }

  return {};
}

GLExpected<void> Context::ApplyFlags(uint16_t flags, uint8_t cull, uint8_t fill)
{
  auto diff = _lastflags ^ flags;
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_RENDERTARGET_SRGB_ENABLE, GL_FRAMEBUFFER_SRGB);
  GL_ERROR("glEnable/glDisable");
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_DEPTH_ENABLE, GL_DEPTH_TEST);
  GL_ERROR("glEnable/glDisable");
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_STENCIL_ENABLE, GL_STENCIL_TEST);
  GL_ERROR("glEnable/glDisable");
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_ALPHA_TO_COVERAGE_ENABLE, GL_SAMPLE_ALPHA_TO_COVERAGE);
  GL_ERROR("glEnable/glDisable");
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_ANTIALIASED_LINE_ENABLE, GL_LINE_SMOOTH);
  GL_ERROR("glEnable/glDisable");
  FlipFlag(diff, flags, FG_PIPELINE_FLAG_MULTISAMPLE_ENABLE, GL_MULTISAMPLE);
  GL_ERROR("glEnable/glDisable");

  if(diff & FG_PIPELINE_FLAG_DEPTH_WRITE_ENABLE)
  {
    glDepthMask((flags & FG_PIPELINE_FLAG_DEPTH_WRITE_ENABLE) ? GL_TRUE : GL_FALSE);
    GL_ERROR("glDepthMask");
  }
  if(diff & FG_PIPELINE_FLAG_FRONT_COUNTER_CLOCKWISE)
  {
    glFrontFace((flags & FG_PIPELINE_FLAG_FRONT_COUNTER_CLOCKWISE) ? GL_CCW : GL_CW);
    GL_ERROR("glFrontFace");
  }

  if(_lastcull != cull)
  {
    if(cull == FG_CULL_MODE_NONE)
    {
      glDisable(GL_CULL_FACE);
      GL_ERROR("glDisable");
    }
    else
    {
      glEnable(GL_CULL_FACE);
      GL_ERROR("glEnable");

      glCullFace(cull == FG_CULL_MODE_BACK ? GL_BACK : GL_FRONT);
      GL_ERROR("glCullFace");
    }

    _lastcull = cull;
  }

  if(_lastfill != fill)
  {
    switch(fill)
    {
    case FG_FILL_MODE_FILL: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
    case FG_FILL_MODE_LINE: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
    case FG_FILL_MODE_POINT: glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
    }
    GL_ERROR("glPolygonMode");
    _lastfill = fill;
  }

  _lastflags = flags;
}

void Context::_flushbatchdraw(Font* font)
{
  glActiveTexture(GL_TEXTURE0);
  GL_ERROR("glActiveTexture");
  glBindTexture(GL_TEXTURE_2D, GetFontTexture(font));
  GL_ERROR("glBindTexture");

  // We've already set up our batch indices so we can just use them
  glDrawElements(GL_TRIANGLES, FlushBatch() * 6, GL_UNSIGNED_INT, nullptr);
  GL_ERROR("glDrawElements");
  glBindVertexArray(0);
  GL_ERROR("glBindVertexArray");
}

/*
int mipmapImageGamma(const unsigned char* const orig, int width, int height, int channels, unsigned char* resampled,
                     int block_size_x, int block_size_y)
{
  if((width < 1) || (height < 1) || (channels < 1) || (orig == NULL) || (resampled == NULL) || (block_size_x < 1) ||
     (block_size_y < 1))
  {
    return 0;
  }
  int mip_width  = std::max(width / block_size_x, 1);
  int mip_height = std::max(height / block_size_y, 1);

  for(int j = 0; j < mip_height; ++j)
  {
    for(int i = 0; i < mip_width; ++i)
    {
      for(int c = 0; c < channels; ++c)
      {
        const int index = (j * block_size_y) * width * channels + (i * block_size_x) * channels + c;
        int u_block     = block_size_x;
        int v_block     = block_size_y;
        // do a bit of checking so we don't over-run the boundaries (necessary for non-square textures!)
        if(block_size_x * (i + 1) > width)
        {
          u_block = width - i * block_size_y;
        }
        if(block_size_y * (j + 1) > height)
        {
          v_block = height - j * block_size_y;
        }
        int block_area  = u_block * v_block;
        float sum_value = 0.0f;
        for(int v = 0; v < v_block; ++v)
          for(int u = 0; u < u_block; ++u)
          {
            sum_value += Context::ToLinearRGB(orig[index + v * width * channels + u * channels] / 255.0f); // Linearize
          }
        resampled[j * mip_width * channels + i * channels + c] =
          (unsigned char)roundf(Context::ToSRGB(sum_value / block_area) * 255.0f); // De-linearize
      }
    }
  }
  return 1;
}

unsigned int Context::_createTexture(const unsigned char* const data, int width, int height, int channels,
                                     unsigned int reuse_texture_ID, unsigned int flags, unsigned int opengl_texture_type,
                                     unsigned int opengl_texture_target, unsigned int texture_check_size_enum)
{
  //	variables
  unsigned char* img;
  unsigned int tex_id;
  unsigned int internal_texture_format = 0, original_texture_format = 0;
  int max_supported_size;

  //	create a copy the image data
  img = (unsigned char*)malloc(width * height * channels);
  memcpy(img, data, width * height * channels);

  //	does the user want me to convert from straight to pre-multiplied alpha?
  if(flags & SOIL_FLAG_MULTIPLY_ALPHA)
  {
    // Only calculate mipmap without linearizing if the user says so
    if(flags & SOIL_FLAG_LINEAR_RGB)
    {
      switch(channels)
      {
      case 2:
        for(int i = 0; i < 2 * width * height; i += 2)
        {
          img[i] = (img[i] * img[i + 1] + 128) >> 8;
        }
        break;
      case 4:
        for(int i = 0; i < 4 * width * height; i += 4)
        {
          img[i + 0] = (img[i + 0] * img[i + 3] + 128) >> 8;
          img[i + 1] = (img[i + 1] * img[i + 3] + 128) >> 8;
          img[i + 2] = (img[i + 2] * img[i + 3] + 128) >> 8;
        }
        break;
      }
    }
    else
    {
      switch(channels)
      {
      case 2:
        for(int i = 0; i < 2 * width * height; i += 2)
        {
          // TODO: It is not clear if 2 channel images can be meaningfully linearized.
          img[i] = (img[i] * img[i + 1] + 128) >> 8;
        }
        break;
      case 4:
        for(int i = 0; i < 4 * width * height; i += 4)
        {
          // TODO: This can be SSE optimized, but needs a non-SSE implementation for other architectures
          float alpha = ToSRGB(img[i + 3] / 255.0f);
          img[i + 0]  = (unsigned char)roundf(((img[i + 0] / 255.0f) * alpha) * 255.0f);
          img[i + 1]  = (unsigned char)roundf(((img[i + 1] / 255.0f) * alpha) * 255.0f);
          img[i + 2]  = (unsigned char)roundf(((img[i + 2] / 255.0f) * alpha) * 255.0f);
        }
        break;
      }
    }
  }
  //	how large of a texture can this OpenGL implementation handle?
  //	texture_check_size_enum will be GL_MAX_TEXTURE_SIZE or SOIL_MAX_CUBE_MAP_TEXTURE_SIZE
  glGetIntegerv(texture_check_size_enum, &max_supported_size);
  //	do I need to make it a power of 2?
  if((flags & SOIL_FLAG_MIPMAPS) ||  //	mipmaps
     (width > max_supported_size) || //	it's too big, (make sure it's
     (height > max_supported_size))  //	2^n for later down-sampling)
  {
    int new_width  = 1;
    int new_height = 1;
    while(new_width < width)
    {
      new_width *= 2;
    }
    while(new_height < height)
    {
      new_height *= 2;
    }
    //	still?
    if((new_width != width) || (new_height != height))
    {
      //	yep, resize
      unsigned char* resampled = (unsigned char*)malloc(channels * new_width * new_height);
      up_scale_image(img, width, height, channels, resampled, new_width, new_height);
      //	nuke the old guy, then point it at the new guy
      SOIL_free_image_data(img);
      img    = resampled;
      width  = new_width;
      height = new_height;
    }
  }
  //	now, if it is too large...
  if((width > max_supported_size) || (height > max_supported_size))
  {
    //	I've already made it a power of two, so simply use the MIPmapping
    //  code to reduce its size to the allowable maximum.
    unsigned char* resampled;
    int reduce_block_x = 1, reduce_block_y = 1;
    int new_width, new_height;
    if(width > max_supported_size)
    {
      reduce_block_x = width / max_supported_size;
    }
    if(height > max_supported_size)
    {
      reduce_block_y = height / max_supported_size;
    }
    new_width  = width / reduce_block_x;
    new_height = height / reduce_block_y;
    resampled  = (unsigned char*)malloc(channels * new_width * new_height);
    //	perform the actual reduction
    if(SOIL_FLAG_LINEAR_RGB)
      mipmap_image(img, width, height, channels, resampled, reduce_block_x, reduce_block_y);
    else
      mipmapImageGamma(img, width, height, channels, resampled, reduce_block_x, reduce_block_y);

    //	nuke the old guy, then point it at the new guy
    SOIL_free_image_data(img);
    img    = resampled;
    width  = new_width;
    height = new_height;
  }
  //	create the OpenGL texture ID handle  (note: allowing a forced texture ID lets me reload a texture)
  tex_id = reuse_texture_ID;
  if(tex_id == 0)
  {
    glGenTextures(1, &tex_id);
  }
  GL_ERROR("glGenTextures");

  // Note: sometimes glGenTextures fails (usually no OpenGL context)
  if(tex_id)
  {
    //	and what type am I using as the internal texture format?
    switch(channels)
    {
    case 1: original_texture_format = GL_LUMINANCE; break;
    case 2: original_texture_format = GL_LUMINANCE_ALPHA; break;
    case 3: original_texture_format = GL_RGB; break;
    case 4: original_texture_format = GL_RGBA; break;
    }
    internal_texture_format = original_texture_format;
    if(!(flags & SOIL_FLAG_LINEAR_RGB))
    {
      switch(channels)
      {
      case 3: internal_texture_format = GL_SRGB8;
      case 4: internal_texture_format = GL_SRGB8_ALPHA8;
      }
    }

    //  bind an OpenGL texture ID
    glBindTexture(opengl_texture_type, tex_id);
    GL_ERROR("glBindTexture");

    //	user want OpenGL to do all the work!
    glTexImage2D(opengl_texture_target, 0, internal_texture_format, width, height, 0, original_texture_format,
                 GL_UNSIGNED_BYTE, img);
    GL_ERROR("glTexImage2D");
    //printf( "OpenGL DXT compressor\n" );

    //	are any MIPmaps desired?
    if(flags & SOIL_FLAG_MIPMAPS)
    {
      int MIPlevel             = 1;
      int MIPwidth             = (width + 1) / 2;
      int MIPheight            = (height + 1) / 2;
      unsigned char* resampled = (unsigned char*)malloc(channels * MIPwidth * MIPheight);
      while(((1 << MIPlevel) <= width) || ((1 << MIPlevel) <= height))
      {
        //	do this MIPmap level
        if(SOIL_FLAG_LINEAR_RGB)
          mipmap_image(img, width, height, channels, resampled, (1 << MIPlevel), (1 << MIPlevel));
        else
          mipmapImageGamma(img, width, height, channels, resampled, (1 << MIPlevel), (1 << MIPlevel));

        //  upload the MIPmaps
        glTexImage2D(opengl_texture_target, MIPlevel, internal_texture_format, MIPwidth, MIPheight, 0,
                     original_texture_format, GL_UNSIGNED_BYTE, resampled);
        GL_ERROR("glTexImage2D");

        //	prep for the next level
        ++MIPlevel;
        MIPwidth  = (MIPwidth + 1) / 2;
        MIPheight = (MIPheight + 1) / 2;
      }
      SOIL_free_image_data(resampled);
      //	instruct OpenGL to use the MIPmaps
      glTexParameteri(opengl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      GL_ERROR("glTexParameteri");
      glTexParameteri(opengl_texture_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
      GL_ERROR("glTexParameteri");
    }
    else
    {
      //	instruct OpenGL _NOT_ to use the MIPmaps
      glTexParameteri(opengl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      GL_ERROR("glTexParameteri");
      glTexParameteri(opengl_texture_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      GL_ERROR("glTexParameteri");
    }
    //	does the user want clamping, or wrapping?
    if(flags & SOIL_FLAG_TEXTURE_REPEATS)
    {
      glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_S, GL_REPEAT);
      GL_ERROR("glTexParameteri");
      glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_T, GL_REPEAT);
      GL_ERROR("glTexParameteri");
      if(opengl_texture_type == GL_TEXTURE_CUBE_MAP)
      {
        //	SOIL_TEXTURE_WRAP_R is invalid if cubemaps aren't supported
        glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_R, GL_REPEAT);
        GL_ERROR("glTexParameteri");
      }
    }
    else
    {
      unsigned int clamp_mode = GL_CLAMP_TO_EDGE;
      glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_S, clamp_mode);
      GL_ERROR("glTexParameteri");
      glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_T, clamp_mode);
      GL_ERROR("glTexParameteri");
      if(opengl_texture_type == GL_TEXTURE_CUBE_MAP)
      {
        //	SOIL_TEXTURE_WRAP_R is invalid if cubemaps aren't supported
        glTexParameteri(opengl_texture_type, GL_TEXTURE_WRAP_R, clamp_mode);
        GL_ERROR("glTexParameteri");
      }
    }
  }
  SOIL_free_image_data(img);
  return tex_id;
}
*/
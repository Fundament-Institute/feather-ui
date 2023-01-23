// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProviderGL.hpp"
#include "VertexArrayObject.hpp"
#include "ProgramObject.hpp"
#include "EnumMapping.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"
#include "Renderbuffer.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstring>

#define kh_pair_hash_func(key) \
  kh_int64_hash_func((static_cast<uint64_t>(kh_ptr_hash_func(key.first)) << 32) | kh_int_hash_func(key.second))

using namespace GL;

// Feather uses a premultiplied compositing pipeline:
// https://apoorvaj.io/alpha-compositing-opengl-blending-and-premultiplied-alpha/
const FG_Blend Context::Premultiply_Blend = {
  FG_Blend_Operand_One,
  FG_Blend_Operand_Inv_Src_Alpha,
  FG_Blend_Op_Add,
  FG_Blend_Operand_One,
  FG_Blend_Operand_Inv_Src_Alpha,
  FG_Blend_Op_Add,
  0b1111,
};

const FG_Blend Context::Normal_Blend = {
  FG_Blend_Operand_Src_Alpha,
  FG_Blend_Operand_Inv_Src_Alpha,
  FG_Blend_Op_Add,
  FG_Blend_Operand_One,
  FG_Blend_Operand_Inv_Src_Alpha,
  FG_Blend_Op_Add,
  0b1111,
};

const FG_Blend Context::Default_Blend = {
  FG_Blend_Operand_One,  FG_Blend_Operand_Zero, FG_Blend_Op_Add, FG_Blend_Operand_One,
  FG_Blend_Operand_Zero, FG_Blend_Op_Add,       0b1111,
};

Context::Context(FG_Vec2 dim) :
  _lastblend(Default_Blend),
  _program(nullptr),
  _dim(dim),
  _indextype(0),
  _lastcull(0),
  _lastfill(0),
  _lastfactor({ 0, 0, 0, 0 }),
  _lastflags(0),
  _primitive(0),
  _statestore({ 0 }),
  _workgroup({ 0, 0, 0 })
{}
Context::~Context() {}

GLExpected<void> Context::BeginDraw(const FG_Rect* area)
{
  GLint box[4] = { 0 };
  RETURN_ERROR(CALLGL(glGetIntegerv, GL_SCISSOR_BOX, box));
  _lastscissor = {
    static_cast<float>(box[0]),
    static_cast<float>(box[1]),
    static_cast<float>(box[2] + box[0]),
    static_cast<float>(box[3] + box[1]),
  };
  return SetScissors({ &_lastscissor, 1 });
}

GLExpected<void> Context::EndDraw() { return {}; }
GLExpected<void> Context::Resize(FG_Vec2 dim)
{
  _dim         = dim;
  _lastscissor = { 0, 0, _dim.x, _dim.y };
  return SetScissors({ &_lastscissor, 1 });
}

GLExpected<void> Context::Dispatch() { return CALLGL(glDispatchCompute, _workgroup.x, _workgroup.y, _workgroup.z); }

GLExpected<void> Context::Barrier(GLbitfield barrier_flags) { return CALLGL(glMemoryBarrier, barrier_flags); }

GLExpected<void> Context::ApplyProgram(const ProgramObject& program)
{
  RETURN_ERROR(CALLGL(glUseProgram, program));
  _program = &program;
  return {};
}

GLExpected<void> Context::Clear(uint8_t clearbits, FG_Color16 RGBA, uint8_t stencil, float depth, std::span<FG_Rect> rects)
{
  RETURN_ERROR(CALLGL(glClearDepth, depth));
  RETURN_ERROR(CALLGL(glClearStencil, stencil));

  std::array<float, 4> colors;
  Context::ColorFloats(RGBA, colors, false);
  RETURN_ERROR(CALLGL(glClearColor, colors[0], colors[1], colors[2], colors[3]));

  GLbitfield flags = 0;
  if(clearbits & FG_ClearFlag_Color)
    flags |= GL_COLOR_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Depth)
    flags |= GL_DEPTH_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Stencil)
    flags |= GL_STENCIL_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Accumulator)
    flags |= GL_ACCUM_BUFFER_BIT;

  if(rects.empty())
  {
    RETURN_ERROR(CALLGL(glClear, flags));
  }

  for(auto& r : rects)
  {
    RETURN_ERROR(CallWithRect(r, glScissor, "glScissor"));
    RETURN_ERROR(CALLGL(glClear, flags));
  }

  RETURN_ERROR(CallWithRect(_lastscissor, glScissor, "glScissor"));
  return {};
}

GLExpected<void> Context::SetViewports(std::span<FG_Viewport> viewports)
{
  if(viewports.size() > 0)
  {
    FG_Rect r = { viewports[0].pos.x, viewports[0].pos.y, viewports[0].pos.x + viewports[0].dim.x,
                  viewports[0].pos.y + viewports[0].dim.y };
    RETURN_ERROR(CallWithRect(r, glViewport, "glViewport"));
  }
  return {};
}
GLExpected<void> Context::SetScissors(std::span<FG_Rect> rects)
{
  if(rects.size() > 0)
  {
    _lastscissor = rects[0];
    RETURN_ERROR(CallWithRect(rects[0], glScissor, "glScissor"));
  }
  return {};
}
GLExpected<void> Context::SetShaderUniforms(const FG_ShaderParameter* uniforms, const FG_ShaderValue* values,
                                            uint32_t count)
{
  for(uint32_t i = 0; i < count; ++i)
  {
    if(uniforms[i].type == FG_Shader_Type_Buffer)
    {
      RETURN_ERROR(_program->set_buffer(values[i].resource, uniforms[i].count, uniforms[i].width, uniforms[i].length));
      continue;
    }

    auto type      = ShaderObject::get_type(uniforms[i]);
    uint32_t count = !uniforms[i].count ? 1 : uniforms[i].count;

    switch(type)
    {
    case GL_DOUBLE:
    case GL_HALF_FLOAT: // we assume you pass in a proper float to fill this
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
      if(count == 1)
      {
        RETURN_ERROR(_program->set_uniform(uniforms[i].name, type, &values[i].f32, count));
        break;
      }
      // Otherwise fallthrough because the value couldn't be stored directly
    default:
      if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
      {
        RETURN_ERROR(_program->set_texture(uniforms[i].name, type, static_cast<GLuint>(values[i].resource)));
      }
      else
      {
        RETURN_ERROR(_program->set_uniform(uniforms[i].name, type, values[i].pf32, count));
      }
      break;
    }
  }
  return {};
}

GLExpected<void> Context::DrawArrays(uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex,
                                     uint32_t startinstance)
{
  if(instancecount > 0)
  {
    RETURN_ERROR(CALLGL(glDrawArraysInstanced, _primitive, startinstance, vertexcount, instancecount));
  }
  else
  {
    RETURN_ERROR(CALLGL(glDrawArrays, _primitive, startvertex, vertexcount));
  }

  return {};
}
GLExpected<void> Context::DrawIndexed(GLsizei indexcount, GLsizei instancecount, uint32_t startindex, int startvertex,
                                      uint32_t startinstance)
{
  if(instancecount > 0)
  {
    // GLenum mode, GLsizei count, GLenum type, const void *indices,GLsizei instancecount

    RETURN_ERROR(CALLGL(glDrawElementsInstanced, _primitive, indexcount, _indextype, nullptr, instancecount));
  }
  else
  {
    RETURN_ERROR(CALLGL(glDrawElements, _primitive, indexcount, _indextype, nullptr));
  }

  return {};
}
GLExpected<void> Context::DrawMesh(uint32_t start, uint32_t count) { return CALLGL(glDrawMeshTasksNV, start, count); }

GL::GLExpected<void> Context::CopySubresource(FG_Resource src, FG_Resource dest, unsigned long srcoffset,
                                              unsigned long destoffset, unsigned long bytes)
{
  if(Buffer::validate(src) && Buffer::validate(dest))
  {
    if(auto e = Buffer(src).bind(GL_COPY_READ_BUFFER))
    {
      if(auto b = Buffer(dest).bind(GL_COPY_WRITE_BUFFER))
      {
        RETURN_ERROR(CALLGL(glCopyBufferSubData, GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcoffset, destoffset, bytes));
      }
      else
        return std::move(b.error());
    }
    else
      return std::move(e.error());
  }
  else
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Mismatched src / dest resources");

  return {};
}

GL::GLExpected<void> Context::CopyResourceRegion(FG_Resource src, FG_Resource dest, int level, FG_Vec3i srcoffset,
                                                 FG_Vec3i destoffset, FG_Vec3i size)
{
  GLuint source      = src;
  GLuint destination = dest;

  if(Texture::validate(src) && Texture::validate(dest))
  {
    if(size.y == 0 && size.z == 0)
    {
      glCopyImageSubData(source, GL_TEXTURE_1D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_1D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, 1, 1);
    }
    else if(size.z == 0)
    {
      glCopyImageSubData(source, GL_TEXTURE_2D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_2D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, size.y, 1);
    }
    else
    {
      glCopyImageSubData(source, GL_TEXTURE_3D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_3D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z);
    }
    GL_ERROR("glCopyImageSubData");
  }
  else if(Renderbuffer::validate(src) && Renderbuffer::validate(dest))
  {
    RETURN_ERROR(CALLGL(glCopyImageSubData, source, GL_RENDERBUFFER, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination,
                        GL_RENDERBUFFER, 0, destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z));
  }
  else
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Mismatched src / dest resources");

  return {};
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
  // case GL_INT_2_10_10_10_REV:
  case GL_UNSIGNED_INT_2_10_10_10_REV:
  case GL_UNSIGNED_INT_10F_11F_11F_REV:
  case GL_FLOAT: return 4;
  case GL_DOUBLE: return 8;
  }
  assert(false);
  return 0;
}

GLExpected<void> Context::FlipFlag(int diff, int flags, int flag, int option)
{
  if(diff & flag)
  {
    if(flags & flag)
    {
      RETURN_ERROR(CALLGL(glEnable, option));
    }
    else
    {
      RETURN_ERROR(CALLGL(glDisable, option));
    }
  }

  return {};
}

GLExpected<void> Context::ApplyBlendFactor(const std::array<float, 4>& factor)
{
  if(_lastfactor != factor)
  {
    RETURN_ERROR(CALLGL(glBlendColor, factor[0], factor[1], factor[2], factor[3]));
    _lastfactor = factor;
  }
  return {};
}

GLExpected<void> Context::ApplyBlend(const FG_Blend& blend, bool force)
{
  if(force || memcmp(&blend, &_lastblend, sizeof(FG_Blend)) != 0)
  {
    RETURN_ERROR(CALLGL(glBlendFuncSeparate, BlendMapping[blend.src_blend], BlendMapping[blend.dest_blend],
                        BlendMapping[blend.src_blend_alpha], BlendMapping[blend.dest_blend_alpha]));
    RETURN_ERROR(CALLGL(glBlendEquationSeparate, BlendOpMapping[blend.blend_op], BlendOpMapping[blend.blend_op_alpha]));

    if(_lastblend.rendertarget_write_mask != blend.rendertarget_write_mask)
    {
      RETURN_ERROR(CALLGL(glColorMask, blend.rendertarget_write_mask & 0b0001, blend.rendertarget_write_mask & 0b0010,
                          blend.rendertarget_write_mask & 0b0100, blend.rendertarget_write_mask & 0b1000));
    }

    _lastblend = blend;
  }

  return {};
}

GLExpected<void> Context::ApplyFill(uint8_t fill)
{
  if(_lastfill != fill)
  {
    switch(fill)
    {
    case FG_Fill_Mode_Fill: RETURN_ERROR(CALLGL(glPolygonMode, GL_FRONT_AND_BACK, GL_FILL)); break;
    case FG_Fill_Mode_Line: RETURN_ERROR(CALLGL(glPolygonMode, GL_FRONT_AND_BACK, GL_LINE)); break;
    case FG_Fill_Mode_Point: RETURN_ERROR(CALLGL(glPolygonMode, GL_FRONT_AND_BACK, GL_POINT)); break;
    }
    _lastfill = fill;
  }
  return {};
}

GLExpected<void> Context::ApplyCull(uint8_t cull)
{
  if(_lastcull != cull)
  {
    if(cull == FG_Cull_Mode_None)
    {
      RETURN_ERROR(CALLGL(glDisable, GL_CULL_FACE));
    }
    else
    {
      RETURN_ERROR(CALLGL(glEnable, GL_CULL_FACE));

      RETURN_ERROR(CALLGL(glCullFace, cull == FG_Cull_Mode_Back ? GL_BACK : GL_FRONT));
    }

    _lastcull = cull;
  }
  return {};
}
GLExpected<void> Context::ApplyFlags(uint16_t flags)
{
  auto diff = _lastflags ^ flags;
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_RenderTarget_SRGB_Enable, GL_FRAMEBUFFER_SRGB));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Depth_Enable, GL_DEPTH_TEST));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Stencil_Enable, GL_STENCIL_TEST));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Alpha_To_Coverage_Enable, GL_SAMPLE_ALPHA_TO_COVERAGE));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Antialiased_Line_Enable, GL_LINE_SMOOTH));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Multisample_Enable, GL_MULTISAMPLE));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Blend_Enable, GL_BLEND));
  RETURN_ERROR(FlipFlag(diff, flags, FG_Pipeline_Flag_Scissor_Enable, GL_SCISSOR_TEST));

  if(diff & FG_Pipeline_Flag_Depth_Write_Enable)
  {
    RETURN_ERROR(CALLGL(glDepthMask, (flags & FG_Pipeline_Flag_Depth_Write_Enable) ? GL_TRUE : GL_FALSE));
  }
  if(diff & FG_Pipeline_Flag_Front_Counter_Clockwise)
  {
    RETURN_ERROR(CALLGL(glFrontFace, (flags & FG_Pipeline_Flag_Front_Counter_Clockwise) ? GL_CCW : GL_CW));
  }

  _lastflags = flags;
  return {};
}

/*
void Context::_flushbatchdraw(Font* font)
{
  RETURN_ERROR(CALLGL(glActiveTexture, GL_TEXTURE0));
  GL_ERROR("glActiveTexture");
  glBindTexture(GL_TEXTURE_2D, GetFontTexture(font));
  GL_ERROR("glBindTexture");

  // We've already set up our batch indices so we can just use them
  glDrawElements(GL_TRIANGLES, FlushBatch() * 6, GL_UNSIGNED_INT, nullptr);
  GL_ERROR("glDrawElements");
  RETURN_ERROR(CALLGL(glBindVertexArray, 0));
  GL_ERROR("glBindVertexArray");
}

int mipmapImageGamma(const unsigned char* const orig, int width, int height, int channels, unsigned char* resampled,
                     int block_size_x, int block_size_y)
{
  if((width < 1) || (height < 1) || (channels < 1) || (orig == nullptr) || (resampled == nullptr) || (block_size_x < 1) ||
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
  RETURN_ERROR(CALLGL(glGetIntegerv, texture_check_size_enum, &max_supported_size));
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
    RETURN_ERROR(CALLGL(glGenTextures, 1, &tex_id));
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
    RETURN_ERROR(CALLGL(glBindTexture, opengl_texture_type, tex_id));
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
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
      GL_ERROR("glTexParameteri");
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
      GL_ERROR("glTexParameteri");
    }
    else
    {
      //	instruct OpenGL _NOT_ to use the MIPmaps
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
      GL_ERROR("glTexParameteri");
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
      GL_ERROR("glTexParameteri");
    }
    //	does the user want clamping, or wrapping?
    if(flags & SOIL_FLAG_TEXTURE_REPEATS)
    {
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_S, GL_REPEAT));
      GL_ERROR("glTexParameteri");
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_T, GL_REPEAT));
      GL_ERROR("glTexParameteri");
      if(opengl_texture_type == GL_TEXTURE_CUBE_MAP)
      {
        //	SOIL_TEXTURE_WRAP_R is invalid if cubemaps aren't supported
        RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_R, GL_REPEAT));
        GL_ERROR("glTexParameteri");
      }
    }
    else
    {
      unsigned int clamp_mode = GL_CLAMP_TO_EDGE;
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_S, clamp_mode));
      GL_ERROR("glTexParameteri");
      RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_T, clamp_mode));
      GL_ERROR("glTexParameteri");
      if(opengl_texture_type == GL_TEXTURE_CUBE_MAP)
      {
        //	SOIL_TEXTURE_WRAP_R is invalid if cubemaps aren't supported
        RETURN_ERROR(CALLGL(glTexParameteri, opengl_texture_type, GL_TEXTURE_WRAP_R, clamp_mode));
        GL_ERROR("glTexParameteri");
      }
    }
  }
  SOIL_free_image_data(img);
  return tex_id;
}
*/
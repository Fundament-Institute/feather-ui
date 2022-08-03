// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "BackendGL.hpp"
#include "FrameBuffer.hpp"
#include <cassert>
#include <cmath>

using namespace GL;

GLExpected<BindRef> FrameBuffer::bind(GLenum target) const noexcept
{
  glBindFramebuffer(target, _ref);
  GL_ERROR("glBindFramebuffer");
  return BindRef{ target, [](GLenum target) { glBindFramebuffer(target, 0); } };
}

GLExpected<Owned<FrameBuffer>> FrameBuffer::create(GLenum target, GLenum type, int level, int zoffset,
                                                   FG_Resource* textures, uint32_t n_textures) noexcept
{
  assert(glFramebufferTexture2D != nullptr);

  GLuint fbgl;
  glGenFramebuffers(1, &fbgl);
  GL_ERROR("glGenFramebuffers");
  Owned<FrameBuffer> fb(fbgl);

  RETURN_ERROR(fb->attach(target, type, level, zoffset, textures, n_textures));
  return fb;
}

GLExpected<void> FrameBuffer::attach(GLenum target, GLenum type, int level, int zoffset, FG_Resource* textures,
                                     uint32_t n_textures) noexcept
{
  if(auto e = bind(target))
  {
    int MaxRendertargets;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MaxRendertargets);
    GL_ERROR("glGetIntergerv");
    if((this->_numberOfColorAttachments + int(n_textures)) > MaxRendertargets)
    {
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Trying to bind more render targets than max possible");
    }
    for(auto i = 0; i < n_textures; i++)
    {
      Texture texture{ textures[i] };
      switch(type)
      {
      case GL_TEXTURE_1D:
        glFramebufferTexture1D(target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments, GL_TEXTURE_1D, texture,
                               level);
        break;
      case GL_TEXTURE_3D:
        glFramebufferTexture3D(target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments, GL_TEXTURE_3D, texture,
                               level, zoffset);
        break;
      default:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments, GL_TEXTURE_2D, texture,
                               level);
        break;
      }
      this->_numberOfColorAttachments++;
      GL_ERROR("glFramebufferTexture");
    }

    auto status = glCheckFramebufferStatus(target);
    if(status != GL_FRAMEBUFFER_COMPLETE)
    {
      return CUSTOM_ERROR(status, "glCheckFramebufferStatus");
    }
  }
  else
    return std::move(e.error());

  return {};
}
GLExpected<void> FrameBuffer::attach2D(GLenum target, GLenum attachment, GLenum type, FG_Resource texture, int level)
{
  if(auto e = bind(target))
  {
    glFramebufferTexture2D(target, attachment, type, Texture(texture), level);
    GL_ERROR("glFramebufferTexture2D");
  }
  else
    return std::move(e.error());

  return {};
}
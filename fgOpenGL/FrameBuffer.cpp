// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProviderGL.hpp"
#include "FrameBuffer.hpp"
#include <cassert>
#include <cmath>

using namespace GL;

GLExpected<BindRef> Framebuffer::bind(GLenum target) const noexcept
{
  RETURN_ERROR(CALLGL(glBindFramebuffer, target, _ref));
  return BindRef{ target, [](GLenum target) { glBindFramebuffer(target, 0); } };
}

GLExpected<Owned<Framebuffer>> Framebuffer::create(GLenum target, GLenum type, int level, int zoffset,
                                                   FG_Resource* textures, uint32_t n_textures) noexcept
{
  GLuint fbgl;
  RETURN_ERROR(CALLGL(glGenFramebuffers, 1, &fbgl));
  Owned<Framebuffer> fb(fbgl);

  RETURN_ERROR(fb->attach(target, type, level, zoffset, textures, n_textures));
  return fb;
}

GLExpected<void> Framebuffer::attach(GLenum target, GLenum type, int level, int zoffset, FG_Resource* textures,
                                     uint32_t n_textures) noexcept
{
  if(auto e = bind(target))
  {
    int MaxRendertargets;
    RETURN_ERROR(CALLGL(glGetIntegerv, GL_MAX_COLOR_ATTACHMENTS, &MaxRendertargets));
    if((this->_numberOfColorAttachments + int(n_textures)) > MaxRendertargets)
    {
      return CUSTOM_ERROR(ERR_TOO_MANY_RENDERTARGETS, "Trying to bind more render targets than max possible");
    }
    for(auto i = 0; i < n_textures; i++)
    {
      Texture texture{ textures[i] };
      switch(type)
      {
      case GL_TEXTURE_1D:
        RETURN_ERROR(CALLGL(glFramebufferTexture1D, target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments,
                            GL_TEXTURE_1D, texture, level));
        break;
      case GL_TEXTURE_3D:
        RETURN_ERROR(CALLGL(glFramebufferTexture3D, target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments,
                            GL_TEXTURE_3D, texture, level, zoffset));
        break;
      default:
        RETURN_ERROR(CALLGL(glFramebufferTexture2D, target, GL_COLOR_ATTACHMENT0 + this->_numberOfColorAttachments,
                            GL_TEXTURE_2D, texture, level));
        break;
      }
      this->_numberOfColorAttachments++;
    }

    auto status = CALLGL(glCheckFramebufferStatus, target);
    if(status.has_error())
      return std::move(e.error());
    if(status.value() != GL_FRAMEBUFFER_COMPLETE)
    {
      return CUSTOM_ERROR(status.value(), "glCheckFramebufferStatus");
    }
  }
  else
    return std::move(e.error());

  return {};
}
GLExpected<void> Framebuffer::attach2D(GLenum target, GLenum attachment, GLenum type, FG_Resource texture, int level)
{
  if(auto e = bind(target))
  {
    RETURN_ERROR(CALLGL(glFramebufferTexture2D, target, attachment, type, Texture(texture), level));
  }
  else
    return std::move(e.error());

  return {};
}
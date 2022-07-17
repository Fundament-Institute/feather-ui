// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "BackendGL.hpp"
#include "FrameBuffer.hpp"
#include <cassert>
#include <cmath>


using namespace GL;

GLExpected<FrameBuffer::GLFrameBufferBindRef> FrameBuffer::bind(GLenum target) const noexcept
{
  glBindFramebuffer(target, _ref);
  GL_ERROR("glBindFramebuffer");
  return GLFrameBufferBindRef(target);
}

GLExpected<FrameBuffer> FrameBuffer::create(GLenum target, GLenum type, int level, int zoffset, std::vector<const Texture*> textures) noexcept
{
  // TODO: Default to GL_DRAW_FRAMEBUFFER?
  assert(glFramebufferTexture2D != nullptr);

  GLuint fbgl;
  glGenFramebuffers(1, &fbgl);
  GL_ERROR("glGenFramebuffers");
  FrameBuffer fb(fbgl);

  fb.attach(target, type, level, zoffset, textures);
  

  auto status = glCheckFramebufferStatus(target);
  if(status != GL_FRAMEBUFFER_COMPLETE)
  {
    return CUSTOM_ERROR(status, "glCheckFramebufferStatus");
  }
  return fb;
}

GLExpected<FrameBuffer> FrameBuffer::attach(GLenum target, GLenum type, int level, int zoffset, std::vector<const Texture*> textures) noexcept
{
  if(auto e = this->bind(target))
  {
    for(const Texture* texture : textures)
    {
      switch(type)
      {
      case GL_TEXTURE_1D: glFramebufferTexture1D(target, this->NumberOfColorAttachments, GL_TEXTURE_1D, *texture, level); break;
      case GL_TEXTURE_3D: glFramebufferTexture3D(target, this->NumberOfColorAttachments, GL_TEXTURE_3D, *texture, level, zoffset); break;
      default: glFramebufferTexture2D(target, this->NumberOfColorAttachments, type, *texture, level); break;
      }
      this->NumberOfColorAttachments++;
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
}
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

GLExpected<FrameBuffer> FrameBuffer::create(GLenum target, const Texture& texture, GLenum attach, GLenum type,
                                            int level, int zoffset) noexcept
{
  // TODO: Default to GL_DRAW_FRAMEBUFFER?
  assert(glFramebufferTexture2D != nullptr);

  GLuint fbgl;
  glGenFramebuffers(1, &fbgl);
  GL_ERROR("glGenFramebuffers");
  FrameBuffer fb(fbgl);
  if(auto e = fb.bind(target))
  {
    switch(type)
    {
    case GL_TEXTURE_1D: glFramebufferTexture1D(target, attach, GL_TEXTURE_1D, texture, level); break;
    case GL_TEXTURE_3D: glFramebufferTexture3D(target, attach, GL_TEXTURE_3D, texture, level, zoffset); break;
    default: glFramebufferTexture2D(target, attach, type, texture, level); break;
    }
    GL_ERROR("glFramebufferTexture");
    auto status = glCheckFramebufferStatus(target);
    if(status != GL_FRAMEBUFFER_COMPLETE)
    {
      return CUSTOM_ERROR(status, "glCheckFramebufferStatus");
    }
  }
  else
    return std::move(e.error());

  return fb;
}
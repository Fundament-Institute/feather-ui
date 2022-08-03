// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "BackendGL.hpp"
#include "Renderbuffer.hpp"

using namespace GL;

GLExpected<BindRef> Renderbuffer::bind(GLenum target) const noexcept
{
  glBindRenderbuffer(target, _ref);
  GL_ERROR("glBindRenderbuffer");
  return BindRef{ target, [](GLenum target) { glBindRenderbuffer(target, 0); } };
}

GLExpected<Owned<Renderbuffer>> Renderbuffer::create(GLenum target, Format format, FG_Vec2i size, int samples)
{
  GLuint rbgl;
  glGenRenderbuffers(1, &rbgl);
  GL_ERROR("glGenRenderbuffers");
  Owned<Renderbuffer> rb(rbgl);
  if(auto bind = rb.bind(target))
  { 
    glRenderbufferStorageMultisample(target, samples, format.internalformat, size.x, size.y);
    GL_ERROR("glRenderbufferStorageMultisample");
  }
  else
    return std::move(bind.error());

  return rb;
}
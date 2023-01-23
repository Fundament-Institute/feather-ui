// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProviderGL.hpp"
#include "Renderbuffer.hpp"

using namespace GL;

GLExpected<BindRef> Renderbuffer::bind(GLenum target) const noexcept
{
  RETURN_ERROR(CALLGL(glBindRenderbuffer, target, _ref));
  return BindRef{ target, [](GLenum target) { glBindRenderbuffer(target, 0); } };
}

GLExpected<Owned<Renderbuffer>> Renderbuffer::create(GLenum target, Format format, FG_Vec2i size, int samples)
{
  GLuint rbgl;
  RETURN_ERROR(CALLGL(glGenRenderbuffers, 1, &rbgl));
  Owned<Renderbuffer> rb(rbgl);
  if(auto bind = rb.bind(target))
  {
    RETURN_ERROR(CALLGL(glRenderbufferStorageMultisample, target, samples, format.internalformat, size.x, size.y));
  }
  else
    return std::move(bind.error());

  return rb;
}
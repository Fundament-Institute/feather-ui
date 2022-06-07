// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__BUFFER_H
#define GL__BUFFER_H

#include "GLRef.h"

namespace GL {
  static constexpr bool IsBuffer(GLuint i) noexcept { return glIsBuffer(i) == GL_TRUE; };
  static constexpr void DeleteBuffer(GLuint i) noexcept { glDeleteBuffers(1, &i); };
  static constexpr void UnbindBuffer(GLenum target) noexcept { glBindBuffer(target, 0); };

  struct Buffer : GLRef<&IsBuffer, &DeleteBuffer>
  {
    typedef GLBindRef<&UnbindBuffer> GLBufferBindRef;

    explicit constexpr Buffer(GLuint shader) noexcept : GLRef(shader) {}
    explicit constexpr Buffer(void* shader) noexcept : GLRef(shader) {}
    constexpr Buffer() noexcept               = default;
    constexpr Buffer(Buffer&& right) noexcept = default;
    constexpr Buffer(const Buffer&)           = delete;
    constexpr ~Buffer() noexcept              = default;
    GLExpected<GLBufferBindRef> bind(GLenum target) const noexcept
    {
      glBindBuffer(target, _ref);
      GL_ERROR("glBindBuffer");
      return GLBufferBindRef(target);
    }

    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&& right) noexcept = default;

    static GLExpected<Buffer> create(GLenum target, void* data, GLsizeiptr bytes)
    {
      GLuint fbgl;
      glGenBuffers(1, &fbgl);
      GL_ERROR("glGenBuffers");
      Buffer fb(fbgl);
      if(auto e = fb.bind(target))
      {
        glBufferData(target, bytes, data, !data ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
        GL_ERROR("glBufferData");
      }
      else
        return std::move(e.error());

      return fb;
    }
  };
}

#endif

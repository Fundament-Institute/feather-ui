// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__BUFFER_H
#define GL__BUFFER_H

#include "Ref.hpp"

namespace GL {
  static bool IsBuffer(GLuint i) noexcept { return glIsBuffer(i) == GL_TRUE; };

  struct Buffer : Ref<&IsBuffer>
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteBuffers(1, &i); };

    explicit constexpr Buffer() noexcept : Ref() {}
    explicit constexpr Buffer(GLuint buffer) noexcept : Ref(buffer) {}
    explicit constexpr Buffer(FG_Resource buffer) noexcept : Ref(static_cast<GLuint>(buffer))
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(is_valid());
#endif
    }
    constexpr Buffer(const Buffer&) = default;
    constexpr ~Buffer() noexcept    = default;
    GLExpected<BindRef> bind(GLenum target) const noexcept
    {
      glBindBuffer(target, _ref);
      GL_ERROR("glBindBuffer");
      return BindRef{ target, [](GLenum target) { glBindBuffer(target, 0); } };
    }

    Buffer& operator=(const Buffer&) = default;

    static GLExpected<Owned<Buffer>> create(GLenum target, void* data, GLsizeiptr bytes)
    {
      GLuint fbgl;
      glGenBuffers(1, &fbgl);
      GL_ERROR("glGenBuffers");
      Owned<Buffer> fb(fbgl);
      if(auto e = fb.bind(target))
      {
        glBufferData(target, bytes, data, !data ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
        GL_ERROR("glBufferData");
      }
      else
        return std::move(e.error());

      return std::move(fb);
    }
  };

}

#endif

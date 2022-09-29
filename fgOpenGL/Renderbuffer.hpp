// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#ifndef GL__RENDER_BUFFER_H
#define GL__RENDER_BUFFER_H

#include "Texture.hpp"

namespace GL {
  struct Renderbuffer : Ref
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteRenderbuffers(1, &i); };

    explicit constexpr Renderbuffer(GLuint buffer) noexcept : Ref(buffer) {}
    explicit constexpr Renderbuffer(FG_Resource buffer) noexcept : Ref(static_cast<GLuint>(buffer & REF_MASK))
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(validate(buffer));
#endif
    }
    constexpr Renderbuffer() noexcept                     = default;
    constexpr Renderbuffer(const Renderbuffer&) = default;
    constexpr ~Renderbuffer() noexcept               = default;
    GLExpected<BindRef> bind(GLenum target) const noexcept;

    inline operator FG_Resource() const noexcept { return _ref | REF_RENDERBUFFER; }
    Renderbuffer& operator=(const Renderbuffer&) = default;

    static bool validate(FG_Resource res) noexcept
    {
      return ((res & REF_RENDERBUFFER) != 0) && (glIsRenderbuffer(static_cast<GLuint>(res & REF_MASK)) == GL_TRUE);
    }
    static GLExpected<Owned<Renderbuffer>> create(GLenum target, Format format, FG_Vec2i size, int samples = 0);
  };
}

#endif

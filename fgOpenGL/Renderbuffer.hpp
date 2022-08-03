// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__RENDER_BUFFER_H
#define GL__RENDER_BUFFER_H

#include "Texture.hpp"

namespace GL {
  static bool IsRenderbuffer(GLuint i) noexcept { return glIsRenderbuffer(i) == GL_TRUE; };

  struct Renderbuffer : Ref<&IsRenderbuffer>
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteRenderbuffers(1, &i); };

    explicit constexpr Renderbuffer(GLuint buffer) noexcept : Ref(buffer) {}
    explicit constexpr Renderbuffer(FG_Resource buffer) noexcept : Ref(static_cast<GLuint>(buffer)) {}
    constexpr Renderbuffer() noexcept                     = default;
    constexpr Renderbuffer(Renderbuffer&& right) noexcept = default;
    constexpr Renderbuffer(const Renderbuffer&)           = delete;
    constexpr ~Renderbuffer() noexcept               = default;
    GLExpected<BindRef> bind(GLenum target) const noexcept;

    Renderbuffer& operator=(const Renderbuffer&) = delete;
    Renderbuffer& operator=(Renderbuffer&& right) noexcept = default;

    static GLExpected<Owned<Renderbuffer>> create(GLenum target, Format format, FG_Vec2i size, int samples = 0);
  };
}

#endif

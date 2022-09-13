// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.hpp"

namespace GL {
  struct Framebuffer : Ref
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteFramebuffers(1, &i); };

    explicit constexpr Framebuffer(GLuint buffer) noexcept : Ref(buffer), _numberOfColorAttachments(0) {}
    explicit constexpr Framebuffer(FG_Resource buffer) noexcept :
      Ref(static_cast<GLuint>(buffer & REF_MASK)), _numberOfColorAttachments(0)
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(validate(buffer));
#endif
    }
    constexpr Framebuffer() noexcept : _numberOfColorAttachments(0) {}
    constexpr Framebuffer(const Framebuffer&)           = default;
    constexpr ~Framebuffer() noexcept                   = default;

    GLExpected<BindRef> bind(GLenum target) const noexcept;

    Framebuffer& operator=(const Framebuffer&) = default;
    inline operator FG_Resource() const noexcept { return _ref | REF_FRAMEBUFFER; }

    static bool validate(FG_Resource res) noexcept
    {
      return ((res & REF_FRAMEBUFFER) != 0) && (glIsFramebuffer(static_cast<GLuint>(res & REF_MASK)) == GL_TRUE);
    }
    // This does not take ownership of the texture
    static GLExpected<Owned<Framebuffer>> create(GLenum target, GLenum type, int level, int zoffset, FG_Resource* textures,
                                          uint32_t n_textures) noexcept;

    GLExpected<void> attach(GLenum target, GLenum type, int level, int zoffset, FG_Resource* textures,
                            uint32_t n_textures) noexcept;
    GLExpected<void> attach2D(GLenum target, GLenum attachment, GLenum type, FG_Resource texture, int level);

    GLuint ColorAttachments() { return this->_numberOfColorAttachments; };
  private:
    int _numberOfColorAttachments;
  };
}

#endif

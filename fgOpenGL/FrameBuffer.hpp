// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.hpp"

namespace GL {
  static bool IsFrameBuffer(GLuint i) noexcept { return glIsFramebuffer(i) == GL_TRUE; };

  struct FrameBuffer : Ref<&IsFrameBuffer>
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteFramebuffers(1, &i); };

    explicit constexpr FrameBuffer(GLuint buffer) noexcept : Ref(buffer), _numberOfColorAttachments(0) {}
    explicit constexpr FrameBuffer(FG_Resource buffer) noexcept :
      Ref(static_cast<GLuint>(buffer)), _numberOfColorAttachments(0)
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(is_valid());
#endif
    }
    constexpr FrameBuffer() noexcept : _numberOfColorAttachments(0) {}
    constexpr FrameBuffer(const FrameBuffer&)           = default;
    constexpr ~FrameBuffer() noexcept                   = default;

    GLExpected<BindRef> bind(GLenum target) const noexcept;

    FrameBuffer& operator=(const FrameBuffer&) = default;

    // This does not take ownership of the texture
    static GLExpected<Owned<FrameBuffer>> create(GLenum target, GLenum type, int level, int zoffset, FG_Resource* textures,
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

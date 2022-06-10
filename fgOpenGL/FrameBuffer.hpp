// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.hpp"

namespace GL {
  static constexpr bool IsFrameBuffer(GLuint i) noexcept { return glIsFramebuffer(i) == GL_TRUE; };
  static constexpr void DeleteFrameBuffer(GLuint i) noexcept { glDeleteFramebuffers(1, &i); };
  static constexpr void UnbindFrameBuffer(GLenum target) noexcept { glBindFramebuffer(target, 0); };

  struct FrameBuffer : Ref<&IsFrameBuffer, &DeleteFrameBuffer>
  {
    typedef BindRef<&UnbindFrameBuffer> GLFrameBufferBindRef;

    explicit constexpr FrameBuffer(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr FrameBuffer(void* shader) noexcept : Ref(shader) {}
    constexpr FrameBuffer() noexcept                    = default;
    constexpr FrameBuffer(FrameBuffer&& right) noexcept = default;
    constexpr FrameBuffer(const FrameBuffer&)           = delete;
    constexpr ~FrameBuffer() noexcept                   = default;
    GLExpected<GLFrameBufferBindRef> bind(GLenum target) const noexcept;

    FrameBuffer& operator=(const FrameBuffer&) = delete;
    FrameBuffer& operator=(FrameBuffer&& right) noexcept = default;

    // This does not take ownership of the texture
    static GLExpected<FrameBuffer> create(GLenum target, const Texture& texture, GLenum attach = GL_COLOR_ATTACHMENT0,
                                          GLenum type = GL_TEXTURE_2D, int level = 0, int zoffset = 0) noexcept;
  };
}

#endif

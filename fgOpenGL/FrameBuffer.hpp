// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.hpp"

namespace GL {
  static bool IsFrameBuffer(GLuint i) noexcept { return glIsFramebuffer(i) == GL_TRUE; };
  static void DeleteFrameBuffer(GLuint i) noexcept { glDeleteFramebuffers(1, &i); };
  static void UnbindFrameBuffer(GLenum target) noexcept { glBindFramebuffer(target, 0); };

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
    static GLExpected<FrameBuffer> create(GLenum target, GLenum type, int level, int zoffset, std::vector<const Texture*>& textures) noexcept;

    GLExpected<FrameBuffer> attach(GLenum target, GLenum type, int level, int zoffset, std::vector<const Texture*>& textures) noexcept;

  private:
    int NumberOfColorAttachments = 0;
  };
}

#endif

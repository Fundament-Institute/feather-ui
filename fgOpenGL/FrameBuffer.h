// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.h"

namespace GL {
  static constexpr bool IsFrameBuffer(GLuint i) noexcept { return glIsFramebuffer(i) == GL_TRUE; };
  static constexpr void DeleteFrameBuffer(GLuint i) noexcept { glDeleteFramebuffers(1, &i); };
  static constexpr void UnbindFrameBuffer(GLenum target) noexcept { glBindFramebuffer(target, 0); };

  struct FrameBuffer : GLRef<&IsFrameBuffer, &DeleteFrameBuffer>
  {
    typedef GLBindRef<&UnbindFrameBuffer> GLFrameBufferBindRef;

    explicit constexpr FrameBuffer(GLuint shader) noexcept : GLRef(shader) {}
    explicit constexpr FrameBuffer(void* shader) noexcept : GLRef(shader) {}
    constexpr FrameBuffer() noexcept                    = default;
    constexpr FrameBuffer(FrameBuffer&& right) noexcept = default;
    constexpr FrameBuffer(const FrameBuffer&)           = delete;
    constexpr ~FrameBuffer() noexcept                   = default;
    GLExpected<GLFrameBufferBindRef> bind(GLenum target) const noexcept
    {
      glBindFramebuffer(target, _ref);
      GL_ERROR("glBindFramebuffer");
      return {};
    }

    FrameBuffer& operator=(const FrameBuffer&) = delete;
    FrameBuffer& operator=(FrameBuffer&& right) noexcept = default;

    // This does not take ownership of the texture
    static GLExpected<FrameBuffer> create(GLenum target, const Texture& texture, GLenum attach = GL_COLOR_ATTACHMENT0,
                                          GLenum type = GL_TEXTURE_2D, int level = 0) noexcept
    {
      // TODO: Default to GL_DRAW_FRAMEBUFFER?
      assert(glFramebufferTexture2D != nullptr);

      GLuint fbgl;
      glGenFramebuffers(1, &fbgl);
      GL_ERROR("glGenFramebuffers");
      FrameBuffer fb(fbgl);
      if(auto e = fb.bind(target))
      {
        switch(type)
        {
        case GL_TEXTURE_1D: glFramebufferTexture1D(target, attach, GL_TEXTURE_1D, texture, level); break;
        case GL_TEXTURE_3D: glFramebufferTexture3D(target, attach, GL_TEXTURE_3D, texture, level); break;
        default: glFramebufferTexture2D(target, attach, type, texture, level); break;
        }
        GL_ERROR("glFramebufferTexture");
        auto status = glCheckFramebufferStatus(target);
        if(status != GL_FRAMEBUFFER_COMPLETE)
        {
          return GLError(status, "glCheckFramebufferStatus");
        }
      }
      else
        return std::move(e.error());

      return fb;
    }
  };
}

#endif

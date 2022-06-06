// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__FRAMEBUFFER_H
#define GL__FRAMEBUFFER_H

#include "Texture.h"

namespace GL {
  static constexpr bool IsFrameBuffer(GLuint i) noexcept { return glIsFramebuffer(i) == GL_TRUE; };
  static constexpr void DeleteFrameBuffer(GLuint i) noexcept { glDeleteFramebuffers(1, &i); };
  template<GLenum TARGET> constexpr void UnbindFrameBuffer() noexcept { glBindFramebuffer(TARGET, 0); };
  struct ShaderObject;

  struct FrameBuffer : GLRef<&IsFrameBuffer, &DeleteFrameBuffer>
  {
    template<GLenum TARGET = GL_FRAMEBUFFER> struct GLFrameBufferBindRef : GLBindRef<&UnbindFrameBuffer<TARGET>>
    {
      constexpr GLFrameBufferBindRef() noexcept                             = default;
      constexpr GLFrameBufferBindRef(GLFrameBufferBindRef&& right) noexcept = default;
      GLFrameBufferBindRef(const GLFrameBufferBindRef&)                     = delete;
      constexpr GLFrameBufferBindRef& operator=(GLFrameBufferBindRef&& right) noexcept = default;
      GLFrameBufferBindRef& operator=(const GLFrameBufferBindRef&) = delete;

      bool is_valid() const noexcept { return glCheckFramebufferStatus(TARGET) == GL_FRAMEBUFFER_COMPLETE; }
    };

    explicit constexpr FrameBuffer(GLuint shader) noexcept : GLRef(shader) {}
    explicit constexpr FrameBuffer(void* shader) noexcept : GLRef(shader) {}
    constexpr FrameBuffer() noexcept                    = default;
    constexpr FrameBuffer(FrameBuffer&& right) noexcept = default;
    constexpr FrameBuffer(const FrameBuffer&)           = delete;
    constexpr ~FrameBuffer() noexcept                   = default;
    template<GLenum TARGET = GL_FRAMEBUFFER> GLExpected<GLFrameBufferBindRef<TARGET>> bind() const noexcept
    {
      glBindFramebuffer(TARGET, _ref);
      GL_ERROR("glBindFramebuffer");
      return {};
    }

    FrameBuffer& operator=(const FrameBuffer&) = delete;
    FrameBuffer& operator=(FrameBuffer&& right) noexcept = default;

    // This does not take ownership of the texture
    template<GLenum TARGET = GL_FRAMEBUFFER>
    static GLExpected<FrameBuffer> create(const Texture& texture, GLenum attach = GL_COLOR_ATTACHMENT0,
                                          GLenum type = GL_TEXTURE_2D, int level = 0)
    {
      assert(glFramebufferTexture2D != nullptr);

      GLuint fbgl;
      glGenFramebuffers(1, &fbgl);
      GL_ERROR("glGenFramebuffers");
      FrameBuffer fb(fbgl);
      if(auto e = fb.bind<TARGET>())
      {
        switch(type)
        {
        case GL_TEXTURE_1D: glFrameBufferTexture1D(TARGET, attach, GL_TEXTURE_1D, texture, level); break;
        case GL_TEXTURE_3D: glFrameBufferTexture3D(TARGET, attach, GL_TEXTURE_3D, texture, level); break;
        default: glFrameBufferTexture2D(TARGET, attach, type, texture, level); break;
        }
        GL_ERROR("glFramebufferTexture");
        auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
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

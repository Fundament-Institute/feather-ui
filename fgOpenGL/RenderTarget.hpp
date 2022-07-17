// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

/*
#ifndef GL__RENDERTARGET_H
#define GL__RENDERTARGET_H

#include "Ref.hpp"

namespace GL {

  struct RenderTarget : Ref<&IsFrameBuffer, &DeleteFrameBuffer>
  {
    typedef BindRef<&UnbindFrameBuffer> GLFramebufferBindRef;

    
    explicit constexpr RenderTarget(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr RenderTarget(void* shader) noexcept : Ref(shader) {}
    constexpr RenderTarget() noexcept               = default;
    constexpr RenderTarget(RenderTarget&& right) noexcept = default;
    constexpr RenderTarget(const RenderTarget&)           = delete;
    constexpr ~RenderTarget() noexcept        = default;
    

    GLExpected<GLFramebufferBindRef> bind() const noexcept
    {
      glBindFramebuffer(GL_FRAMEBUFFER, _ref);
      GL_ERROR("glBindFramebuffer");
      return GLFramebufferBindRef(GL_FRAMEBUFFER);
    }


    RenderTarget& operator=(const RenderTarget&) = delete;
    RenderTarget& operator=(RenderTarget&& right) noexcept = default;
    

    static GLExpected<RenderTarget> create(FG_Vec2i size)
    {
      GLuint fbgl;
      glGenFramebuffers(1, &fbgl);
      GL_ERROR("glGenFramebuffers");

      RenderTarget rt(fbgl);
      if(auto e = rt.bind())
      {
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        GL_ERROR("glFramebufferTexture");

        auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE)
        {
          return CUSTOM_ERROR(status, "glCheckFramebufferStatus");
        }
      }
      else
        return std::move(e.error());


      //GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
      //glDrawBuffers(1, DrawBuffers);



      //glBindBuffer(GL_FRAMEBUFFER, 0);

      return rt;
    }
  };
}

#endif
*/
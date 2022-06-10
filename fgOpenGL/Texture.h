// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__TEXTURE_H
#define GL__TEXTURE_H

#include "Ref.h"
#include "Format.h"

namespace GL {
  static constexpr bool IsTexture(GLuint i) noexcept { return glIsTexture(i) == GL_TRUE; };
  static constexpr void DeleteTexture(GLuint i) noexcept { glDeleteTextures(1, &i); };
  constexpr void UnbindTexture(GLenum target) noexcept { glBindTexture(target, 0); };
  struct ShaderObject;

  union Filter
  {
    struct
    {
      uint8_t mip_filter : 2;
      uint8_t mag_filter : 2;
      uint8_t min_filter : 2;
      uint8_t anisotropic : 1;
      uint8_t comparison : 1;
    };
    uint8_t value;

    static GLenum GLFilter(bool mip, bool other);
  };

  static_assert(sizeof(Filter) == sizeof(uint8_t), "Filter is unexpected size!");

  struct Texture : Ref<&IsTexture, &DeleteTexture>
  {
    struct GLTextureBindRef : BindRef<&UnbindTexture>
    {
      constexpr GLTextureBindRef() noexcept                         = delete;
      constexpr GLTextureBindRef(GLenum target) noexcept : BindRef(target) {}
      constexpr GLTextureBindRef(GLTextureBindRef&& right) noexcept = default;
      GLTextureBindRef(const GLTextureBindRef&)                     = delete;
      constexpr GLTextureBindRef& operator=(GLTextureBindRef&& right) noexcept = default;
      GLTextureBindRef& operator=(const GLTextureBindRef&) = delete;

      template<class T>
      GLExpected<void> set(GLenum parameter, T value) noexcept(std::is_nothrow_convertible_v<T, GLint> ||
                                                               std::is_nothrow_convertible_v<T, GLfloat>)
      {
        static_assert(!(std::is_convertible_v<T, GLint> && std::is_convertible_v<T, GLfloat>),
                      "T can be converted to either float or int!");
        static_assert(std::is_convertible_v<T, GLint> || std::is_convertible_v<T, GLfloat>,
                      "T must be convertible to either float or int!");

        if constexpr(std::is_convertible_v<T, GLint>)
        {
          glTexParameteri(_target, parameter, value);
        }
        else if constexpr(std::is_convertible_v<T, GLfloat>)
        {
          glTexParameterf(_target, parameter, value);
        }
      }

      template<class T>
      GLExpected<void> setv(GLenum parameter, T* value) noexcept(std::is_nothrow_convertible_v<T, GLint> ||
                                                                 std::is_nothrow_convertible_v<T, GLfloat>)
      {
        static_assert(!(std::is_convertible_v<T, GLint> && std::is_convertible_v<T, GLfloat>),
                      "T can be converted to either float or int!");
        static_assert(std::is_convertible_v<T, GLint> || std::is_convertible_v<T, GLfloat>,
                      "T must be convertible to either float or int!");

        if constexpr(std::is_convertible_v<T, GLint>)
        {
          glTexParameteriv(_target, parameter, value);
        }
        else if constexpr(std::is_convertible_v<T, GLfloat>)
        {
          glTexParameterfv(_target, parameter, value);
        }
      }

      GLExpected<void> apply_sampler(const FG_Sampler& sampler) { return Texture::apply_sampler(_target, sampler); }
    };

    explicit constexpr Texture(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr Texture(void* shader) noexcept : Ref(shader) {}
    constexpr Texture() noexcept                = default;
    constexpr Texture(Texture&& right) noexcept = default;
    constexpr Texture(const Texture&)           = delete;
    constexpr ~Texture() noexcept               = default;
    GLExpected<GLTextureBindRef> bind(GLenum target) const noexcept
    {
      glBindTexture(target, _ref);
      GL_ERROR("glBindTexture");
      return GLTextureBindRef(target);
    }

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&& right) noexcept = default;

    static GLExpected<Texture> create2D(GLenum target, Format format, FG_Vec2i size, const FG_Sampler& sampler,
                                        void* data = nullptr,
                                        int levelorsamples = 0)
    {
      GLuint texgl;
      glGenTextures(1, &texgl);
      GL_ERROR("glGenTextures");
      Texture tex(texgl);
      if(auto bind = tex.bind(target))
      {
        if(target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_PROXY_TEXTURE_2D_MULTISAMPLE)
          glTexImage2DMultisample(target, levelorsamples, format.internalformat, size.x, size.y, GL_FALSE);
        else
          glTexImage2D(target, levelorsamples, format.internalformat, size.x, size.y, 0, format.components,
                       format.type, data);

        GL_ERROR("glTexImage2D");
        if(auto e = bind.value().apply_sampler(sampler)) {}
        else
          return std::move(e.error());
      }
      else
        return std::move(bind.error());

      return tex;
    }

  private:
    static GLExpected<void> apply_sampler(GLenum target, const FG_Sampler& sampler);
  };
}

#endif

// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__TEXTURE_H
#define GL__TEXTURE_H

#include "GLRef.h"
#include "GLFormat.h"

namespace GL {
  static constexpr bool IsTexture(GLuint i) noexcept { return glIsTexture(i) == GL_TRUE; };
  static constexpr void DeleteTexture(GLuint i) noexcept { glDeleteTextures(1, &i); };
  template<GLenum TARGET> constexpr void UnbindTexture() noexcept { glBindTexture(TARGET, 0); };
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

  struct Texture : GLRef<&IsTexture, &DeleteTexture>
  {
    template<GLenum TARGET = GL_TEXTURE_2D> struct GLTextureBindRef : GLBindRef<&UnbindTexture<TARGET>>
    {
      constexpr GLTextureBindRef() noexcept                         = default;
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

        if constexpr(std::is_convertible_v<T, GLint>)
        {
          glTexParameteri(TARGET, parameter, value);
        }
        else if constexpr(std::is_convertible_v<T, GLfloat>)
        {
          glTexParameterf(TARGET, parameter, value);
        }
        else
        {
          static_assert(false, "T must be convertible to either float or int!");
        }
      }

      template<class T> GLExpected<void> set<T*>(GLenum parameter, T* value)
      {
        static_assert(!(std::is_convertible_v<T, GLint> && std::is_convertible_v<T, GLfloat>),
                      "T can be converted to either float or int!");

        if constexpr(std::is_convertible_v<T, GLint>)
        {
          glTexParameteriv(TARGET, parameter, value);
        }
        else if constexpr(std::is_convertible_v<T, GLfloat>)
        {
          glTexParameterfv(TARGET, parameter, value);
        }
        else
        {
          static_assert(false, "T must be convertible to either float or int!");
        }
      }

      GLExpected<void> apply_sampler(const FG_Sampler& sampler) { return Texture::apply_sampler(TARGET, sampler); }
    };

    explicit constexpr Texture(GLuint shader) noexcept : GLRef(shader) {}
    explicit constexpr Texture(void* shader) noexcept : GLRef(shader) {}
    constexpr Texture() noexcept                = default;
    constexpr Texture(Texture&& right) noexcept = default;
    constexpr Texture(const Texture&)           = delete;
    constexpr ~Texture() noexcept               = default;
    template<GLenum TARGET = GL_TEXTURE_2D> GLExpected<GLTextureBindRef<TARGET>> bind() const noexcept
    {
      glBindTexture(TARGET, _ref);
      GL_ERROR("glBindTexture");
      return {};
    }

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&& right) noexcept = default;

    template<GLenum TARGET = GL_TEXTURE_2D>
    static GLExpected<Texture> create2D(GLFormat format, FG_Vec2i size, const FG_Sampler& sampler, void* data = nullptr,
                                        int level = 0)
    {
      GLuint texgl;
      glGenTextures(1, &texgl);
      GL_ERROR("glGenTextures");
      Texture tex(texgl);
      if(auto bind = tex.bind<TARGET>())
      {
        glTexImage2D(GL_TEXTURE_2D, level, format.internalformat, size.x, size.y, 0, format.components, format.type, data);
        GL_ERROR("glTexImage2D");
        if(auto e = bind.apply_sampler(sampler)) {}
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

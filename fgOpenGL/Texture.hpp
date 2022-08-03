// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#ifndef GL__TEXTURE_H
#define GL__TEXTURE_H

#include "Ref.hpp"
#include "Format.hpp"

namespace GL {
  static bool IsTexture(GLuint i) noexcept { return glIsTexture(i) == GL_TRUE; };

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

  struct Texture : Ref<&IsTexture>
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteTextures(1, &i); };

    struct TextureBindRef : BindRef
    {
      constexpr TextureBindRef() noexcept = delete;
      constexpr TextureBindRef(GLenum target, BINDRESET reset) noexcept : BindRef(target, reset) {}
      constexpr TextureBindRef(TextureBindRef&& right) noexcept = default;
      TextureBindRef(const TextureBindRef&)                       = delete;
      TextureBindRef& operator=(TextureBindRef&& right) noexcept = default;
      TextureBindRef& operator=(const TextureBindRef&) = delete;

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

    explicit constexpr Texture(GLuint tex) noexcept : Ref(tex) {}
    explicit constexpr Texture(FG_Resource tex) noexcept : Ref(static_cast<GLuint>(tex))
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(is_valid());
#endif
    }
    constexpr Texture() noexcept                = default;
    constexpr Texture(const Texture&) noexcept = default;
    constexpr ~Texture() noexcept               = default;
    GLExpected<TextureBindRef> bind(GLenum target) const noexcept;

    Texture& operator=(const Texture&) noexcept = default;

    static GLExpected<Owned<Texture>> create2D(GLenum target, Format format, FG_Vec2i size, const FG_Sampler& sampler,
                                               void* data = nullptr, int levelorsamples = 0);

  private:
    static GLExpected<void> apply_sampler(GLenum target, const FG_Sampler& sampler);
  };
}

#endif

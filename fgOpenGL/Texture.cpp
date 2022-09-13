// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProviderGL.hpp"
#include "Texture.hpp"
#include "EnumMapping.hpp"
#include <cassert>
#include <cmath>

using namespace GL;

static GLenum GLFilter(bool mip, bool other)
{
  if(mip && other)
    return GL_LINEAR_MIPMAP_LINEAR;
  if(mip)
    return GL_NEAREST_MIPMAP_LINEAR;
  if(other)
    return GL_LINEAR_MIPMAP_NEAREST;
  return GL_NEAREST_MIPMAP_NEAREST;
}

GLExpected<Texture::TextureBindRef> Texture::bind(GLenum target) const noexcept
{
  glBindTexture(target, _ref);
  GL_ERROR("glBindTexture");
  return Texture::TextureBindRef{ target, [](GLenum target) { glBindTexture(target, 0); } };
}

GLExpected<void> Texture::apply_sampler(GLenum target, const FG_Sampler& sampler)
{
  Filter f;
  f.value = sampler.filter;

  // TODO: figure out when the texture has a mipmap
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, f.mag_filter ? GL_LINEAR : GL_NEAREST);
  GL_ERROR("glTexParameteri");
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, f.min_filter ? GL_LINEAR : GL_NEAREST);
  GL_ERROR("glTexParameteri");

  glTexParameterf(target, GL_TEXTURE_MAX_LOD, sampler.max_lod);
  GL_ERROR("glTexParameterf");
  glTexParameterf(target, GL_TEXTURE_MIN_LOD, sampler.min_lod);
  GL_ERROR("glTexParameterf");
  glTexParameterf(target, GL_TEXTURE_LOD_BIAS, sampler.mip_bias);
  GL_ERROR("glTexParameterf");

  if(f.anisotropic)
  {
    glTexParameterf(target, GL_TEXTURE_MAX_ANISOTROPY, sampler.max_anisotropy);
    GL_ERROR("glTexParameterf");
  }

  if(sampler.comparison != FG_Comparison_Disabled && f.comparison)
  {
    glTexParameteri(target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    GL_ERROR("glTexParameteri");
    if(sampler.comparison >= ArraySize(ComparisonMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Comparison outside of bounds");

    glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, ComparisonMapping[sampler.comparison]);
    GL_ERROR("glTexParameteri");
  }

  return {};
}

GLExpected<Owned<Texture>> Texture::create2D(GLenum target, Format format, FG_Vec2i size, const FG_Sampler& sampler,
                                             void* data, int levelorsamples)
{
  GLuint texgl;
  glGenTextures(1, &texgl);
  GL_ERROR("glGenTextures");
  Owned<Texture> tex(texgl);
  if(auto bind = tex.bind(target))
  {
    if(target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_PROXY_TEXTURE_2D_MULTISAMPLE)
    {
      glTexImage2DMultisample(target, levelorsamples, format.internalformat, size.x, size.y, GL_FALSE);
      GL_ERROR("glTexImage2DMultisample");
    }
    else
    {
      glTexImage2D(target, levelorsamples, format.internalformat, size.x, size.y, 0, format.components, format.type, data);
      GL_ERROR("glTexImage2D");
    }

    RETURN_ERROR(bind.value().apply_sampler(sampler));
  }
  else
    return std::move(bind.error());

  return tex;
}
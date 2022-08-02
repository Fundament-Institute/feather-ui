// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "BackendGL.hpp"
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

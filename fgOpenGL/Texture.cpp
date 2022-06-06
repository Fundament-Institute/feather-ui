// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#include "BackendGL.h"
#include "Texture.h"
#include "EnumMapping.h"
#include <assert.h>
#include <math.h>

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
  f.value = sampler.Filter;

  // TODO: figure out when the texture has a mipmap
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, f.mag_filter ? GL_LINEAR : GL_NEAREST);
  GL_ERROR("glTexParameteri");
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, f.min_filter ? GL_LINEAR : GL_NEAREST);
  GL_ERROR("glTexParameteri");


  glTexParameterf(target, GL_TEXTURE_MAX_LOD, sampler.MaxLOD);
  GL_ERROR("glTexParameterf");
  glTexParameterf(target, GL_TEXTURE_MIN_LOD, sampler.MinLOD);
  GL_ERROR("glTexParameterf");
  glTexParameterf(target, GL_TEXTURE_LOD_BIAS, sampler.MipBias);
  GL_ERROR("glTexParameterf");

  if(f.anisotropic)
  {
    glTexParameterf(target, GL_TEXTURE_MAX_ANISOTROPY, sampler.MaxAnisotropy);
    GL_ERROR("glTexParameterf");
  }

  if(sampler.Comparison != FG_COMPARISON_DISABLED && f.comparison)
  {
    glTexParameteri(target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    GL_ERROR("glTexParameteri");
    if(sampler.Comparison >= ArraySize(ComparisonMapping))
      return GLError(ERR_INVALID_PARAMETER, "Comparison outside of bounds");

    glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, ComparisonMapping[sampler.Comparison]);
    GL_ERROR("glTexParameteri");
  }
   
}

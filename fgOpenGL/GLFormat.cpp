// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#include "GLFormat.h"
#include <assert.h>
#include <math.h>

using namespace GL;

// https://github.com/richardmg/dachshund/blob/7f5404b89aab84ae77995725d2b408109f514479/src/3rdparty/angle/src/libANGLE/es3_format_type_combinations.json

constexpr GLFormat GLFormat::Map(GLint internalformat) noexcept
{
  // The standard allows multiple valid type combinations for certain formats. We pick the most specific
  // type possible. The alternatives are left commented out in case we need to implement them as fallbacks.
  // Some formats are commented out because we don't have the extensions to support them enabled.
  switch(internalformat)
  {
  case GL_RGBA8: return GLFormat{ GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGB5_A1: return GLFormat{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGBA4: return GLFormat{ GL_RGBA4, GL_RGBA, GL_UNSIGNED_BYTE };
  case GL_SRGB8_ALPHA8: return GLFormat{ GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE };
  case GL_RGBA8_SNORM: return GLFormat{ GL_RGBA8_SNORM, GL_RGBA, GL_BYTE };
  case GL_RGBA4: return GLFormat{ GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4 };
  case GL_RGB10_A2: return GLFormat{ GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  // case GL_RGB5_A1: return GLFormat{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  case GL_RGB5_A1: return GLFormat{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case GL_RGBA16F: return GLFormat{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT };
  // case GL_RGBA16F: return GLFormat{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT_OES };
  case GL_RGBA32F: return GLFormat{ GL_RGBA32F, GL_RGBA, GL_FLOAT };
  // case GL_RGBA16F: return GLFormat{ GL_RGBA16F, GL_RGBA, GL_FLOAT };
  case GL_RGBA8UI: return GLFormat{ GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RGBA8I: return GLFormat{ GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE };
  case GL_RGBA16UI: return GLFormat{ GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RGBA16I: return GLFormat{ GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT };
  case GL_RGBA32UI: return GLFormat{ GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT };
  case GL_RGBA32I: return GLFormat{ GL_RGBA32I, GL_RGBA_INTEGER, GL_INT };
  case GL_RGB10_A2UI: return GLFormat{ GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV };
  case GL_RGB8: return GLFormat{ GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE };
  // case GL_RGB565: return GLFormat{ GL_RGB565, GL_RGB, GL_UNSIGNED_BYTE };
  case GL_SRGB8: return GLFormat{ GL_SRGB8, GL_RGB, GL_UNSIGNED_BYTE };
  case GL_RGB8_SNORM: return GLFormat{ GL_RGB8_SNORM, GL_RGB, GL_BYTE };
  // case GL_RGB565: return GLFormat{ GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case GL_R11F_G11F_B10F: return GLFormat{ GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV };
  case GL_RGB9_E5: return GLFormat{ GL_RGB9_E5, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV };
  case GL_RGB16F: return GLFormat{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT };
  // case GL_RGB16F: return GLFormat{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT_OES };
  // case GL_R11F_G11F_B10F: return GLFormat{ GL_R11F_G11F_B10F, GL_RGB, GL_HALF_FLOAT };
  // case GL_R11F_G11F_B10F: return GLFormat{ GL_R11F_G11F_B10F, GL_RGB, GL_HALF_FLOAT_OES };
  // case GL_RGB9_E5: return GLFormat{ GL_RGB9_E5, GL_RGB, GL_HALF_FLOAT };
  // case GL_RGB9_E5: return GLFormat{ GL_RGB9_E5, GL_RGB, GL_HALF_FLOAT_OES };
  case GL_RGB32F: return GLFormat{ GL_RGB32F, GL_RGB, GL_FLOAT };
  // case GL_RGB16F: return GLFormat{ GL_RGB16F, GL_RGB, GL_FLOAT };
  // case GL_R11F_G11F_B10F: return GLFormat{ GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT };
  // case GL_RGB9_E5: return GLFormat{ GL_RGB9_E5, GL_RGB, GL_FLOAT };
  case GL_RGB8UI: return GLFormat{ GL_RGB8UI, GL_RGB_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RGB8I: return GLFormat{ GL_RGB8I, GL_RGB_INTEGER, GL_BYTE };
  case GL_RGB16UI: return GLFormat{ GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RGB16I: return GLFormat{ GL_RGB16I, GL_RGB_INTEGER, GL_SHORT };
  case GL_RGB32UI: return GLFormat{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case GL_RGB32I: return GLFormat{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case GL_RG8: return GLFormat{ GL_RG8, GL_RG, GL_UNSIGNED_BYTE };
  case GL_RG8_SNORM: return GLFormat{ GL_RG8_SNORM, GL_RG, GL_BYTE };
  case GL_RG16F: return GLFormat{ GL_RG16F, GL_RG, GL_HALF_FLOAT };
  // case GL_RG16F: return GLFormat{ GL_RG16F, GL_RG, GL_HALF_FLOAT_OES };
  case GL_RG32F: return GLFormat{ GL_RG32F, GL_RG, GL_FLOAT };
  // case GL_RG16F: return GLFormat{ GL_RG16F, GL_RG, GL_FLOAT };
  case GL_RG8UI: return GLFormat{ GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RG8I: return GLFormat{ GL_RG8I, GL_RG_INTEGER, GL_BYTE };
  case GL_RG16UI: return GLFormat{ GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RG16I: return GLFormat{ GL_RG16I, GL_RG_INTEGER, GL_SHORT };
  case GL_RG32UI: return GLFormat{ GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT };
  case GL_RG32I: return GLFormat{ GL_RG32I, GL_RG_INTEGER, GL_INT };
  case GL_R8: return GLFormat{ GL_R8, GL_RED, GL_UNSIGNED_BYTE };
  case GL_R8_SNORM: return GLFormat{ GL_R8_SNORM, GL_RED, GL_BYTE };
  case GL_R16F: return GLFormat{ GL_R16F, GL_RED, GL_HALF_FLOAT };
  // case GL_R16F: return GLFormat{ GL_R16F, GL_RED, GL_HALF_FLOAT_OES };
  case GL_R32F: return GLFormat{ GL_R32F, GL_RED, GL_FLOAT };
  // case GL_R16F: return GLFormat{ GL_R16F, GL_RED, GL_FLOAT };
  case GL_R8UI: return GLFormat{ GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE };
  case GL_R8I: return GLFormat{ GL_R8I, GL_RED_INTEGER, GL_BYTE };
  case GL_R16UI: return GLFormat{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case GL_R16I: return GLFormat{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case GL_R32UI: return GLFormat{ GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT };
  case GL_R32I: return GLFormat{ GL_R32I, GL_RED_INTEGER, GL_INT };
  case GL_RGBA: return GLFormat{ GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGBA: return GLFormat{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4 };
  // case GL_RGBA: return GLFormat{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case GL_RGB: return GLFormat{ GL_RGB, GL_RGB, GL_UNSIGNED_BYTE };
  // case GL_RGB: return GLFormat{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case GL_LUMINANCE_ALPHA: return GLFormat{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE };
  case GL_LUMINANCE: return GLFormat{ GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE };
  case GL_ALPHA: return GLFormat{ GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE };
  // case GL_SRGB_ALPHA_EXT: return GLFormat{ GL_SRGB_ALPHA_EXT, GL_SRGB_ALPHA_EXT, GL_UNSIGNED_BYTE };
  // case GL_SRGB_EXT: return GLFormat{ GL_SRGB_EXT, GL_SRGB_EXT, GL_UNSIGNED_BYTE };
  case GL_RG: return GLFormat{ GL_RG, GL_RG, GL_UNSIGNED_BYTE };
  // case GL_RG: return GLFormat{ GL_RG, GL_RG, GL_FLOAT };
  // case GL_RG: return GLFormat{ GL_RG, GL_RG, GL_HALF_FLOAT };
  // case GL_RG: return GLFormat{ GL_RG, GL_RG, GL_HALF_FLOAT_OES };
  case GL_RED: return GLFormat{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  // case GL_RED: return GLFormat{ GL_RED, GL_RED, GL_FLOAT };
  // case GL_RED: return GLFormat{ GL_RED, GL_RED, GL_HALF_FLOAT };
  // case GL_RED: return GLFormat{ GL_RED, GL_RED, GL_HALF_FLOAT_OES };
  case GL_DEPTH_STENCIL: return GLFormat{ GL_DEPTH_STENCIL, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case GL_DEPTH_COMPONENT16: return GLFormat{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case GL_DEPTH_COMPONENT24: return GLFormat{ GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  // case GL_DEPTH_COMPONENT16: return GLFormat{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  case GL_DEPTH_COMPONENT32F: return GLFormat{ GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT };
  case GL_DEPTH24_STENCIL8: return GLFormat{ GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case GL_DEPTH32F_STENCIL8: return GLFormat{ GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV };
  // case GL_DEPTH_COMPONENT32_OES: return GLFormat{ GL_DEPTH_COMPONENT32_OES, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8 };
  // case GL_DEPTH_COMPONENT: return GLFormat{ GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case GL_DEPTH_COMPONENT:
    return GLFormat{ GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
    // case GL_SRGB8_ALPHA8_EXT: return GLFormat{ GL_SRGB8_ALPHA8_EXT, GL_SRGB_ALPHA_EXT, GL_UNSIGNED_BYTE };
    /* case GL_SRGB8:
      return GLFormat{ GL_SRGB8, GL_SRGB_EXT, GL_UNSIGNED_BYTE };
    case GL_RGBA: return GLFormat{ GL_RGBA, GL_RGBA, GL_FLOAT };
    case GL_RGB: return GLFormat{ GL_RGB, GL_RGB, GL_FLOAT };
    case GL_LUMINANCE_ALPHA: return GLFormat{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_FLOAT };
    case GL_LUMINANCE: return GLFormat{ GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT };
    case GL_ALPHA: return GLFormat{ GL_ALPHA, GL_ALPHA, GL_FLOAT };
    case GL_RGBA: return GLFormat{ GL_RGBA, GL_RGBA, GL_HALF_FLOAT_OES };
    case GL_RGB: return GLFormat{ GL_RGB, GL_RGB, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE_ALPHA: return GLFormat{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT };
    case GL_LUMINANCE_ALPHA: return GLFormat{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE: return GLFormat{ GL_LUMINANCE, GL_LUMINANCE, GL_HALF_FLOAT };
    case GL_LUMINANCE: return GLFormat{ GL_LUMINANCE, GL_LUMINANCE, GL_HALF_FLOAT_OES };
    case GL_ALPHA: return GLFormat{ GL_ALPHA, GL_ALPHA, GL_HALF_FLOAT };
    case GL_ALPHA: return GLFormat{ GL_ALPHA, GL_ALPHA, GL_HALF_FLOAT_OES };
    case GL_BGRA_EXT: return GLFormat{ GL_BGRA_EXT, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_ALPHA8_EXT: return GLFormat{ GL_ALPHA8_EXT, GL_ALPHA, GL_UNSIGNED_BYTE };
    case GL_LUMINANCE8_EXT: return GLFormat{ GL_LUMINANCE8_EXT, GL_LUMINANCE, GL_UNSIGNED_BYTE };
    case GL_LUMINANCE8_ALPHA8_EXT: return GLFormat{ GL_LUMINANCE8_ALPHA8_EXT, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE };
    case GL_ALPHA32F_EXT: return GLFormat{ GL_ALPHA32F_EXT, GL_ALPHA, GL_FLOAT };
    case GL_LUMINANCE32F_EXT: return GLFormat{ GL_LUMINANCE32F_EXT, GL_LUMINANCE, GL_FLOAT };
    case GL_LUMINANCE_ALPHA32F_EXT: return GLFormat{ GL_LUMINANCE_ALPHA32F_EXT, GL_LUMINANCE_ALPHA, GL_FLOAT };
    case GL_ALPHA16F_EXT: return GLFormat{ GL_ALPHA16F_EXT, GL_ALPHA, GL_HALF_FLOAT };
    case GL_ALPHA16F_EXT: return GLFormat{ GL_ALPHA16F_EXT, GL_ALPHA, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE16F_EXT: return GLFormat{ GL_LUMINANCE16F_EXT, GL_LUMINANCE, GL_HALF_FLOAT };
    case GL_LUMINANCE16F_EXT: return GLFormat{ GL_LUMINANCE16F_EXT, GL_LUMINANCE, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE_ALPHA16F_EXT: return GLFormat{ GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT };
    case GL_LUMINANCE_ALPHA16F_EXT: return GLFormat{ GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES };
    case GL_BGRA8_EXT: return GLFormat{ GL_BGRA8_EXT, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_BGRA4_ANGLEX: return GLFormat{ GL_BGRA4_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT };
    case GL_BGRA4_ANGLEX: return GLFormat{ GL_BGRA4_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_BGR5_A1_ANGLEX: return GLFormat{ GL_BGR5_A1_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT };
    case GL_BGR5_A1_ANGLEX: return GLFormat{ GL_BGR5_A1_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_R16_EXT: return GLFormat{ GL_R16_EXT, GL_RED, GL_UNSIGNED_SHORT };
    case GL_RG16_EXT: return GLFormat{ GL_RG16_EXT, GL_RG, GL_UNSIGNED_SHORT };
    case GL_RGB16_EXT: return GLFormat{ GL_RGB16_EXT, GL_RGB, GL_UNSIGNED_SHORT };
    case GL_RGBA16_EXT: return GLFormat{ GL_RGBA16_EXT, GL_RGBA, GL_UNSIGNED_SHORT };
    case GL_R16_SNORM_EXT: return GLFormat{ GL_R16_SNORM_EXT, GL_RED, GL_SHORT };
    case GL_RG16_SNORM_EXT: return GLFormat{ GL_RG16_SNORM_EXT, GL_RG, GL_SHORT };
    case GL_RGB16_SNORM_EXT: return GLFormat{ GL_RGB16_SNORM_EXT, GL_RGB, GL_SHORT };
    case GL_RGBA16_SNORM_EXT: return GLFormat{ GL_RGBA16_SNORM_EXT, GL_RGBA, GL_SHORT };
    */
  }
}

constexpr GLFormat GLFormat::Create(uint8_t format, bool sRGB) noexcept
{
  switch(format)
  {
  case FG_PixelFormat_UNKNOWN: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R32G32B32A32_TYPELESS: return GLFormat{ GL_RGBA, GL_RGBA, GL_FLOAT };
  case FG_PixelFormat_R32G32B32A32_FLOAT: return GLFormat{ GL_RGBA32F, GL_RGBA, GL_FLOAT };
  case FG_PixelFormat_R32G32B32A32_UINT: return GLFormat{ GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32B32A32_INT: return GLFormat{ GL_RGBA32I, GL_RGBA_INTEGER, GL_INT };
  case FG_PixelFormat_R32G32B32_TYPELESS: return GLFormat{ GL_RGB, GL_RGB, GL_FLOAT };
  case FG_PixelFormat_R32G32B32_FLOAT: return GLFormat{ GL_RGB32F, GL_RGB, GL_FLOAT };
  case FG_PixelFormat_R32G32B32_UINT: return GLFormat{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32B32_INT: return GLFormat{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case FG_PixelFormat_R16G16B16A16_TYPELESS: return GLFormat{ GL_RGBA16, GL_RGBA, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16A16_FLOAT: return GLFormat{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16A16_UNORM: return GLFormat{ GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16A16_UINT: return GLFormat{ GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16A16_SNORM: return GLFormat{ GL_RGBA16_SNORM, GL_RGBA, GL_SHORT };
  case FG_PixelFormat_R16G16B16A16_INT: return GLFormat{ GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT };
  case FG_PixelFormat_R16G16B16_TYPELESS: return GLFormat{ GL_RGB16, GL_RGB, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16_FLOAT: return GLFormat{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16_UNORM: return GLFormat{ GL_RGB16, GL_RGB, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16_UINT: return GLFormat{ GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16_SNORM: return GLFormat{ GL_RGB16_SNORM, GL_RGBA, GL_SHORT };
  case FG_PixelFormat_R16G16B16_INT: return GLFormat{ GL_RGB16I, GL_RGB_INTEGER, GL_SHORT };
  case FG_PixelFormat_R32G32_TYPELESS: return GLFormat{ GL_RG, GL_RG, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32_FLOAT: return GLFormat{ GL_RG32F, GL_RG, GL_FLOAT };
  case FG_PixelFormat_R32G32_UINT: return GLFormat{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32_INT: return GLFormat{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case FG_PixelFormat_R32G8X24_TYPELESS: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_D32_FLOAT_S8X24_UINT:
    return GLFormat{ GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV };
  case FG_PixelFormat_R32_FLOAT_X8X24_TYPELESS: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_X32_TYPELESS_G8X24_UINT: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R10G10B10A2_TYPELESS: return GLFormat{ GL_RGBA, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  case FG_PixelFormat_R10G10B10A2_UNORM: return GLFormat{ GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  case FG_PixelFormat_R10G10B10A2_UINT: return GLFormat{ GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV };
  case FG_PixelFormat_R11G11B10_FLOAT: return GLFormat{ GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV };
  case FG_PixelFormat_R8G8B8A8_TYPELESS: return GLFormat{ sRGB ? GL_SRGB_ALPHA : GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_UNORM: return GLFormat{ sRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_UINT: return GLFormat{ GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_SNORM: return GLFormat{ GL_RGBA8_SNORM, GL_RGBA, GL_BYTE };
  case FG_PixelFormat_R8G8B8A8_INT: return GLFormat{ GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE };
  case FG_PixelFormat_R8G8B8X8_TYPELESS: return GLFormat{ sRGB ? GL_SRGB : GL_RGB, GL_RGB, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_UNORM: return GLFormat{ sRGB ? GL_SRGB8 : GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_SNORM: return GLFormat{ GL_RGB8_SNORM, GL_RGB, GL_BYTE };
  case FG_PixelFormat_R8G8B8X8_UINT: return GLFormat{ GL_RGB8UI, GL_RGB_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_INT: return GLFormat{ GL_RGB8I, GL_RGB_INTEGER, GL_BYTE };
  case FG_PixelFormat_R16G16_TYPELESS: return GLFormat{ GL_RG16, GL_RG, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16_FLOAT: return GLFormat{ GL_RG16F, GL_RG, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16_UNORM: return GLFormat{ GL_RG16, GL_RG, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16_UINT: return GLFormat{ GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16_SNORM: return GLFormat{ GL_RG16_SNORM, GL_RG, GL_SHORT };
  case FG_PixelFormat_R16G16_INT: return GLFormat{ GL_RG16I, GL_RG_INTEGER, GL_SHORT };
  case FG_PixelFormat_R32_TYPELESS: return GLFormat{ GL_RED, GL_RED, GL_FLOAT };
  case FG_PixelFormat_R32_FLOAT: return GLFormat{ GL_R32F, GL_RED, GL_FLOAT };
  case FG_PixelFormat_R32_UINT: return GLFormat{ GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32_INT: return GLFormat{ GL_R32I, GL_RED_INTEGER, GL_INT };
  case FG_PixelFormat_D32_FLOAT: return GLFormat{ GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_FLOAT };
  case FG_PixelFormat_D24_UNORM_X8_TYPELESS: return GLFormat{ GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  case FG_PixelFormat_R24_UNORM_X8_TYPELESS: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_D16_UNORM: return GLFormat{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R24G8_TYPELESS: return GLFormat{ GL_RG, GL_RG, GL_UNSIGNED_INT_24_8 };
  case FG_PixelFormat_D24_UNORM_S8_UINT: return GLFormat{ GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case FG_PixelFormat_X24_TYPELESS_G8_UINT: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R8G8_TYPELESS: return GLFormat{ GL_RG, GL_RG, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_UNORM: return GLFormat{ GL_RG8, GL_RG, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_UINT: return GLFormat{ GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_SNORM: return GLFormat{ GL_RG8_SNORM, GL_RG, GL_BYTE };
  case FG_PixelFormat_R8G8_INT: return GLFormat{ GL_RG8I, GL_RG_INTEGER, GL_BYTE };
  case FG_PixelFormat_R16_TYPELESS: return GLFormat{ GL_R16, GL_RED, GL_HALF_FLOAT };
  case FG_PixelFormat_R16_FLOAT: return GLFormat{ GL_R16F, GL_RED, GL_HALF_FLOAT };
  case FG_PixelFormat_R16_UNORM: return GLFormat{ GL_R16, GL_RED, GL_SHORT };
  case FG_PixelFormat_R16_UINT: return GLFormat{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16_SNORM: return GLFormat{ GL_R16_SNORM, GL_RED, GL_SHORT };
  case FG_PixelFormat_R16_INT: return GLFormat{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case FG_PixelFormat_R8_TYPELESS: return GLFormat{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8_UNORM: return GLFormat{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8_UINT: return GLFormat{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R8_SNORM: return GLFormat{ GL_R8_SNORM, GL_RED, GL_BYTE };
  case FG_PixelFormat_R8_INT: return GLFormat{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case FG_PixelFormat_A8_UNORM: return GLFormat{ GL_ALPHA8, GL_ALPHA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R1_UNORM: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R9G9B9E5_SHAREDEXP: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R8G8_B8G8_UNORM: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_G8R8_G8B8_UNORM: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_R5G6B5_UNORM: return GLFormat{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case FG_PixelFormat_R5G5B5A1_UNORM: return GLFormat{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case FG_PixelFormat_B5G6R5_UNORM: return GLFormat{ GL_BGR, GL_BGR, GL_UNSIGNED_SHORT_5_6_5_REV };
  case FG_PixelFormat_B5G5R5A1_UNORM: return GLFormat{ GL_BGRA, GL_BGRA, GL_UNSIGNED_SHORT_1_5_5_5_REV };
  case FG_PixelFormat_B8G8R8A8_UNORM: return GLFormat{ GL_BGRA, GL_BGRA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_B8G8R8X8_UNORM: return GLFormat{ GL_BGR, GL_BGR, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R10G10B10_XR_BIAS_A2_UNORM: return GLFormat{ 0, 0, 0 };
  case FG_PixelFormat_B8G8R8A8_TYPELESS: return GLFormat{ GL_BGRA, GL_BGRA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_B8G8R8X8_TYPELESS: return GLFormat{ GL_BGR, GL_BGR, GL_UNSIGNED_BYTE };
  }
  return GLFormat{ 0, 0, 0 };
}

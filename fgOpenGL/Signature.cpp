// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "BackendGL.h"
#include "Signature.h"
#include "Context.h"

using namespace GL;

Signature::Signature(Backend* backend, std::span<FG_Asset*> buffers, std::span<FG_ShaderParameter> parameters)
{
  _buffers.insert(_buffers.begin(), buffers.begin(), buffers.end());
  _parameters.insert(_parameters.begin(), parameters.begin(), parameters.end());
  _strides.resize(_buffers.size());

  for(size_t i = 0; i < _buffers.size(); ++i)
  {
    for(auto& param : _parameters)
    {      
      _strides[i] += Context::GetBytes(Context::GetShaderType(param.type)) * Context::GetMultiCount(param.length, param.multi);
    }

    if((_buffers[i]->bytes % _strides[i]) > 0)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "%u bytes can't be evenly divided by %u stride (remainder %u)!",
                       _buffers[i]->bytes, _strides[i], _buffers[i]->bytes % _strides[i]);
    }
  }
}
Signature::~Signature() {}
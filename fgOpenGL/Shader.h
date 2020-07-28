// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__SHADER_H
#define GL__SHADER_H

#include "../backend.h"
#include "khash.h"
#include <utility>

struct GLFWwindow;

namespace GL {
  struct kh_attributes_s;
  class Backend;

  struct Attribute
  {
    Attribute(const char* name, int type, float* d = nullptr);
    
    const char* name;
    int type; // GL_VEC2, GL_FLOAT, etc.
    union
    {
      float data[4]; // store up to 4 in the struct
      float* pdata;
    };
  };

  struct Shader
  {
    enum KIND : unsigned int
    {
      GLOBAL = 0,
      VERTEX,
      PIXEL,
      MAX,
    };

    struct Data
    {
      KIND kind;
      int type;
      const char* name;
      ptrdiff_t offset; // For vertex data
    };

    Shader() : _pixel(nullptr), _vertex(nullptr), _geometry(nullptr), _layout(nullptr) {}
    Shader(const Shader& copy);
    Shader(Shader&& mov) : _pixel(mov._pixel), _vertex(mov._vertex), _geometry(mov._geometry), _layout(mov._layout)
    {
      mov._pixel    = nullptr;
      mov._vertex   = nullptr;
      mov._geometry = nullptr;
      mov._layout   = nullptr;
    }
    Shader(const char* pixel, const char* vertex, const char* geometry, std::initializer_list<Data> data);
    template<typename... Args> Shader(const char* pixel, const char* vertex, const char* geometry, Args&&... data)
    {
      init(pixel, vertex, geometry);
      (add(data), ...);
    }

    ~Shader();
    // Creates the shader in the current context
    unsigned int Create(Backend* backend) const;
    // Destroys the shader from the current context
    void Destroy(Backend* backend, unsigned int shader) const;
    // Sets vertex data for an EXISTING vertex data buffer that has ALREADY been bound via glBufferData
    void SetVertices(Backend* backend, unsigned int shader, size_t stride) const;
    static int GetTypeCount(int type);
    static void SetUniform(Backend* backend, unsigned int shader, const Attribute& attr);

    Shader& operator=(const Shader& copy);
    Shader& operator=(Shader&& mov);

  private:
    void init(const char* pixel, const char* vertex, const char* geometry);
    bool add(const Data& data);
    void destruct();

    char* _pixel;
    char* _vertex;
    char* _geometry;
    kh_attributes_s* _layout;
  };
}

#endif

// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__EXAMPLE_SKIN_H
#define FG__EXAMPLE_SKIN_H

#include "field.h"
#include "feather/skin.h"
#include <string>
#include <vector>
#include <unordered_map>

struct Skin
{
  std::string id;
  Field* data;
  std::unordered_map<std::string, std::vector<fgCalcNode>> fields;
  std::vector<Skin> children;
  fgSkinNode node;

  void Eval()
  {
    node = {
      id.c_str(),
      kh_init_unresolved(),
      kh_init_field(),
      data,
    };

    int r;
    for(auto& [k, v] : fields)
    {
      khiter_t iter = kh_put_unresolved(node.unresolved, k.c_str(), &r);
      kh_val(node.unresolved, iter) = { v.data(), v.size() };
    }
    for(auto& v : children)
    {
      v.Eval();
      v.node.sibling = node.children;
      node.children = &v.node;
    }
  }
};

#endif
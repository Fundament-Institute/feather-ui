local F = require 'feather.shared'
local rtree = require 'feather.rtree'
local Transform = require 'feather.transform'
local CT = require 'std.constraint'
local C = terralib.includecstring [[
#include <stdio.h>
]]

struct Layout {
  abs : F.Vec3
  rel : F.Vec3
  dim : F.Vec3
  -- reldim : F.Vec3
  rot : F.Vec3 -- replace with bivectors
  -- anchor : F.Vec3 -- defaults to rel
}

Layout.methods.apply = CT(function(t, u) return CT.TemplateMethod(Layout, {CT.Pointer(t), CT.Pointer(u)}, Transform,
  function (S, N, P)
    return terra(self : S, node : N, parent : P) : Transform
      node.pos.x = parent.extent.x * ((self.rel.x - 0.5f) * 2.0f) + self.abs.x
      node.pos.y = parent.extent.y * ((self.rel.y - 0.5f) * 2.0f) + self.abs.y
      node.pos.z = parent.extent.z * ((self.rel.z - 0.5f) * 2.0f) + self.abs.z
      node.extent = self.dim / 2.0f -- + self.reldim * parent.extent (need component-wise multiply)
      -- node.pos = (node.extent * ((self.anchor - 0.5f) * 2.0f))
      --node.pos.x = node.pos.x - node.extent.x * (self.rel.x * 2.0f)
      --node.pos.y = node.pos.y - node.extent.y * (self.rel.y * 2.0f)
      --node.pos.z = node.pos.z - node.extent.z * (self.rel.z * 2.0f)
      node.rot = self.rot
      return Transform.translate(node.pos)
    end
  end) end, CT.TerraType, CT.TerraType)


return Layout
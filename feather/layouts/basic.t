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
  reldim : F.Vec3
  rot : F.Bivector3
  anchor : F.Vec3 -- defaults to rel
}

Layout.methods.zero = constant(`Layout{F.zero, F.zero, F.zero, F.zero, F.zero, F.zero})

Layout.methods.create = macro(function(exp)
  if exp:gettype().convertible == "named" then
    return quote
      var l : Layout = Layout.zero

      escape
        local hasanchor = false
        for i,v in ipairs(exp:gettype().entries) do
          emit(quote l.[v.field] = exp.[v.field] end)
          if v.field == "anchor" then hasanchor = true end
        end

        if not hasanchor then
          l.anchor = l.rel
        end
      end
    in
      l
    end
  else
    error("Must pass in a named tuple")
  end
end)

Layout.methods.apply = CT(function(t, u) return CT.TemplateMethod(Layout, {CT.Pointer(t), CT.Pointer(Transform), CT.Pointer(u)}, {},
  function (S, N, T, P)
    return terra(self : S, node : N, t : T, parent : P) : {}
      node.transform.pos = parent.extent:component_mul((self.rel:component_sub(0.5f)):component_mul(2.0f)) + self.abs
      node.extent = self.dim:component_div(2.0f) + parent.extent:component_mul(self.reldim)
      --node.transform.pos = node.transform.pos:component_sub(node.extent:component_mul((self.anchor:component_sub(0.5f)):component_mul(2.0f)))
      node.transform.rot = self.rot
    end
  end) end, CT.TerraType, CT.TerraType)


return Layout
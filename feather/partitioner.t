local F = require 'feather.shared'
local CT = require 'std.constraint'

-- This defines an abstract interface for the spatial partitioning data structure used by feather.
return CT.MetaConstraint(function(a) return CT.All(
  CT.Method("create", {CT.Pointer(a), &F.Vec3, &F.Vec3, &F.Vec3, &F.Veci}, CT.Pointer(a)),
  CT.Method("destroy", {CT.Pointer(a)}, nil),
  CT.Method("contain", {CT.Pointer(a)}, nil),
  CT.Method("set", {CT.Pointer(a), &F.Vec3, &F.Vec3, &F.Vec3, &F.Veci}, nil),
  CT.Method("setparent", {CT.Pointer(a), CT.Pointer(a)}, nil),
  CT.MetaConstraint(function(t) return CT.Method("query", {F.Vec3, F.Vec3, CT.Value(t), CT.Function({CT.Pointer(a), &F.Vec3, &F.Vec3, CT.Value(t)}, bool)}, bool) end, CT.TerraType)
) end, CT.TerraType)
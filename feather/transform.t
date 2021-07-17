local F = require 'feather.shared'

  --store the data of a transformation, used for accumulating transformations while traversing the rtree.
  --Simple translation for now
struct transform {
  pos: F.Vec3
  rot: F.Bivector3
  scale: F.Vec3
}
terra transform:compose(other: &transform)
  return transform{self.pos + other.pos, F.zero, F.zero}
end
terra transform:invert()
  return transform{-self.pos, -self.rot, F.vec3(1f / self.scale.x, 1f / self.scale.y, 1f / self.scale.z)}
end
terra transform.methods.identity()
  return transform{F.zero, F.zero, F.zero}
end
terra transform.methods.translate(r: F.Vec3)
  return transform{r, F.zero, F.zero}
end

return transform
local F = require 'feather.shared'

  --store the data of a transformation, used for accumulating transformations while traversing the rtree.
  --Simple translation for now
struct transform {
  r: F.Vec3
                   }
terra transform:compose(other: &transform)
  return transform{self.r + other.r}
end
terra transform:invert()
  return transform{-self.r}
end
terra transform.methods.identity()
  return transform{F.vec3(0.0f, 0.0f, 0.0f)}
end
terra transform.methods.translate(r: F.Vec3)
  return transform{r}
end

return transform
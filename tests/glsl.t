

local glsl = require 'feather.glsl'

local terra frag_main(uv: glsl.Vec2, color: glsl.Vec4, texture: glsl.Sampler2D)
  var c: glsl.Vec4 = glsl.texture2D(texture, uv)
  return glsl.vec4(c.rgb * color.rgb * glsl.vec3(color.a, color.a, color.a), c.a * color.a)
      end

print(glsl.compile({main = frag_main}))

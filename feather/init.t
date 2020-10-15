local core = require 'feather.core'
local util = require 'feather.util'
local shared = require 'feather.shared'

local template_names = {"window", "rect"}

local templates = {}
for i, name in ipairs(template_names) do
  templates[name] = require('feather.templates.'..name)
end

return util.override(util.override(core, templates), {Vec3D = shared.Vec3D, Color = shared.Color})

local core = require 'feather.core'
local util = require 'feather.util'
local shared = require 'feather.shared'
local backend = require 'feather.backend'
local message = require 'feather.message'

local template_names = {"window", "rect"}

local templates = {}
for i, name in ipairs(template_names) do
  templates[name] = require('feather.templates.'..name)
end

local function merge(tbl)
  local res = {}
  for i, v in ipairs(tbl) do
    res = util.override(res, v)
  end
  return res
end

return merge{
  core,
  templates,
  {Vec3D = shared.Vec3D, Color = shared.Color, Backend = backend.Backend, Behavior = message.Behavior}
}

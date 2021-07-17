local core = require 'feather.core'
local util = require 'feather.util'
local shared = require 'feather.shared'
local backend = require 'feather.backend'
local message = require 'feather.message'

local template_names = {"window", "rect", "triangle", "circle", "mousearea", "arc", "line"}
local operator_names = {"let", "each", "letcontext", "getcontext", "If"} -- cond, match, range, each_index, each_pair

local templates = {}
for i, name in ipairs(template_names) do
  templates[name] = require('feather.templates.'..name)
end

local operators = {}
for i, name in ipairs(operator_names) do
  operators[name] = require('feather.operators.'..name)
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
  operators,
  {veci = shared.veci, vec3 = shared.vec3, vec = shared.vec, Veci = shared.Veci, Vec3 = shared.Vec3, Vec = shared.Vec, zero = shared.zero, Color = shared.Color, Backend = backend.Backend, Behavior = message.Behavior}
}

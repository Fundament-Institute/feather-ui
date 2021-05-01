local Util = require 'feather.util'

local M = {}

M.expression_spec_mt = {}
--make a user-provided constant value into an interface-conforming expression
function M.constant(v)
    return setmetatable({
      generate = function(self, type_context, type_environment)
        return {
          enter = function(self, context, environment) end,
          update = function(self, context, environment) end,
          exit = function(self, context) end,
          get = function(self, context) return v end, -- v should already be a quote
          storage_type = terralib.types.unit
        }
      end
    }, M.expression_spec_mt)
end

--ensure that a user-provided expression-thing is a proper expression object
function M.parse(expr) 
  local t = type(expr)
  if t == "table" then
    if getmetatable(expr) == M.expression_spec_mt then
      return expr
    else
      terralib.printraw(expr)
      error("Expression in unknown table format")
    end
  elseif t == "number" or t == "string" or t == "boolean" then
    return M.constant(`[expr])
  else
      error("Expression is an unsupported type: " .. t)
  end

  return expr
end

--compute the type that a concrete expr.get() will return in a given type_environment
function M.type(concrete_expr, type_context, type_environment)
  -- Turn this into a quote because symbols don't have gettype() for some godawful reason
  local v = `[concrete_expr.get(
    symbol(concrete_expr.storage_type),
    Util.fake_context(type_context),
    Util.fake_environment(type_environment)
  )]
  return v:gettype()
end

return M

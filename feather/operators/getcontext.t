
local Expression = require 'feather.expression'
local core = require 'feather.core'
local override = require 'feather.util'.override
local map_pairs = require 'feather.util'.map_pairs

-- Gets a value from a context
return setmetatable({},{__index=function(self, name)
  return setmetatable({
      generate = function(self, type_context, type_environment)
        return {
          enter = function(self, context, environment) end,
          update = function(self, context, environment) end,
          layout = function(self, context, environment) end,
          exit = function(self, context) end,
          get = function(self, context) return context[name] end,
          storage_type = terralib.types.unit
        }
      end
    }, Expression.expression_spec_mt)
end})

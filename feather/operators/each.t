
local Expression = require 'feather.expression'
local DynArray = require 'feather.dynarray'
local core = require 'feather.core'
local override = require 'feather.util'.override
local map_pairs = require 'feather.util'.map_pairs

-- Each loop, which instantiates a body for each element in a provided collection
return function(name)
  return function(collection)
    return function(raw_body)
      local collection_expr = Expression.parse(collection)
      local body = core.make_body(raw_body)
      local function generate(self, type_context, type_environment)
        local concrete_collection_expr = collection_expr:generate(type_context, type_environment)
        local collection_type = Expression.type(concrete_collection_expr, type_context, type_environment)
        if collection_type.elem == nil then
          error("Collection type " .. tostring(collection_type) .. " has no .elem entry! All valid collections must include the element type in [collection].elem")
        end
        local body_fn, body_type = body(type_context, override(type_environment, {[name] = collection_type.elem}))
        local struct each_result_elem {
          collection: collection_type
          bodies: DynArray(body_type)
        }
        
        -- Loop through and call update on each element in both arrays
        -- If collection ends first, call exit on remaining bodies elements
        -- If bodies ends first, initialize new elements in bodies
        local function update_bodies(self, context, environment)
          return quote
            var index : int = 0
            for v in self.collection do 
              if index < self.bodies.size then
                [body_fn.update(`self.bodies(index), context, override(environment, {[name] = `v}))]
              else
                self.bodies:resize(index + 1)
                [body_fn.enter(`self.bodies(index), context, override(environment, {[name] = `v}))]
              end
              index = index + 1
            end 

            for i=index, self.bodies.size do
              [body_fn.exit(`self.bodies(i), context)]
            end

            self.bodies:resize(index)
          end
        end

        return {
          enter = function(self, context, environment)
            return quote
              [concrete_collection_expr.enter(`self.collection, context, environment)]
              self.bodies:init()
              [update_bodies(self, context, environment)]
            end
          end,
          update = function(self, context, environment)
            return quote
              [concrete_collection_expr.update(`self.collection, context, environment)]
              [update_bodies(self, context, environment)]
            end
          end,
          exit = function(self, context)
            return quote
              [concrete_collection_expr.exit(`self.collection, context, environment)]
              for v in self.bodies do
                [body_fn.exit(`v, context)]
              end
            end
          end,
          render = function(self, context, environment)
            return quote
              for v in self.bodies do
                [body_fn.render(`v, context, environment)]
              end
            end
          end
        }, each_result_elem
      end

      return setmetatable({generate = generate}, core.outline_mt)
    end
  end
end

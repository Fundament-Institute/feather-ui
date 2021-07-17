
local Expression = require 'feather.expression'
local core = require 'feather.core'
local override = require 'feather.util'.override
local map_pairs = require 'feather.util'.map_pairs

-- Let outline, which binds some variables into scope for the body.
return function(bindings)
  local exprs = {}
  for k, v in pairs(bindings) do
    exprs[k] = Expression.parse(v)
  end
  return function(rawbody)
    local body = core.make_body(rawbody)
    local function generate(self, type_context, type_environment)
      local concrete_exprs = {}
      local texprs = {}
      for k, v in pairs(exprs) do
        concrete_exprs[k] = v:generate(type_context, type_environment)
        texprs[k] = Expression.type(concrete_exprs[k], type_context, type_environment)
      end
      local variables = {}
      for k, v in pairs(concrete_exprs) do
        table.insert(variables, {field = k, type = v.storage_type})
      end
      local store = terralib.types.newstruct("letcontext_store")
      store.entries = variables

      local function body_ctx(context, s)
        return override(context, map_pairs(concrete_exprs, function(k, v) return k, v.get(`s.[k]) end))
      end

      local body_fn, body_type = body(override(type_context, texprs), type_environment)

      return {
        enter = function(self, context, environment)
          return quote
            escape
              for k, v in pairs(concrete_exprs) do
                emit quote [v.enter(`self._0.[k], context, environment)] end
              end
            end
            [body_fn.enter(`self._1, body_ctx(context, `self._0), environment)]
          end
        end,
        update = function(self, context, environment)
          return quote
            escape
              for k, v in pairs(concrete_exprs) do
                emit quote [v.update(`self._0.[k], context, environment)] end
              end
            end
            [body_fn.update(`self._1, body_ctx(context, `self._0), environment)]
          end
        end,
        layout = function(self, context, environment)
          return quote
            [body_fn.layout(`self._1, body_ctx(context, `self._0), environment)]
          end
        end,
        exit = function(self, context)
          return quote
            [body_fn.exit(`self._1, body_ctx(context, `self._0))]
            escape
              for k, v in pairs(concrete_exprs) do
                emit quote [v.exit(`self._0.[k], context, environment)] end
              end
            end
          end
        end,
        render = function(self, context, environment)
          return body_fn.render(`self._1, body_ctx(context, `self._0), environment)
        end
      }, tuple(store, body_type)
    end

    return setmetatable({generate = generate}, core.outline_mt)
  end
end

local Closure = require 'feather.closure'
local Util = require 'feather.util'
local Expression = require 'feather.expression'

local C = terralib.includec "stdio.h"

local function merge_envs(env_inner, env_outer)
  local env_merged = setmetatable({}, {
      __index = function(_, k)
        if env_inner[k] ~= nil then
          return env_inner[k]
        else
          return env_outer[k]
        end
      end
  })
  return env_merged
end

local function logged_fake_env(base_env)
  local environment_log = {}
  local logged_fake_environment = setmetatable({}, {
      __index = function(_, k)
        local v = base_env[k]
        environment_log[k] = v
        return v
      end
  })
  return logged_fake_environment, environment_log
end

local function extract_store(store_exp)
  local env = {}
  local store_t = (`[store_exp]):gettype()
  if store_t:ispointer() then store_t = store_t.type end
  if not store_t:isstruct() then error "a store type must be a struct or a pointer to a struct" end
  for _, v in ipairs(store_t.entries) do
    env[v.field] = `store_exp.[v.field]
  end
  return env
end

local bind_syntax = {
  name = "bind_syntax";
  entrypoints = {"bind", "bindevent"};
  keywords = {};
  
  expression = function(self, lex)
    
    if lex:nextif("bind") then
      local expr = lex:terraexpr()

      return function(local_environment)
        return setmetatable({
          generate = function(self, type_context, type_environment)
            local local_env = local_environment()
            local fake_env = Util.fake_environment(type_environment)
            local expr_type = expr(merge_envs(fake_env, local_env)):gettype()
            local expr_methods = {
              enter = function(self, context, passed_env)
                return quote
                  [self] = [expr(merge_envs(passed_env,
                                             local_env))]
                end
              end,
              
              update = function(self, context, passed_env)
                return quote
                  [self] = [expr(merge_envs(passed_env,
                                             local_env))]
                end
              end,
              
              exit = function(self, context)
                return quote
                  
                end
              end,
              
              get = function(self, context, passed_env)
                return `[self]
              end,

              storage_type = expr_type
            }
            
            return expr_methods
          end
        }, Expression.expression_spec_mt)
      end
      
    elseif lex:nextif "bindevent" then
      
      lex:expect "("
      local param_names = terralib.newlist()
      local param_types = terralib.newlist()
      while not lex:nextif ")" do
        param_names:insert(lex:expect(lex.name).value)
        lex:expect ":"
        param_types:insert(lex:luaexpr())
        if not lex:matches ")" then
          lex:expect ","
        end
      end
      
      local body = lex:terrastats()
      lex:expect "end"

      return function(local_environment)
        return setmetatable({
          generate = function(self, type_context, type_env)
            local fake_env = Util.fake_environment(type_env)
            local logged_fake_env, environment_log = logged_fake_env(fake_env)
            local local_env = local_environment()

            local params_env = {}
            for i, name in ipairs(param_names) do
              param_types[i] = param_types[i](local_env)
              params_env[name] = symbol(param_types[i], name)
            end

            -- generate the body against a fake instrumented environment to capture the environmental dependencies which must be stored, then discard the resulting tree
            body(merge_envs(params_env, merge_envs(logged_fake_env, local_env)))

            local store = terralib.types.newstruct("closure_store")
            local logged_fields = {}

            for k,v in pairs(environment_log) do
              table.insert(logged_fields, {field = k, type = (`[v]):gettype()})
            end

            store.entries = logged_fields

            local handler_params = {}
            for i, name in ipairs(param_names) do handler_params[i] = params_env[name] end
            local terra event_handler(store: &store, [handler_params])
              [body(merge_envs(params_env, merge_envs(extract_store(store), local_env)))]
                  end

            local closure = Closure.closure(param_types -> event_handler.type.returntype)
            
            local closure_methods = {
              enter = function(self, context, passed_env)
                return quote
                  self.store = [context.allocator]:alloc([store])
                  escape
                    for k, _ in pairs(environment_log) do
                      emit(quote [&store](self.store).[k] = [passed_env[k]] end)
                    end
                  end
                  self.fn = [(`self):gettype().fn_type](event_handler)
                end
              end,
              
              update = function(self, context, passed_environment)
                return quote
                  escape
                    for k, _ in pairs(environment_log) do
                      emit(quote [&store](self.store).[k] = [passed_environment[k]] end)
                    end
                  end
                end
              end,
              
              exit = function(self, context)
                return quote
                  [context.allocator]:delete(self.store)
                end
              end,
              
              get = function(self, context, passed_environment)
                return self
              end,
              storage_type = closure
            }
            return closure_methods
          end
                            }, Expression.expression_spec_mt)
      end
    else
      lexer:errorexpected "bind or bindevent"
    end
  end;
}

return bind_syntax

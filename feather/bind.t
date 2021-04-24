local Closure = require 'feather.closure'
local Util = require 'feather.util'
local Expression = require 'feather.expression'

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

local function logged_fake_env(type_environment)
  local environment_log = {}
  local logged_fake_environment = setmetatable({}, {
      __index = function(_, k)
        local v = type_environment[k]
        environment_log[k] = v
        return v
      end
  })
  return logged_fake_environment, environment_log
end

local function extract_store(store_exp)
  local env = {}
  local store_t = store_exp:gettype()
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
          generate = function(context, type_environment)
          print("TEST 2")
            local fake_env = Util.fake_environment(type_environment)
          print("TEST 3")
            local expr_type = expr(fake_env):gettype()
          print("TEST 4")
            print(expr_type)
            local expr_methods = {
              enter = function(self, context, passed_environment)
                return quote
                  @[self] = [expr(merge_envs(local_environment,
                                            passed_environment))]
                end
              end,
              
              update = function(self, context, passed_environment)
                return quote
                  @[self] = [expr(merge_envs(local_environment,
                                            passed_environment))]
                end
              end,
              
              exit = function(self, context)
                return quote
                  
                end
              end,
              
              get = function(self, context, passed_environment)
                return `@[self]
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
        param_types:insert(lex:expr())
        if not lex:matches ")" then
          lex:expect ","
        end
      end
      
      local body = lex:terrastats()
      lex:expect "end"

      return function(local_environment)
        local function generate(context, type_environment)
          print("TEST 1")
          local logged_fake_environment, environment_log = logged_fake_env(type_environment)

          local params_env = {}
          for i, name in ipairs(param_names) do
            fake_params_env[name] = symbol(param_types[i], name)
          end

            -- generate the body against a fake instrumented environment to capture the environmental dependencies which must be stored, then discard the resulting tree
          body(merge_envs(params_env, merge_envs(logged_fake_environment, local_environment)))

          local store = terralib.types.newstruct("closure_store")
          local logged_fields = {}

          for k,v in pairs(environment_log) do
            table.insert(logged_fields, {field = k, type = v})
          end

          store.entries = logged_fields

          local closure = Closure.closure(param_types -> {})

          local handler_params = {}
          for i, name in ipairs(param_names) do handler_params[i] = params_env[name] end
          local terra event_handler(store: store, [handler_params])
            [body(merge_envs(params_env, merge_envs(extract_store(store), local_environment)))]
          end
          
          local closure_methods = {
            enter = function(self, context, passed_environment)
              return quote
                self.store = [context.allocator]:alloc([store])
                escape
                  for k, _ in pairs(environment_log) do
                    emit(quote self.store.[k] = [passed_environment[k]] end)
                  end
                end
                self.fn = event_handler
              end
            end,
            
            update = function(self, context, passed_environment)
              return quote
                escape
                  for k, _ in pairs(environment_log) do
                    emit(quote self.store.[k] = [passed_environment[k]] end)
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
            end
          }
          return closure_methods, closure
        end
        return generate
      end
    else
      lexer:errorexpected "bind or bindevent"
    end
  end;
}

return bind_syntax

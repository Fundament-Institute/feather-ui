local Closure = require 'feather.closure'

local function merge_envs(env_local, env_passed)
  local env_merged = setmetatable({}, {
      __index = function(_, k)
        if env_local[k] ~= nil then
          return env_local[k]
        else
          return env_passed[k]
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

local bind_syntax = {
  name = "bind_syntax";
  entrypoints = {"bind", "bindevent"};
  keywords = {};
  
  expression = function(self, lex)
    
    if lex:nextif("bind") then
      local expr = lex:terraexpr()

      return function(local_environment)
        local function generate(context, type_environment)
          local fake_environment = {}
          for sym, typ in pairs(type_environment) do
            fake_environment[sym] = symbol(typ)
          end
          local expr_type = expr(fake_environment):gettype()
          
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
            
            get = function(self, context)
              return `@[self]
            end
          }
          return expr_methods, expr_type
        end
        return generate
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
          local logged_fake_environment, environment_log = logged_fake_env(type_environment)

          local function body_fn(merged_environment)
            local function closed_fn(call_store, call_params)
              -- TODO: Inject magic
            end
          end

          -- FIXME: This probably isn't right, but it's a sketch
          body_fn(merge_envs(param_names, merge_envs(logged_fake_environment, local_environment)))

          local store = terralib.types.newstruct("closure_store")
          local logged_fields = {}

          for k,v in pairs(environment_log) do
            table.insert(logged_fields, {field = k, type = v})
          end

          store.entries = logged_fields

          local closure = Closure.closure(param_types -> {})
          
          local closure_methods = {
            enter = function(self, context, passed_environment)
              return quote
                self.store = [context.allocator]:alloc([store])
                escape
                  for k, _ in pairs(environment_log) do
                    emit(`self.store.[k] = [passed_environment[k]])
                  end
                end
                -- FIXME: This probably isn't right, but it's a sketch
                self.fn = [body_fn(merge_envs(call_params, merge_envs(passed_environment, local_environment)))]
              end
            end,
            
            update = function(self, context, passed_environment)
              return quote
                escape
                  for k, _ in pairs(environment_log) do
                    emit(`self.store.[k] = [passed_environment[k]])
                  end
                end
              end
            end,
            
            exit = function(self, context)
              return quote
                [context.allocator]:delete(self.store)
              end
            end,
            
            get = function(self, context)
              return quote
                -- FIXME: Again, probably not right, but a sketch
                [self](...)
              end
            end
          }
          return closure_methods, closure
        end
        return generate
      end
      
      -- old version
      return function(env_local_fn)
        local env_local = env_local_fn()
        return function(env_passed, args_passed)
          if #args_passed ~= #param_names then
            error "wrong number of args provided for event binding"
          end
          
          local env_outside = merge_envs(env_local, env_passed)
          local arg_env = {}
          
          for i = 1, #param_names do
            arg_env[param_names[i]] = args_passed[i]
          end
          
          local env_inside = merge_envs(arg_env, env_outside)
          return body(env_inside)
        end
      end
      
    else
      lexer:errorexpected "bind or bindevent"
    end
  end;
}

return bind_syntax

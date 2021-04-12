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

local bind_syntax = {
  name = "bind_syntax";
  entrypoints = {"bind", "bindevent"};
  keywords = {};
  
  expression = function(self, lex)
    
    if lex:nextif("bind") then
      local expr = lex:terraexpr()

      local function binding(local_environment)
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
      return binding
      
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

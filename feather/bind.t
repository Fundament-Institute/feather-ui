local bind_syntax = {
  name = "bind_syntax";
  entrypoints = {"bind"};
  keywords = {};
  expression = function(self, lex)
    lex:expect("bind")
    local expr = lex:terraexpr()
    return function(env_local_fn)
      local env_local = env_local_fn()
      return function(env_passed)
        local env_merged = setmetatable({}, {
            __index = function(_, k)
              if env_local[k] ~= nil then
                return env_local[k]
              else
                return env_passed[k]
              end
            end
        })
        return expr(env_merged)
      end
    end
  end;
}
return bind_syntax

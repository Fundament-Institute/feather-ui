local Expression = require 'feather.expression'
local core = require 'feather.core'

local C = terralib.includec "stdio.h"

local function generate(self, type_context, type_environment)
  if #self.conds > 254 then
    print "arbitrary limit for number of else branches in an if reached.\n\z
           Linear sequential search through the conditions is arbitrarily disallowed beyond 254 branches for performance reasons.\n\z
           You must do one of\n\z
           - Break the conditions up into a tree to speed the checks\n\z
           - Split the clauses up into fragments so that a single fragment call getting the fragment from the application can handle any option\n\z
           - Switch to using a match or switchcase construct instead of an if-elseif chain"
    error "too many elseif operators chained"
  end
  local preds, conseqs, conseq_types = {}, {}, {}
  for i, v in ipairs(self.conds) do
    preds[i] = v.pred:generate(type_context, type_environment)
    conseqs[i], conseq_types[i] = v.conseq(type_context, type_environment)
  end
  conseqs[#conseqs+1], conseq_types[#conseqs+1] = self.else_conseq(type_context, type_environment)
  local struct If_elem {
    position: uint8
                       }
  for i, v in ipairs(preds) do
    table.insert(If_elem.entries, { type = v.storage_type, field = "_pred_"..(i-1) })
  end
  local body_union = {}
  for i, v in ipairs(conseqs) do
    table.insert(body_union, { type = conseq_types[i], field = "_body_"..(i-1) })
  end
  table.insert(If_elem.entries, body_union)

  return {
    enter = function(self, context, environment)
      local after_conds = label "after_conds"
      local after_bodies = label "after_bodies"
      return quote
        self.position = 0
        escape
          for i, v in ipairs(preds) do
            local pred = `self.["_pred_"..(i-1)]
            emit quote
              [v.enter(pred, context, environment)]
              if [v.get(pred, context, environment)] then
                self.position = [i - 1]
                goto [after_conds]
              end
                 end
          end
        end
        self.position = [#preds] -- if no predicate matches, use the else
        :: [after_conds] ::
        escape
          for i, v in ipairs(conseqs) do
            local conseq = `self.["_body_"..(i-1)]
            emit quote
              if self.position == [i - 1] then
                [v.enter(conseq, context, environment)]
                goto [after_bodies]
              end
                 end
          end
        end
        :: [after_bodies] ::
      end
    end,
    update = function(self, context, environment)
      local after_conds_enter = label "after_conds_enter"
      local after_bodies_exit = label "after_bodies_exit"
      local after_bodies_enter = label "after_bodies_enter"
      local after_conds_exit = label "after_conds_exit"
      return quote
        var prev_pos = self.position
        escape
          for i, v in ipairs(preds) do
            local pred = `self.["_pred_"..(i-1)]
            emit quote
              if prev_pos < [i - 1] then
                [v.enter(pred, context, environment)]
              else
                [v.update(pred, context, environment)]
              end
              -- C.printf(["alternative "..i.." evaluates to %d\n"], [v.get(pred, context, environment)])
              if [v.get(pred, context, environment)] then
                -- C.printf(["accepting alternative "..(i).."\n"])
                self.position = [i - 1]
                goto [after_conds_enter]
              end
              -- C.printf(["rejecting alternative "..(i).."\n"])
                 end
          end
        end
        self.position = [#preds] -- if no predicate matches, use the else
        :: [after_conds_enter] ::
        if prev_pos ~= self.position then
          escape
            for i, v in ipairs(conseqs) do
              local conseq = `self.["_body_"..(i-1)]
              emit quote
                if prev_pos == [i - 1] then
                  [v.exit(conseq, context)]
                  goto [after_bodies_exit]
                end
                   end
            end
          end
          :: [after_bodies_exit] ::
          escape
            for i, v in ipairs(conseqs) do
              local conseq = `self.["_body_"..(i-1)]
              emit quote
                if self.position == [i - 1] then
                  [v.enter(conseq, context, environment)]
                  goto [after_bodies_enter]
                end
                   end
            end
          end
          :: [after_bodies_enter] ::
        end
        escape
          for i, v in ipairs(preds) do
            local pred = `self.["_pred_"..(i-1)]
            emit quote
              if prev_pos > [i - 1] and self.position < [i - 1] then
                [v.exit(pred, context)]
              end
                 end
          end
        end
        :: [after_conds_exit] ::
      end
    end,
    exit = function(self, context)
      local after_preds_exit = label "after_preds_exit"
      local after_bodies_exit = label "after_bodies_exit"
      return quote
        escape
          for i, v in ipairs(preds) do
            local pred = `self.["_pred_"..(i-1)]
            emit quote
              if self.position <= [i - 1] then
                [v.exit(pred, context)]
              else
                goto [after_preds_exit]
              end
                 end
          end
        end
        :: [after_preds_exit] ::
        escape
          for i, v in ipairs(conseqs) do
            local conseq = `self.["_body_"..(i-1)]
            emit quote
              if self.position == [i - 1] then
                [v.exit(conseq, context)]
                goto [after_bodies_exit]
              end
                end
          end
        end
        :: [after_bodies_exit] ::
      end
    end,
    render = function(self, context, environment)
      local after_bodies_render = label "after_bodies_render"
      return quote
        escape
          for i, v in ipairs(conseqs) do
            emit quote
              if self.position == [i - 1] then
                [v.render(`self.["_body_"..(i-1)], context, environment)]
                goto [after_bodies_render]
              end
                 end
          end
        end
        :: [after_bodies_render] ::
      end
    end
  }, If_elem
end

local function Else(self, body)
  return setmetatable( {
      conds = self.conds,
      else_conseq = core.make_body(body),
      generate = generate
  }, core.outline_mt )
end

local function ElseIf(self, pred)
  return function(body)
    local cond = {pred = Expression.parse(pred), conseq = core.make_body(body)}
    local new_conds = {}
    --TODO: using a linear copy here rather than a persistant datastructure is quadratic in number of ElseIf clauses chained. This will cause compile time problems for large chains
    --but I can't be bothered to write the appropriate data structure right now.
    for i, v in ipairs(self.conds) do
      new_conds[i] = v
    end
    new_conds[#new_conds + 1] = cond
    return setmetatable( {
        conds = new_conds,
        else_conseq = self.else_conseq,
        generate = generate,
        ElseIf = ElseIf,
        Else = Else
    }, core.outline_mt )
  end
end

local function If(pred)
  return function(body)
    return setmetatable( {
        conds = { {pred = Expression.parse(pred), conseq = core.make_body(body)} },
        else_conseq = core.make_body{},
        generate = generate,
        ElseIf = ElseIf,
        Else = Else
    }, core.outline_mt )
  end
end

return If

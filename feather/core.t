local F = require 'feather.shared'
local alloc = require 'std.alloc'
local rtree = require 'feather.rtree'
local backend = require 'feather.backend'
local override = require 'feather.util'.override
local Msg = require 'feather.message'
local Virtual = require 'feather.virtual'
local Closure = require 'feather.closure' 
local C = terralib.includecstring [[#include <stdio.h>]]

local M = {}


-- shadow is a function for handling lexical scoping in Feather templates by building tables that shadow the namespaces of enclosing scopes
local shadow_parent = {}
local shadow_mt = {
  __index = function(self, key)
    return self[shadow_parent][key]
  end
}
local function shadow(base)
  return setmetatable({[shadow_parent] = base}, shadow_mt)
end


local function map_pairs(tab, fn)
  local res = {}
  for k, v in pairs(tab) do
    local k_, v_ = fn(k, v)
    res[k_] = v_
  end
  return res
end


  --NYI, maybe unneeded
local struct layout {}

local struct layout_stack {
  head: layout
  tail: &layout_stack
                         }

local struct outline_context {
  layouts: layout_stack
                             }


local function make_outline(info)


end

  --store the data of a transformation, used for accumulating transformations while traversing the rtree.
  --Simple translation for now
struct M.transform {
  r: F.Vec3D
                   }
terra M.transform:compose(other: &M.transform)
  return M.transform{self.r + other.r}
end
terra M.transform:invert()
  return M.transform{-self.r}
end
terra M.transform.methods.identity()
  return M.transform{F.vec3D(0.0f, 0.0f, 0.0f)}
end


--metatables for templates and outlines
local template_mt = {}
local outline_mt = {__index = {}}


--unique values for use as key identifiers
M.context = {}
M.required = {}
local typed_event_spec_mt = {}
local event_spec_mt = {
    __call = function(self, ...)
      return setmetatable({args = {...}, required = self.required}, typed_event_spec_mt)
    end
}
M.event = setmetatable({required = false}, event_spec_mt)
M.requiredevent = setmetatable({required = true}, event_spec_mt)


--Expression instantiation functions
--TODO: rewrite with implementations that handle the real expressions

--make a user-provided constant value into an interface-conforming expression
local function constant_expression(v) return function() return v end end

--ensure that a user-provided expression-thing is a proper expression object
local function expression_parse(expr) return expr end

--compute the type of an expression in a particular type_environment and context.
--TODO: this interface needs t obe changed to support expressions wiht intermediate caching or auxiliary storage
local function type_expression(expr, context, type_environment)
  return expr(map_pairs(type_environment, function(k, v) if terralib.types.istype(v) then return k, symbol(v) else return k, nil end end)):gettype()
end

-- handle an expression being evaluated for the first time
-- TODO: this interface needs to be changed to support expressions with intermediate caching or auxiliary storage
local function expression_enter(expr, context, environment)
  return expr(environment)
end

-- handle an expression being reevaluated after some data changed.
-- TODO: this interface needs to be changed to support expressions with intermediate caching or auxiliary storage
local function expression_update(expr, context, environment)
  return expr(environment)
end


-- body is used both as a distinguished unique value and as an outline which looks up the provided body in the enclosing environment under it's own key
M.body = setmetatable({
    generate = function(type_environment)
      return type_environment[M.body](type_environment)
    end
                      }, outline_mt)

-- build an outline object as the body of a template from a user provided table.
local function make_body(rawbody)
  local function generate(context, type_environment)
    local elements = {}
    local types = {}
    for i, b in ipairs(rawbody) do
      elements[i], types[i] = b:generate(context, type_environment)
    end

    return {
      enter = function(self, context, environment)
        return quote
          escape
            for i, e in ipairs(elements) do
              local x = `self.["_"..(i-1)]
              emit(e.enter(x, context, environment))
            end
          end
               end
      end,
      update = function(self, context, environment)
        return quote
          escape
            for i, e in ipairs(elements) do
              emit(e.update(`self.["_"..(i-1)], context, environment))
            end
          end
        end
      end,
      exit = function(self, context)
        return quote
          escape
            for i, e in ipairs(elements) do
              emit(e.exit(`self.["_"..(i-1)], context))
            end
          end
        end
      end,
      render = function(self, context)
        return quote
          escape
            for i, e in ipairs(elements) do
              emit(e.render(`self.["_"..(i-1)], context))
            end
          end
        end
      end,
    }, tuple(unpack(types))
  end
  return generate
end

-- Calling a template produces an outline node.
function template_mt:__call(desc)
  local outline = {args = {}, template = self, has_unbound_body = false}
  for i = 1, #self.params.names do
    local name = self.params.names[i]
    if desc[name] == nil and self.params.required[name] then
      error("parameter " .. name .. " is required but not specified in outline definition")
    end
    outline.args[name] = desc[name] ~= nil and expression_parse(desc[name]) or self.params.defaults[name]
  end
  if self.params.has_body then
    local rawbody = {n = #desc}
    for i = 1, #desc do
      rawbody[i] = desc[i]
    end
    outline.body = make_body(rawbody)
  elseif desc[1] then
    error "attempted to pass a body to a template which doesn't accept one"
  end
  function outline:generate(context, type_environment)
    local binds = {}
    if self.body then
      local body_fns, body_type = self.body(context, type_environment)
      binds[M.body] = function()
        return {
          enter = function(self, context, environment)
            return body_fns.enter(self, context, environment[M.body])
          end,
          update = function(self, context, environment)
            return body_fns.update(self, context, environment[M.body])
          end,
          exit = function(self, context)
            return body_fns.exit(self, context)
          end,
          render = function(self, context)
            return body_fns.render(self, context)
          end
        }, body_type
      end
    else
      binds[M.body] = function()
        return {
          enter = function() return {} end,
          update = function() return {} end,
          exit = function() return {} end,
          render = function() return {} end
        }, terralib.types.unit
      end
    end
    for k, v in pairs(self.args) do
      if self.template.params.events[k] then
        binds[k] = Closure.closure(self.template.params.events[k].args -> {})
      else
        binds[k] = type_expression(v, context, type_environment)
      end
    end
    local fns, type = self.template:generate(context, binds)
    return {
      enter = function(data, context, environment)
        local binds = {
          [M.body] = environment,
        }
        for k, v in pairs(self.args) do
          binds[k] = expression_enter(v, context, environment)
        end
        return fns.enter(data, context, binds)
      end,
      update = function(data, context, environment)
        local binds = {
          [M.body] = environment,
        }
        for k, v in pairs(self.args) do
          binds[k] = expression_update(v, context, environment)
        end
        return fns.update(data, context, binds)
      end,
      exit = function(data, context)
        return fns.exit(data, context)
      end,
      render = function(data, context)
        return fns.render(data, context)
      end
    }, type
  end
  return setmetatable(outline, outline_mt)
end

-- parse an argument declaration table into a usable specification object
local function parse_params(desc)
  local res = {required = {}, defaults = {}, names = {}, events = {}, has_body = false}
  for k, v in pairs(desc) do
    if type(k) ~= "string" then
      if k == 1 and v == M.body then
        res.has_body = true
      else
        error("invalid argument specifier provided to declaration", 2)
      end
    else
      table.insert(res.names, k)
      if v == M.required then
        res.required[k] = true
      else
        local val_mt = getmetatable(v)
        if val_mt == event_spec_mt or val_mt == typed_event_spec_mt then
          if v.required then
            res.required[k] = true
          else
            res.defaults[k] = constant_expression(`F.EmptyCallable {})
          end
          res.events[k] = v
        else
          res.defaults[k] = constant_expression(v)
        end
      end
    end
  end
  return res
end

-- A raw template allows taking full control of the outline generation, including custom memory layout, caching behavior, and the full details of positioning and rendering.
-- this should be used primarily for internal core templates and should rarely if ever appear in user code.
-- if this gets used in user code, it probably means that either the code is doing something wrong or Feather is missing features.
function M.raw_template(params)
  local params = parse_params(params)
  return function(generate)
    local self = {
      params = params,
      generate = generate
    }
    return setmetatable(self, template_mt)
  end
end

-- A basic template automatically scaffolds sensible default generation behavior including handling positioning, layout, and caching, but leaves rendering up to custom code.
-- This can be used to implement custom rendering behavior that needs to interact directly with the backend which can't be represented in existing standard templates.
-- User code should use this sparingly, prefering to build templates from provided standard templates instead of custom rendering code.
function M.basic_template(params)
  local params_abbrev = parse_params(params)
  local params_full = parse_params(params)

  table.insert(params_full.names, "pos")
  table.insert(params_full.names, "ext")
  table.insert(params_full.names, "rot")
  local zerovec = constant_expression(`[F.Vec3D]{array(0.0f, 0.0f, 0.0f)})
  params_full.defaults.pos = zerovec
  params_full.defaults.ext = zerovec
  params_full.defaults.rot = zerovec

  return function(render)
    local function generate(self, context, type_environment)
      local typ = terralib.types.newstruct("basic_template")
      for i, name in ipairs(params_abbrev.names) do
        typ.entries[i] = {field = name, type = type_environment[name]}
      end
      Virtual.extends(Msg.Receiver)(typ)
      typ:complete()
      
      local function unpack_store(store)
        local res = {}
        for i, name in ipairs(params_abbrev.names) do
          res[name] = `store.[name]
        end
        return res
      end

      local body_fns, body_type
      if params_abbrev.body then
        body_fns, body_type = type_environment[M.body](context, type_environment)
      end

      return {
        enter = function(self, context, environment)
          return quote
            escape
              for i, name in ipairs(params_abbrev.names) do
                emit quote self._0.[name] = [environment[name] ] end
              end
            end

            self._0.vftable = [self:gettype().entries[1][2]].virtual_initializer
            var pos = [environment.pos]
            var ext = [environment.ext]
            var rot = [environment.rot]
            var z_index = [F.Veci]{array(0, 0)}
            self._2 = [context.rtree]:create([context.rtree_node], &pos, &ext, &rot, &z_index)
            self._2.data = [&Msg.Receiver](&self._0)
            var local_transform = M.transform{self._2.pos}
            var transform = [context.transform]:compose(&local_transform)
            escape
              if params_abbrev.body then
                emit(body_fns.enter(
                       `self._1,
                       override(context, {rtree_node = `self._2, transform = `transform}),
                       override(environment, unpack_store(`self._0))
                ))
              end
            end
          end
        end,
        update = function(self, context, environment)
          return quote
            escape
              for i, name in ipairs(params_abbrev.names) do
                emit quote self._0.[name] = [environment[name] ] end
              end
            end
            self._2.pos = [environment.pos]
            self._2.extent = [environment.ext]
            self._2.rot = [environment.rot]
            var local_transform = M.transform{self._2.pos}
            var transform = [context.transform]:compose(&local_transform)
            escape
              if params_abbrev.body then
                emit(body_fns.update(
                       `self._1,
                       override(context, {rtree_node = `self._2, transform = `transform}),
                       override(environment, unpack_store(`self._0))
                ))
              end
            end
          end
        end,
        exit = function(self, context)
          return quote
            escape
              if params_abbrev.body then
                emit(body_fns.exit(`self._1, override(context, {rtree_node = `self._2})))
              end
            end
            [context.rtree]:destroy(self._2)
          end
        end,
        render = function(self, context)
          return quote
            var local_transform = M.transform{self._2.pos}
            var transform = [context.transform]:compose(&local_transform)
            [render(override(context, {transform = `transform, rtree_node = `self._2}), unpack_store(`self._0))]
          end
        end
      }, tuple(typ, body_type or terralib.types.unit, context.rtree_node)
    end

    local self = {
      params = params_full,
      generate = generate,
    }
    return setmetatable(self, template_mt)
  end
end

-- build a template from an outline defined in terms of existing templates; this should be the primary method of defining templates in user code.
function M.template(params)
  local params = parse_params(params)
  return function(defn)
    local defn_body = make_body(defn)

    local function generate(self, context, type_environment)
      local defn_body_fns, defn_body_type = defn_body(context, type_environment)

      local element = terralib.types.newstruct("template_store")
      for i, name in ipairs(params.names) do
        element.entries[i] = {field = name, type = type_environment[name]}
      end
      Virtual.extends(Msg.Receiver)(element)

      local function unpack_store(store)
        local res = {}
        for i, name in ipairs(params.names) do
          res[name] = `store.[name]
        end
        return res
      end

      return {
        enter = function(self, context, environment)
          local initializers = {}
          for i, name in ipairs(params.names) do
            initializers[i] = quote self._0.[name] = [environment[name] ] end
          end
          return quote
            [initializers]
            [defn_body_fns.enter(`self._1, context, override(environment, unpack_store(`self._0)))]
                 end
        end,
        update = function(self, context, environment)
          local updates = {}
          for i, name in ipairs(params.names) do
            updates[i] = quote self._0.[name] = [environment[name] ] end
          end
          return quote
            [updates];
            [defn_body_fns.update(`self._1, context, override(environment, unpack_store(`self._0)))]
          end
        end,
        exit = function(self, context)
          return defn_body_fns.exit(`self._1, context)
        end,
        render = function(self, context)
          return defn_body_fns.render(`self._1, context)
        end
      }, tuple(element, defn_body_type)
    end

    local self = {
      params = params,
      generate = generate
    }
    return setmetatable(self, template_mt)
  end
end

-- Let outline, which binds some variables into scope for the body.
function M.let(bindings)
  local exprs = {}
  for k, v in pairs(bindings) do
    exprs[k] = expression_parse(v)
  end
  return function(rawbody)
    local body = make_body(rawbody)
    local function generate(self, context, type_environment)
      local texprs = {}
      for k, v in pairs(exprs) do
        texprs[k] = type_expression(v, context, type_environment)
      end
      local variables = {}
      for k, v in pairs(texprs) do
        table.insert(variables, {field = k, type = v})
      end
      local store = terralib.types.newstruct("let_store")
      store.entries = variables

      local function body_env(environment, s)
        return override(environment, map_pairs(exprs, function(k, v) return k, `s.[k] end))
      end

      local body_fn, body_type = body(context, override(type_environment, texprs))


      return {
        enter = function(self, context, environment)
          return quote
            escape
              for k, v in pairs(exprs) do
                emit quote self._0.[k] = [expression_enter(v, context, environment)] end
              end
            end
            [body_fn.enter(`self._1, context, body_env(environment, `self._0))]
                 end
        end,
        update = function(self, context, environment)
          return quote
            escape
              for k, v in pairs(exprs) do
                emit quote self._0.[k] = [expression_update(v, context, environment)] end
              end
            end
            [body_fn.update(`self._1, context, body_env(environment, `self._0))]
                 end
        end,
        exit = function(self, context)
          return quote
            [body_fn.exit(`self._1, context)]
                 end
        end,
        render = function(self, context)
          return body_fn.render(`self._1, context)
        end
      }, tuple(store, body_type)
    end

    return setmetatable({generate = generate}, outline_mt)
  end
end

-- Each loop, which instantiates a body for each element in a provided collection
function M.each(name)
  error "NYI: each loop is not yet implemented" --TODO: complete implementation
  return function(collection)
    return function(raw_body)
      local collection_expr = expression_parse(collection)
      local body = make_body(raw_body)
      local function generate(self, context, type_environment)
        local collection_type = type_expression(collection_expr, context, type_environment)
        local body_fn, body_type = body(context, override(type_environment, {[name] = collection_type.elem}))
        local struct each_elem {
          collection: collection_type
          bodies: ArrayList(body_type) --this type doesn't exist yet
                               }
        
        --TODO: IMPLEMENT ME
      end
    end
  end
end


-- The UI root, accepting a descriptor of the entire application's UI and the interface it binds against, producing a UI object which can be compiled and instantiated to actually put things on screen.
function M.ui(desc)
  local body_gen = make_body(desc)

  local type_environment = {}
  local query_names = {}

  local allocator_type = alloc.default_allocator:gettype()
  local rtree_type = rtree(allocator_type)
  local application_type = desc.application
  local backend_type = &backend.Backend

  for k, v in pairs(desc.queries) do
    table.insert(query_names, k)
  end

  table.sort(query_names)

  local query_name_lookup = {}
  local query_binding = terralib.types.newstruct("query_store")
  -- local query_store = (&opaque)[#query_names]
  local query_store_initializers = {}
  for i, v in ipairs(query_names) do
    queries_binding_type.entries:insert {field = v, type = desc.queries[v]}
    query_name_lookup[v] = i
  end

  type_environment.app = application_type

  local struct ui_base {
    allocator: allocator_type
    application: application_type
    queries: query_binding
    backend: backend_type
  }

  local struct window_base(Virtual.extends(Msg.Receiver)) {
    rtree: rtree_type
  }
  window_base:complete()

  local type_context = {
    ui = ui_base,
    window_base = window_base,
    rtree = rtree_type,
    rtree_node = &rtree_type.Node,
    allocator = allocator_type,
    backend = backend_type
  }

  local body_fns, body_type = body_gen(type_context, type_environment)

  local struct ui {
    allocator: allocator_type
    application: application_type
    queries: query_binding
    backend: backend_type
    data: body_type
                  }

  ui.query_store = query_binding

  -- terralib.printraw(ui)

  -- method to initialize the UI
  terra ui:init(application: application_type, queries: query_binding, backend: backend_type)
    self.allocator = alloc.default_allocator
    self.application = application
    --[[escape
      for i, v in ipairs(query_names) do
        emit quote self.queries[ [i] ] = queries.[v] end
      end
      end]]
    self.queries = queries
    self.backend = backend
  end

  terra ui:destruct()
    
  end

  local function make_context(self)
    return {
      backend = `self.backend
    }
  end

  local function make_environment(self)
    local res = {
      app = `self.application
    }
    for i, v in ipairs(query_names) do
      res[v] = macro(function(...) local args = {...}; return `self.queries.[v](self.application, [args]) end)
    end
    return res
  end

  do
    local self = symbol(ui)
    local foo = body_fns.enter(`self.data,
                    make_context(self),
                    make_environment(self)
    )
  end

  -- UI lifecycle methods.
  terra ui:enter()
    [body_fns.enter(`self.data,
                    make_context(self),
                    make_environment(self)
    )]
  end
  terra ui:update()
    [body_fns.update(
       `self.data,
       make_context(self),
       make_environment(self)
    )]
  end
  terra ui:exit()
    [body_fns.exit(`self.data, make_context(self))]
  end
  terra ui:render()
    [body_fns.render(`self.data, make_context(self))]
  end

  local struct BehaviorParamPack {
    w: &opaque
    ui: &opaque
    m: &Msg.Message
  }

  local terra mousequery(n : &rtree_type.Node, p : &F.Vec3D, r : &F.Vec3D, params : BehaviorParamPack) : bool 
    if n.data ~= nil then
      return Msg.DefaultBehavior([&Msg.Receiver](n.data), params.w, params.ui, params.m).mouseMove >= 0
    end
    return false
  end

  terra ui.methods.behavior(r : &Msg.Receiver, w: &opaque, ui: &opaque, m: &Msg.Message) : Msg.Result
    if r ~= nil then 
      switch [int32](m.kind.val) do
        case [Msg.Receiver.virtualinfo.info["MouseDown"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseDblClick"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseUp"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseOn"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseOff"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseMove"].index] then
          goto MouseEvent
        case [Msg.Receiver.virtualinfo.info["MouseScroll"].index] then
          ::MouseEvent::
          var base = [&window_base](r)
          var params = BehaviorParamPack{w, ui, m}
          return Msg.Result{terralib.select(base.rtree:query(F.vec3D(m.mouseMove.x,m.mouseMove.y,0.0f), F.vec3D(0.0f,0.0f,1.0f), params, mousequery), 0, -1)}
        end
      end
      return Msg.DefaultBehavior(r, w, ui, m)
    end

    -- handle global events here (like global key hooks)
    return Msg.Result{-1}
  end

  return ui
end

return M

local F = require 'feather.shared'
local alloc = require 'std.alloc'
local rtree = require 'feather.rtree'
local backend = require 'feather.backend'
local override = require 'feather.util'.override
local map_pairs = require 'feather.util'.map_pairs
local Msg = require 'feather.message'
local Virtual = require 'feather.virtual'
local Closure = require 'feather.closure' 
local C = terralib.includecstring [[#include <stdio.h>]]
local Expression = require 'feather.expression'

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

  --NYI, maybe unneeded
local struct layout {}

local struct layout_stack {
  head: layout
  tail: &layout_stack
}

local struct outline_context {
  layouts: layout_stack
}

  --store the data of a transformation, used for accumulating transformations while traversing the rtree.
  --Simple translation for now
struct M.transform {
  r: F.Vec3
                   }
terra M.transform:compose(other: &M.transform)
  return M.transform{self.r + other.r}
end
terra M.transform:invert()
  return M.transform{-self.r}
end
terra M.transform.methods.identity()
  return M.transform{F.vec3(0.0f, 0.0f, 0.0f)}
end


--metatables for templates and outlines
local template_mt = {}
M.outline_mt = {__index = {}, __call = function(self, ...) return self:generate(...) end}

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

-- body is used both as a distinguished unique value and as an outline which looks up the provided body in the enclosing environment under it's own key
M.body = setmetatable({
    generate = function(type_environment)
      return type_environment[M.body](type_environment)
    end
    }, M.outline_mt)

-- build an outline object as the body of a template from a user provided table.
function M.make_body(rawbody)
  local function generate(type_context, type_environment)
    local elements = {}
    local types = {}
    for i, b in ipairs(rawbody) do
      elements[i], types[i] = b:generate(type_context, type_environment)
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
      render = function(self, context, environment)
        return quote
          escape
            for i, e in ipairs(elements) do
              emit(e.render(`self.["_"..(i-1)], context, environment))
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
  for k, v in pairs(desc) do
    if not self.params.names_map[k] and type(k) == "string" then
      print(debug.traceback("WARNING: passed parameter to template that it doesn't declare: "..k, 2))
    end
  end
  for i = 1, #self.params.names do
    local name = self.params.names[i]
    if desc[name] == nil and self.params.required[name] then
      error("parameter " .. name .. " is required but not specified in outline definition")
    end
    outline.args[name] = desc[name] ~= nil and Expression.parse(desc[name]) or self.params.defaults[name]
  end
  if self.params.has_body then
    local rawbody = {n = #desc}
    for i = 1, #desc do
      rawbody[i] = desc[i]
    end
    outline.body = M.make_body(rawbody)
  elseif desc[1] then
    error "attempted to pass a body to a template which doesn't accept one"
  end
  function outline:generate(type_context, type_environment)
    local binds = {}
    local concrete_args = map_pairs(self.args, function(k, v) return k, v:generate(type_context, type_environment) end)
    if self.body then
      local body_fns, body_type = self.body(type_context, type_environment)
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
          render = function(self, context, environment)
            return body_fns.render(self, context, environment)
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
    for k, v in pairs(concrete_args) do
      -- if self.template.params.events[k] then
      --   binds[k] = Closure.closure(self.template.params.events[k].args -> {})
      -- else
        binds[k] = Expression.type(v, type_context, type_environment)
      -- end
    end
    local fns, type = self.template:generate(type_context, binds)
    
    local store = terralib.types.newstruct("outline_store")
    for k, v in pairs(concrete_args) do
      table.insert(store.entries, {field = k, type = v.storage_type})
    end

    return {
      enter = function(data, context, environment)
        local binds = {
          [M.body] = environment,
        }
        for k, v in pairs(concrete_args) do
          binds[k] = v.get(`data._1.[k], context, environment)
        end
        return quote
          escape
            for k, v in pairs(concrete_args) do
              emit(v.enter(`data._1.[k], context, environment))
            end
          end
          [fns.enter(`data._0, context, binds)]
        end
      end,
      update = function(data, context, environment)
        local binds = {
          [M.body] = environment,
        }
        for k, v in pairs(concrete_args) do
          binds[k] = v.get(`data._1.[k], context, environment)
        end
        return quote
          escape
            for k, v in pairs(concrete_args) do
              emit(v.update(`data._1.[k], context, environment))
            end
          end
          [fns.update(`data._0, context, binds)]
        end
      end,
      exit = function(data, context)
        for k, v in pairs(concrete_args) do
          v.exit(`data._1.[k], context, {} )--TODO: provide a proper environment
        end
        return quote
          [fns.exit(`data._0, context)]
          escape
            for k, v in pairs(concrete_args) do
              emit(v.exit(`data._1.[k], context))
            end
          end
        end
      end,
      render = function(data, context, environment)
        local binds = {
          [M.body] = environment,
        }
        for k, v in pairs(concrete_args) do
          binds[k] = v.get(`data._1.[k], context, environment)
        end
        return quote
          [fns.render(`data._0, context, binds)]
        end
      end
    }, tuple(type, store)
  end
  return setmetatable(outline, M.outline_mt)
end

-- parse an argument declaration table into a usable specification object
local function parse_params(desc)
  local res = {required = {}, defaults = {}, names = {}, names_map = {}, events = {}, has_body = false}
  for k, v in pairs(desc) do
    if type(k) ~= "string" then
      if k == 1 and v == M.body then
        res.has_body = true
      else
        error("invalid argument specifier provided to declaration", 2)
      end
    else
      table.insert(res.names, k)
      res.names_map[k] = true
      if v == M.required then
        res.required[k] = true
      else
        local val_mt = getmetatable(v)
        if val_mt == event_spec_mt or val_mt == typed_event_spec_mt then
          if v.required then
            res.required[k] = true
          else
            res.defaults[k] = Expression.constant(`F.EmptyCallable {})
          end
          res.events[k] = v
        else
          res.defaults[k] = Expression.constant(v)
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
  params_full.names_map.pos = true
  params_full.names_map.ext = true
  params_full.names_map.rot = true
  local zerovec = Expression.constant(`[F.Vec3]{array(0.0f, 0.0f, 0.0f)})
  params_full.defaults.pos = zerovec
  params_full.defaults.ext = zerovec
  params_full.defaults.rot = zerovec

  return function(render)
    local function generate(self, type_context, type_environment)
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
        body_fns, body_type = type_environment[M.body](type_context, type_environment)
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
      }, tuple(typ, body_type or terralib.types.unit, type_context.rtree_node)
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
    local defn_body = M.make_body(defn)

    local function generate(self, type_context, type_environment)
      local defn_body_fns, defn_body_type = defn_body(type_context, type_environment)

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
        render = function(self, context, environment)
          return defn_body_fns.render(`self._1, context, environment)
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

-- The UI root, accepting a descriptor of the entire application's UI and the interface it binds against, producing a UI object which can be compiled and instantiated to actually put things on screen.
function M.ui(desc)
  local body_gen = M.make_body(desc)

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
    [body_fns.render(
      `self.data,
      make_context(self),
      make_environment(self)
    )]
  end

  local struct BehaviorParamPack {
    w: &Msg.Window
    ui: &opaque
    m: &Msg.Message
  }

  local terra mousequery(n : &rtree_type.Node, p : &F.Vec3, r : &F.Vec3, params : BehaviorParamPack) : bool 
    if n.data ~= nil then
      return Msg.DefaultBehavior([&Msg.Receiver](n.data), params.w, params.ui, params.m).mouseMove >= 0
    end
    return false
  end

  terra ui.methods.behavior(r : &Msg.Receiver, w: &Msg.Window, ui: &opaque, m: &Msg.Message) : Msg.Result
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
          return Msg.Result{terralib.select(base.rtree:query(F.vec3(m.mouseMove.x,m.mouseMove.y,0.0f), F.vec3(0.0f,0.0f,1.0f), params, mousequery), 0, -1)}
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

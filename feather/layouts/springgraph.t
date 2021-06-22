local M = {}

local core = require 'feather.core'
local ga = require 'std.ga'
local dynarray = require 'feather.dynarray'
local expression = require 'feather.expression'
local override = require 'feather.util'.override
local Math = require 'std.math'
local F = require 'feather.shared'
local C = terralib.includecstring [[
#include <stdio.h>
]]

local function makeSystem(dimension)
  local g = ga(float, dimension)
  local struct particle {
    pos: g.vector_t
    vel: g.vector_t
    mass: g.scalar_t
    charge: g.scalar_t
                        }
  
  local struct spring {
    conns: tuple(uint, uint)
    strength: g.scalar_t -- [0, 1]
    length: g.scalar_t -- [0, oo)
                      }

  local struct system {
    gravity: g.scalar_t -- [0, oo)
    drag: g.scalar_t -- [0, 1]
    particles: dynarray(particle)
    springs: dynarray(spring)
                      }

  system.particle = particle
  system.spring = spring
  system.algebra = g

  terra system:update()
    -- C.printf "starting frame\n"
    -- position based dynamics
    -- initial velocity, gravity, and drag application.
    for i = 0, self.particles.size do
      var part = &self.particles(i)
      -- C.printf("pos: %f %f\n", part.pos.v[0], part.pos.v[1])
      var vel = part.vel
      part.vel = part.pos
      part.pos = part.pos + vel * (1 - self.drag) --reuse velocity to store old position; will be recomputed at end
      -- part.pos, part.vel = part.pos + part.vel * (1 - self.drag), part.pos --reuse velocity to store old position; will be recomputed at end
      -- part.vel, part.pos = part.pos, part.pos + part.vel * (1 - self.drag) --reuse velocity to store old position; will be recomputed at end
      part.pos = part.pos - self.gravity * part.pos:normalize()
    end
    -- apply spring constraints
    for spring in self.springs:iter() do
      var p0 = &self.particles(spring.conns._0)
      var p1 = &self.particles(spring.conns._1)
      var diff = p0.pos - p1.pos
      var correction = diff:normalize() * (spring.length - diff:magnitude()) * spring.strength
      var w0, w1 = 1/p0.mass, 1/p1.mass
      p0.pos = p0.pos + w0/(w0 + w1) * correction
      p1.pos = p1.pos - w1/(w0 + w1) * correction
    end
    -- electrostatic repulsion between particles
    -- TODO: implement spatial approximation tree to get better performance
    for i = 0, self.particles.size - 1 do
      for j = i + 1, self.particles.size do
        var p0, p1 = &self.particles(i), &self.particles(j)
        var diff = p0.pos - p1.pos
        var correction = diff:normalize() * (p0.charge * p1.charge) / diff:mag2()
        var w0, w1 = 1/p0.mass, 1/p1.mass
        p0.pos = p0.pos + w0/(w0 + w1) * correction
        p1.pos = p1.pos - w1/(w0 + w1) * correction
      end
    end
    -- compute final velocity
    for i = 0, self.particles.size do
      var part = &self.particles(i)
      part.vel = part.pos - part.vel --velocity field stored old position during updates; recover velocity from position change
    end
  end

  -- system.methods.update:disas()

  terra system:getPos(id: uint)
    return self.particles(id).pos
  end

  return system 
end
local makeSystem = terralib.memoize(makeSystem)


local System2D = makeSystem(2)

M.spring = core.raw_expression{
  strength = `0.0f,
  length = core.required,
  pair = core.required,
} (
  function(self, type_context, type_environment)
    return {
      enter = function(self, context, environment)
        return quote context.graph_system.springs:add([type_context.graph_system.spring]{
            [environment.pair],
            [type_context.graph_system.algebra.scalar_t]([environment.strength]),
            [type_context.graph_system.algebra.scalar_t]([environment.length])
                                                     })
          self = environment.pair
        end
      end,
      update = function(self, context, environment) return quote self = environment.pair end end,
      exit = function(self, context) return quote end end,
      get = function(self, context)
        local v =`{context.graph_system.particles(self._0).pos, context.graph_system.particles(self._1).pos}
        return v
      end, 
      storage_type = tuple(uint, uint)
    }
  end
)

M.node = core.raw_expression{
  id = core.required,
  mass = core.required,
  charge = core.required,
}(
  function(self, type_context, type_environment)
    return {
      enter = function(self, context, environment)
        local pos = `[type_context.graph_system.algebra.vector](Math.sin(environment.id*2.4f)*100f, Math.cos(environment.id*2.4f)*100f)
        local vel = `[type_context.graph_system.algebra.vector](0.0f,0.0f)
        local mass = `[type_context.graph_system.algebra.scalar]([environment.mass])
        local charge = `[type_context.graph_system.algebra.scalar](environment.charge)
        return quote context.graph_system.particles:add([type_context.graph_system.particle]{
              pos, vel, mass, charge
                                                       })
          self = environment.id
        end
      end,
      update = function(self, context, environment) return quote self = environment.id end end,
      exit = function(self, context) return {} end,
      get = function(self, context) local v = `context.graph_system.particles(self).pos; return v end,
      storage_type = uint
    }
  end
)

M.graph = core.raw_template {
  gravity = `0.1f,
  drag = `0.02f,
  pos = core.required,
  dim = core.required,
  core.body
} (
  function(self, type_context, type_environment)
    local body_fns, body_type = type_environment[core.body](override(type_context, {graph_system = System2D}), type_environment)
    local function override_context(self, context, transform)
      return override(context, {
        graph_system = `self.system,
        transform = transform,
        accumulated_parent_transform = transform,
        rtree_node = `self.rtree_node,
      })
    end

    local struct store {
      system: System2D
      body: body_type
      rtree_node: type_context.rtree_node
                       }

    local fn_table = {
      enter = function(self, context, environment)
        return quote
          self.system.gravity = [environment.gravity]
          self.system.drag = [environment.drag]
          self.system.particles:init()
          self.system.springs:init()
          var zero = F.vec3(0.0f, 0.0f, 0.0f)
          var dim = [environment.dim]
          var z_index = F.veci(0,0)
          self.rtree_node = [context.rtree]:create([context.rtree_node], &environment.pos, &dim, &zero, &z_index)
          self.rtree_node.data = nil -- we don't process messages, so we set data to nil
          var local_transform = core.transform.translate(self.rtree_node.pos)
          self.rtree_node.transform = context.transform:compose(&self.rtree_node.transform)
          [body_fns.enter(`self.body, override_context(self, context, `core.transform.identity()), environment)]
        end
      end,
      update = function(self, context, environment)
        return quote
          self.system:update()
          self.rtree_node.pos = [environment.pos]
          self.rtree_node.transform = core.transform.translate(self.rtree_node.pos)
          [body_fns.update(`self.body, override_context(self, context, `core.transform.identity()), environment)]
        end
      end,
      exit = function(self, context)
        return quote
          [body_fns.exit(`self.body, override_context(self, context))]
          self.system.particles:destruct()
          self.system.springs:destruct()
        end
      end,
      render = function(self, context)
        return quote
            var accumulated = [context.accumulated_parent_transform]:compose(&self.rtree_node.transform)
            [body_fns.render(`self.body, override_context(self, context, `accumulated))]
          end
      end
    }

    return fn_table, store
  end
)









return M

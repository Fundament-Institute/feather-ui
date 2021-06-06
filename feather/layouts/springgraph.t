local M = {}

local core = require 'feather.core'
local ga = require 'std.ga'
local dynarray = require 'feather.dynarray'
local expression = require 'feather.expression'
local override = require 'feather.util'.override
local Math = require 'std.math'

local function makeSystem(dimension)
  local g = ga(float, dimension)
  local struct particle {
    pos: g.vector_t
    vel: g.vector_t
    mass: g.scalar_t
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
    -- position based dynamics
    -- initial velocity, gravity, and drag application.
    for i = 1, self.particles.size do
      var part = &self.particles(i)
      part.pos, part.vel = part.pos + part.vel * (1 - self.drag), part.pos --reuse velocity to store old position; will be recomputed at end
      part.pos = part.pos - self.gravity * part.pos:normalize()
    end
    -- apply spring constraints
    for spring in self.springs:iter() do
      var p0 = &self.particles(spring.conns._0)
      var p1 = &self.particles(spring.conns._1)
      var diff = p0.pos - p1.pos
      var correction = diff:normalize() * (spring.length - diff:magnitude())
      var w0, w1 = 1/p0.mass, 1/p1.mass
      p0.pos = p0.pos - w0/(w0 + w1) * correction
      p1.pos = p1.pos + w1/(w0 + w1) * correction
    end
    -- compute final velocity
    for i = 1, self.particles.size do
      var part = &self.particles(i)
      part.vel = part.pos - part.vel --velocity field stored old position during updates; recover velocity from position change
    end
  end

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
            [type_context.graph_system.algebra.scalar_t]{[environment.strength]},
            [type_context.graph_system.algebra.scalar_t]{[environment.length]}
          })
        end
      end,
      update = function(self, context, environment) end,
      exit = function(self, context) end,
      get = function(self, context) return `{context.graph_system.particles[ [environment.pair._0] ].pos,context.graph_system.particles[ [environment.pair._1] ].pos} end, 
      storage_type = terralib.types.unit
    }
  end
)

M.node = core.raw_expression{
  id = core.required,
  mass = core.required,
}(
  function(self, type_context, type_environment)
    return {
      enter = function(self, context, environment)
        return quote context.graph_system.particles:add([type_context.graph_system.particle]{
            [type_context.graph_system.algebra.vector_t]{Math.sin(environment.id*2.4)*100, Math.cos(environment.id*2.4)*100},
            [type_context.graph_system.algebra.vector_t]{0.0,0.0},
            [type_context.graph_system.algebra.scalar_t]{[environment.mass]}
          })
        end
      end,
      update = function(self, context, environment) end,
      exit = function(self, context) end,
      get = function(self, context) return `context.graph_system.particles[ [environment.id] ].pos end,
      storage_type = terralib.types.unit
    }
  end
)

M.graph = core.raw_template {
  gravity = `0.1f,
  drag = `0.02f,
} (
  function(self, type_context, type_environment)
    local body_fns, body_type = type_environment[core.body](override(type_context, {graph_system = System2D}), type_environment)

    local function override_context(self, context)
      return override(context, {
        graph_system = `self,
      })
    end

    local fn_table = {
      enter = function(self, context, environment)
        return quote
          self.gravity = [environment.gravity]
          self.drag = [environment.drag]
          self.particles:init()
          self.springs:init()
          [body_fns.enter(`self.body, override_context(self, context), environment)]
        end
      end,
      update = function(self, context, environment)
        return quote
          self:update()
          [body_fns.update(`self.body, override_context(self, context), environment)]
        end
      end,
      exit = function(self, context)
        return quote
          [body_fns.exit(`self.body, override_context(self, context))]
          self.particles:destruct()
          self.springs:destruct()
        end
      end,
      render = function(self, context)
        return quote
            var transform = core.transform.identity()
            [body_fns.render(`self.body, override_context(self, context))]
          end
      end
    }

    return fn_table, System2D
  end
)









return M

local M = {}

local core = require 'feather.core'
local ga = require 'std.ga'
local dynarray = require 'feather.dynarray'
local expression = require 'feather.expression'

local function makeSystem(dimension)
  local g = ga(float, dimension)
  local struct particle {
    pos: g.vector
    vel: g.vector
    mass: g.scalar
                        }
  
  local struct spring {
    conns: tuple(uint, uint)
    strength: g.scalar -- [0, 1]
    length: g.scalar -- [0, oo)
                      }

  local struct system {
    gravity: g.scalar -- [0, oo)
    drag: g.scalar -- [0, 1]
    particles: dynarray(particle)
    springs: dynarray(spring)
                      }

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


local system = makeSystem(2)

M.spring

M.node = expression.raw_expression(
  function(self, type_context, type_environment)
)

M.graph = core.raw_template {
  gravity = `0.1f,
  drag = `0.02f,
  
} {
  
}









return M

-- This is a 3D r-tree with rotation and z-indexing

local F = require 'feather.shared'
local Alloc = require 'std.alloc'
local CT = require 'std.constraint'
local Math = require 'std.math'
local Util = require 'feather.util'
local C = require 'feather.libc'
local LL = require 'std.ll'
local Msg = require 'feather.message'
local Transform = require 'feather.transform'

local struct Node {
  transform : Transform
  extent : F.Vec3
  zindex : F.Veci
  data : &Msg.Receiver
  next : &Node -- next sibling
  prev : &Node
  children : &Node -- first child
  last : &Node -- last child
  parent : &Node
  planar : bool
}

terra Node:iter()
  var it : LL.MakeIterator(Node)
  it.cur = self.children
  return it
end

-- https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/
-- Generates an AABB from an OBB
local terra GenAABB(pos : F.Vec3, extent : F.Vec3, rot : F.Bivector3) : {F.Vec3, F.Vec3}
  if rot.xy == 0.0f and rot.xz == 0.0f and rot.yz == 0.0f then
    return {pos,extent}
  end

  --var p = F.GA3.exp(-rot)*pos*F.GA3.exp(rot)

  var rmat = Util.MatrixRotation(rot.yz, rot.xz, rot.xy)
  var p = F.Vec3{Util.VecMultiply3D(pos.v, rmat)}
  
  Util.MatrixAbs(&rmat)
  var e = F.Vec3{Util.VecMultiply3D(extent.v, rmat)}

  return {p,e}
end

local terra TransformRay(n : &Node, p : &F.Vec3, v : &F.Vec3) : {}
  @p = @p - n.transform.pos -- We first fix the point relative to this node, since all rotations are done around this node's origin

  if n.transform.rot.xz ~= 0 or n.transform.rot.yz ~= 0 then
    -- Transform p and v into this node's reference frame
    var rmat = Util.MatrixRotation(n.transform.rot.yz, n.transform.rot.xz, n.transform.rot.xy)
    v.v = Util.VecMultiply3D(v.v, rmat)
    p.v = Util.VecMultiply3D(p.v, rmat)
  elseif n.transform.rot.xy ~= 0 then
    Util.Rotate2D(v.v[0], v.v[1], n.transform.rot.xy, &v.v[0], &v.v[1])
    Util.Rotate2D(p.v[0], p.v[1], n.transform.rot.xy, &p.v[0], &p.v[1])
  end
end

-- Ray intersection with a 3D AABB centered on the origin
-- adapted from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
local terra RayBoxIntersect(pos : &F.Vec3, v : &F.Vec3, extent : &F.Vec3) : float
  var txmin = (-extent.x - pos.x) / v.x
  var txmax = (extent.x - pos.x) / v.x

  if txmin > txmax then txmin, txmax = txmax, txmin end 

  var tymin = (-extent.y - pos.y) / v.y
  var tymax = (extent.y - pos.y) / v.y

  if tymin > tymax then tymin, tymax = tymax, tymin end

  if (txmin > tymax) or (tymin > txmax) then
      return -1 
  end

  if tymin > txmin then
      txmin = tymin
  end

  if tymax < txmax then
      txmax = tymax
  end

  var tzmin = (-extent.z - pos.z) / v.z
  var tzmax = (extent.z - pos.z) / v.z

  if tzmin > tzmax then tzmin, tzmax = tzmax, tzmin end

  if (txmin > tzmax) or (tzmin > txmax) then
      return -1
  end

  if tzmin > txmin then
      txmin = tzmin
  end

  if tzmax < txmax then
      txmax = tzmax
  end
  
  return terralib.select(txmin < 0, txmax, txmin) 
end

-- Construct an RTree type from an allocator
local RTree = Util.type_template(function(A)
  local struct rtree {
    allocator : A
    root : &Node
	}

  rtree.Node = Node
  rtree.methods.RayBoxIntersect = RayBoxIntersect

  terra rtree:create(parent : &Node, tf : &Transform, extent : F.Vec3, zindex : F.Veci) : &Node
    var n : &Node = self.allocator:alloc(Node, 1)
    n.prev = nil
    n.next = nil
    n.children = nil
    n.last = nil
    n.parent = parent
    n.planar = true
    if parent ~= nil then
      LL.Prepend(n, &parent.children, &parent.last)
    else
      LL.Prepend(n, &self.root)
    end
    self:set(n, tf, extent, zindex)
    return n
  end

  local terra detach(node : &Node, root : &&Node) : {}
    if node.parent ~= nil then
      LL.Remove(node, &node.parent.children, &node.parent.last)
    else
      LL.Remove(node, root)
    end
  end

  terra rtree:destroy(node : &Node) : {}
    detach(node, &self.root)

    while node.children ~= nil do
      self:destroy(node.children)
    end

    self.allocator:free(node)
  end

  -- Sets the extend of a node to exactly contain it's children. This moves the children's relative position.
  terra rtree:contain(node : &Node) : {}
    var cur : &Node = node.children
    if cur ~= nil then
      var pos : F.Vec3 = cur.transform.pos
      var dim : F.Vec3 = [F.Vec3]{array(0.0f,0.0f,0.0f)}

      while cur ~= nil do
        var curextent : F.Vec3
        var curpos : F.Vec3
        curpos, curextent = GenAABB(cur.transform.pos, cur.extent, cur.transform.rot)
        escape
          for i = 0,2 do emit(quote pos.v[i] = Math.min(pos.v[i], cur.transform.pos.v[i] - cur.extent.v[i]) end) end
          for i = 0,2 do emit(quote dim.v[i] = Math.max(pos.v[i], cur.transform.pos.v[i] + cur.extent.v[i]) end) end
        end
      end

      node.extent = dim / 2.0f

      -- Our new position is relative to where our old position was, so we adjust all our children first
      var relative = pos + node.extent
      cur = node.children
      while cur ~= nil do
        cur.transform.pos = cur.transform.pos + relative
      end

      -- Then we move our own position in our parent to reflect our new rectangle center.
      node.transform.pos = node.transform.pos - relative 
    else
      node.extent.v = array(0.0f,0.0f,0.0f)
    end
  end

  terra rtree:set(node : &Node, tf : &Transform, extent : F.Vec3, zindex : F.Veci) : {}
    node.transform = @tf
    node.extent = extent
    node.zindex = zindex
  end

  terra rtree:sort(node : &Node, child : &Node) : {}
    if child ~= nil then
      -- Assume children are all sorted
      if child.parent == node then
        for i in node:iter() do
          if child.zindex.x > i.zindex.x then
            LL.Insert(child, i, &node.children)
            return
          end
        end
        LL.Append(child, &node.last, &node.children)
      end
    else
      -- To do a full sort, we detach all children temporarily, then rebuild the list
      var children = node.children
      node.children = nil
      node.last = nil

      while children ~= nil do
        children.prev = nil
        var cur = children
        children = children.next
        cur.next = nil
        self:sort(node, cur)
      end
    end
  end

  terra rtree:setparent(node : &Node, parent : &Node) : {}
    detach(node, &self.root)
    node.parent = parent

    if node.parent ~= nil then
      LL.Prepend(node, &node.parent.children, &node.parent.last)
    else
      LL.Prepend(node, &self.root)
    end
  end

  rtree.methods.query = CT(function(t) return CT.TemplateMethod(rtree, {F.Vec3, F.Vec3, CT.Value(t), CT.Function({&Node, &F.Vec3, &F.Vec3, CT.Value(t)}, bool)}, bool,
  function (S, P, V, D, FN)
    local terra recurse :: { &Node, F.Vec3, F.Vec3, D, FN } -> bool
    local terra check2D(cur : &Node, p : F.Vec3, v : F.Vec3, d : D, f : FN) : bool 
      TransformRay(cur, &p, &v)

      if cur.extent.z ~= 0 or cur.transform.pos.z ~= 0 or cur.transform.rot.yz ~= 0 or cur.transform.rot.xz ~= 0 then
        var t = RayBoxIntersect(&p, &v, &cur.extent)
        if t >= 0 then
          return recurse(cur, (p + v * t), v, d, f)
        end
      else
        if Math.abs(p.x) <= cur.extent.x and Math.abs(p.y) < cur.extent.y then
          return recurse(cur, p, v, d, f)
        end
      end

      return false
    end

    local struct NodeList { n : &Node, t : float, p : F.Vec3, v : F.Vec3, next : &NodeList };
    terra recurse(n : &Node, pos : F.Vec3, dir : F.Vec3, d : D, f : FN) : bool      
      if n.extent.z == 0 then
        if n.planar then
          dir.x = 0f
          dir.y = 0f
          dir = dir:normalize()
        end

        -- 2D nodes are sorted from highest to lowest z-index, regardless of any 3D rotation that was applied
        var cur : &Node = n.children

        -- If the 2D node was hit from behind we go backwards
        if cur ~= nil and dir.z < 0 then
          while cur.next ~= nil do
            cur = cur.next
          end
        end
        
        while cur ~= nil and cur.zindex.x >= 0 do -- x is the "topmost" zindex
          if check2D(cur, pos, dir, d, f) then
            return true
          end

          cur = terralib.select(dir.z < 0, cur.prev, cur.next)
        end

        if f(n, &pos, &dir, d) then
          return true
        end

        while cur ~= nil do
          if check2D(cur, pos, dir, d, f) then
            return true
          end

          cur = terralib.select(dir.z < 0, cur.prev, cur.next)
        end

        return false
      end

      var closest : &NodeList = nil
      var cur : &Node = n.children

      -- Find intersections and insert them (sorted by closeness) into a list
      while cur ~= nil do
        var p = pos
        var v = dir
        TransformRay(cur, &p, &v)
        var t = RayBoxIntersect(&p, &v, &cur.extent)
        if t >= 0 then
          var prev : &NodeList = nil
          var insert = closest
          var item = [&NodeList](C.fg_alloca([terralib.sizeof(NodeList)]))
          @item = NodeList{cur, t, p, v, nil }

          -- We use insertion sort here because we don't expect to have more than 8 elements.
          while insert ~= nil and (insert.t > item.t or (insert.t == item.t and insert.n.zindex.x > item.n.zindex.x)) do
            prev = insert
            insert = insert.next
          end

          item.next = insert
          if prev == nil then
            closest = item
          else
            prev.next = item
          end
        end
        cur = cur.next
      end

      -- Go through our list trying to get one of the nodes to accept the message
      while closest ~= nil do
        -- See if this node (or it's children) accepts the message
        if recurse(closest.n, (closest.p + closest.v * closest.t), closest.v, d, f) then
          return true
        end
        closest = closest.next
      end
      
      -- if all our children rejected the message, try processing it ourselves
      return f(n, &pos, &dir, d)
    end

    return terra(self : S, pos : P, dir : V, data : D, fn : FN) : bool
      var cur = self.root
      while cur ~= nil do
        var v = dir:normalize()
        var p = pos
        TransformRay(cur, &p, &v)
        return recurse(cur, p, v, data, fn)
      end
      return false
    end
  end) end, CT.TerraType)

  terra rtree:init() : {}
    self.root = nil
  end

  local terra freenode(cur : &Node) : {}
    while cur ~= nil do
      freenode(cur.children)
      var n : &Node = cur.next
      Alloc.free(cur)
      cur = n
    end
  end

  terra rtree:destruct() : {}
    freenode(self.root)
  end

  return rtree
end)

return RTree

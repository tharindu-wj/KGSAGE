// The neighbourhood graph's layout: d3-force, run to rest before first paint.
//
// Pure: no React, no DOM, so test/layout.test.mjs can run it under Node and
// prove the two properties a reviewer relies on --
//   1. deterministic: the same case and depth give the same positions, in
//      any browser, every time (a seeded random source and fixed starting
//      positions; the simulation never runs on a timer);
//   2. the argument stays readable: the three focus entities are PINNED
//      where the retired ego figures put them -- anchor left, true value
//      upper right, replacement lower right -- and every other node settles
//      by the focus entity it was reached from, so the true edge and the
//      corrupted edge always cross open space.
// A stable frame also means paging from case to case keeps the three focus
// entities on the same spots of the screen.

import {
  forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
} from 'd3-force'

export const WIDTH = 960
export const HEIGHT = 540

export const PINS = {
  anchor: [300, 270],
  true_filler: [672, 132],
  candidate: [672, 408],
}

// where each cluster settles, pushed away from the middle; shared
// neighbours sit BELOW the corrupted edge (anchor -> replacement), so the
// paths through them read as a detour and never cover its label
const HOMES = {
  anchor: [128, 270],
  true_filler: [862, 72],
  candidate: [862, 468],
  shared: [440, 452],
}

export const RADIUS = {
  anchor: 17, true_filler: 15, candidate: 15, shared: 8, hop1: 7, hop2: 4.5,
}

export function radiusOf(node) {
  if (node.role === 'context') return node.hop >= 2 ? RADIUS.hop2 : RADIUS.hop1
  return RADIUS[node.role]
}

function fnv1a(text) {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash >>> 0
}

function lcg(seed) {
  let state = seed >>> 0 || 1
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0
    return state / 4294967296
  }
}

// Label geometry, shared with Graph.jsx so the boxes the layout keeps
// apart are the ones the renderer draws. Focus entities carry a name and a
// role tag under the node; shared neighbours and one-hop context a short
// name; two-hop context no label at all (hover tells).
export const LABEL = {
  focus: { size: 13.5, max: 34, lines: 2 },
  small: { size: 11, max: 24, lines: 1 },
}
const CHAR_WIDTH = 0.57   // average glyph width of the system sans, in em

export function labelSpec(node) {
  if (node.role === 'anchor' || node.role === 'true_filler' || node.role === 'candidate') return LABEL.focus
  if (node.role === 'shared' || node.hop <= 1) return LABEL.small
  return null
}

export const truncate = (text, n) => (text.length <= n ? text : `${text.slice(0, n - 1)}…`)

function labelBox(node, text, r) {
  const spec = labelSpec(node)
  if (!spec || !text) return null
  const shown = truncate(text, spec.max)
  const width = shown.length * spec.size * CHAR_WIDTH + 6
  const height = spec.lines === 2 ? 30 : spec.size + 5
  // the box's centre sits under the node, where the text is drawn
  return { w: width, h: height, dy: r + (spec.lines === 2 ? 17 : 8) }
}

// A d3 force that pushes overlapping label boxes apart, along whichever
// axis overlaps less. Pinned nodes do not move; their neighbour moves for
// both. O(n^2) over labelled nodes only -- a few dozen at most.
function forceLabelBoxes(boxOf) {
  let nodes = []
  function force(alpha) {
    const strength = 0.6 * alpha + 0.05
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i]
      const boxA = boxOf(a)
      if (!boxA) continue
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j]
        const boxB = boxOf(b)
        if (!boxB) continue
        const dx = (b.x + b.vx) - (a.x + a.vx)
        const dy = (b.y + boxB.dy + b.vy) - (a.y + boxA.dy + a.vy)
        const overlapX = (boxA.w + boxB.w) / 2 - Math.abs(dx)
        const overlapY = (boxA.h + boxB.h) / 2 - Math.abs(dy)
        if (overlapX <= 0 || overlapY <= 0) continue
        const aFixed = a.fx != null
        const bFixed = b.fx != null
        if (aFixed && bFixed) continue
        const share = aFixed ? 0 : bFixed ? 1 : 0.5
        if (overlapX < overlapY) {
          const push = (dx < 0 ? -1 : 1) * overlapX * strength
          a.vx -= push * share
          b.vx += push * (1 - share)
        } else {
          const push = (dy < 0 ? -1 : 1) * overlapY * strength
          a.vy -= push * share
          b.vy += push * (1 - share)
        }
      }
    }
  }
  force.initialize = (simulationNodes) => { nodes = simulationNodes }
  return force
}

// The nodes and edges a depth shows: the focus entities and shared
// neighbours always, context nodes up to `depth` hops from a focus entity.
export function visibleSubgraph(neighbourhood, depth) {
  const nodes = neighbourhood.nodes.filter(
    (node) => node.role !== 'context' || node.hop <= depth)
  const ids = new Set(nodes.map((node) => node.id))
  const edges = neighbourhood.edges.filter(
    (edge) => ids.has(edge.h) && ids.has(edge.t))
  return { nodes, edges }
}

// The true and corrupted edges join pinned nodes, so their relation labels
// sit at fixed spots -- the clean-up treats those as boxes no other label
// may cover. (Geometry matches Graph.jsx: text centred at the midpoint,
// baseline 6 above it.)
export const EDGE_LABEL = { size: 11.5, max: 30, lift: 6 }

function edgeLabelObstacles(edges, nodes, relationLabelOf) {
  const roleOf = Object.fromEntries(nodes.map((node) => [node.id, node.role]))
  const obstacles = []
  for (const edge of edges) {
    if (edge.kind !== 'true' && edge.kind !== 'corrupted') continue
    const a = PINS[roleOf[edge.h]]
    const b = PINS[roleOf[edge.t]]
    if (!a || !b) continue
    const text = truncate(relationLabelOf(edge.r) || '', EDGE_LABEL.max)
    const x = (a[0] + b[0]) / 2
    const y = (a[1] + b[1]) / 2 - EDGE_LABEL.lift - EDGE_LABEL.size / 2 + 2
    obstacles.push({ x, y, fx: x, fy: y,
      box: { w: text.length * EDGE_LABEL.size * CHAR_WIDTH + 8, h: EDGE_LABEL.size + 6, dy: 0 } })
  }
  return obstacles
}

export function layout(neighbourhood, { depth = 1, seed = '', labelOf = null, relationLabelOf = null } = {}) {
  const { nodes, edges } = visibleSubgraph(neighbourhood, depth)
  const simNodes = nodes.map((node, i) => {
    const pin = PINS[node.role]
    const home = pin || HOMES[node.role === 'shared' ? 'shared' : node.near] || HOMES.anchor
    // a golden-angle spiral around the node's home: a deterministic start
    const angle = i * 2.399963229728653
    const reach = 40 + 9 * Math.sqrt(i)
    return {
      id: node.id, role: node.role, hop: node.hop, r: radiusOf(node), home,
      x: pin ? pin[0] : home[0] + reach * Math.cos(angle),
      y: pin ? pin[1] : home[1] + reach * Math.sin(angle),
      ...(pin ? { fx: pin[0], fy: pin[1] } : {}),
    }
  })

  // one spring per linked pair, however many relations link it
  const pairs = new Map()
  for (const edge of edges) {
    if (edge.h === edge.t) continue
    const key = edge.h < edge.t ? `${edge.h}\u0000${edge.t}` : `${edge.t}\u0000${edge.h}`
    if (!pairs.has(key)) pairs.set(key, { source: edge.h, target: edge.t })
  }

  // A shared neighbour is tied to two pinned nodes far apart; at full
  // strength those two springs drag it onto the straight line between them,
  // which is the corrupted edge. Its springs stay slack and its home holds
  // it below that line instead.
  const touchesShared = (link) => link.source.role === 'shared' || link.target.role === 'shared'
  const homeStrength = (d) => (d.fx != null ? 0 : d.role === 'shared' ? 0.35 : 0.07)
  const simulation = forceSimulation(simNodes)
    .randomSource(lcg(fnv1a(seed)))
    .force('link', forceLink([...pairs.values()]).id((d) => d.id)
      .distance((link) => 30 + link.source.r + link.target.r)
      .strength((link) => (touchesShared(link) ? 0.04 : 0.45)))
    .force('charge', forceManyBody()
      .strength((d) => (d.hop >= 2 ? -36 : -150)).distanceMax(260))
    .force('collide', forceCollide((d) => d.r + (d.hop >= 2 ? 3 : 15)).iterations(2))
    .force('x', forceX((d) => d.home[0]).strength(homeStrength))
    .force('y', forceY((d) => d.home[1]).strength(homeStrength))
    .stop()
  if (labelOf) {
    for (const node of simNodes) node.box = labelBox(node, labelOf(node.id), node.r)
    simulation.force('labels', forceLabelBoxes((d) => d.box))
  }
  const ticksToRest = Math.ceil(Math.log(simulation.alphaMin())
                                / Math.log(1 - simulation.alphaDecay()))
  simulation.tick(ticksToRest)
  // As the simulation cools the springs re-pack a few labels; a short,
  // deterministic clean-up separates whatever still overlaps, the fixed
  // edge labels included.
  if (labelOf) {
    const obstacles = relationLabelOf ? edgeLabelObstacles(edges, nodes, relationLabelOf) : []
    resolveLabelOverlaps(simNodes.concat(obstacles))
  }
  const positions = {}
  for (const node of simNodes) {
    positions[node.id] = { x: node.x, y: node.y, r: node.r, box: node.box || null }
  }
  return { nodes, edges, positions, viewBox: viewBoxOf(simNodes) }
}

// Move label-carrying nodes apart until no two boxes overlap (or the pass
// budget runs out): each overlapping pair splits along its smaller overlap,
// a pinned node never moves. Positions only -- the simulation is at rest.
function resolveLabelOverlaps(simNodes, passes = 240) {
  const labelled = simNodes.filter((node) => node.box)
  for (let pass = 0; pass < passes; pass++) {
    let moved = false
    for (let i = 0; i < labelled.length; i++) {
      for (let j = i + 1; j < labelled.length; j++) {
        const a = labelled[i]
        const b = labelled[j]
        const dx = b.x - a.x
        const dy = (b.y + b.box.dy) - (a.y + a.box.dy)
        const overlapX = (a.box.w + b.box.w) / 2 - Math.abs(dx) + 1
        const overlapY = (a.box.h + b.box.h) / 2 - Math.abs(dy) + 1
        if (overlapX <= 0 || overlapY <= 0) continue
        const aFixed = a.fx != null
        const bFixed = b.fx != null
        if (aFixed && bFixed) continue
        const share = aFixed ? 0 : bFixed ? 1 : 0.5
        if (overlapX < overlapY) {
          const push = (dx < 0 ? -1 : 1) * overlapX
          a.x -= push * share
          b.x += push * (1 - share)
        } else {
          const push = (dy < 0 ? -1 : 1) * overlapY
          a.y -= push * share
          b.y += push * (1 - share)
        }
        moved = true
      }
    }
    if (!moved) return
  }
}

// How many pairs of label boxes still overlap -- for the layout test.
export function labelOverlaps(positions) {
  const placed = Object.values(positions).filter((p) => p.box)
  let overlaps = 0
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const a = placed[i]
      const b = placed[j]
      if (Math.abs(a.x - b.x) < (a.box.w + b.box.w) / 2 - 1
          && Math.abs((a.y + a.box.dy) - (b.y + b.box.dy)) < (a.box.h + b.box.h) / 2 - 1) overlaps += 1
    }
  }
  return overlaps
}

// The fixed frame, widened only when a node settles outside it; the
// margins leave room for the labels drawn under the nodes.
export function viewBoxOf(simNodes) {
  let x0 = 0, y0 = 0, x1 = WIDTH, y1 = HEIGHT
  for (const node of simNodes) {
    x0 = Math.min(x0, node.x - node.r - 90)
    x1 = Math.max(x1, node.x + node.r + 90)
    y0 = Math.min(y0, node.y - node.r - 16)
    y1 = Math.max(y1, node.y + node.r + 40)
  }
  return [x0, y0, x1 - x0, y1 - y0]
}

// Edge geometry: a straight line, or a gentle curve when several edges
// join the same pair. Offsets are fixed relative to the pair's canonical
// orientation, so two relations running opposite ways still part.
export function edgeShapes(edges, positions) {
  const groups = new Map()
  for (const edge of edges) {
    const key = edge.h < edge.t ? `${edge.h}\u0000${edge.t}` : `${edge.t}\u0000${edge.h}`
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(edge)
  }
  const shapes = []
  for (const group of groups.values()) {
    group.forEach((edge, k) => {
      const a = positions[edge.h]
      const b = positions[edge.t]
      if (!a || !b) return
      if (edge.h === edge.t) {
        const top = a.y - a.r
        shapes.push({ edge, loop: true,
          d: `M${a.x - 5},${top} C${a.x - 22},${top - 26} ${a.x + 22},${top - 26} ${a.x + 5},${top}`,
          mx: a.x, my: top - 20 })
        return
      }
      const spread = (k - (group.length - 1) / 2) * 16
      const canonical = edge.h < edge.t ? 1 : -1
      shapes.push({ edge, ...curve(a, b, spread * canonical) })
    })
  }
  return shapes
}

function curve(a, b, offset) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const length = Math.hypot(dx, dy) || 1
  const cx = (a.x + b.x) / 2 - (dy / length) * offset
  const cy = (a.y + b.y) / 2 + (dx / length) * offset
  const [sx, sy] = toward(a, cx, cy, a.r + 1.5)
  const [ex, ey] = toward(b, cx, cy, b.r + 2.5)
  return {
    d: `M${sx.toFixed(1)},${sy.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`,
    // the curve's own midpoint (t = 0.5), where its label sits
    mx: 0.25 * sx + 0.5 * cx + 0.25 * ex,
    my: 0.25 * sy + 0.5 * cy + 0.25 * ey,
  }
}

function toward(point, cx, cy, distance) {
  const dx = cx - point.x
  const dy = cy - point.y
  const length = Math.hypot(dx, dy) || 1
  return [point.x + (dx / length) * distance, point.y + (dy / length) * distance]
}

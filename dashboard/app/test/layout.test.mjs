// The layout's two promises, checked under Node over real cases:
// deterministic (same case, same depth -> same positions) and readable
// (focus entities pinned; context nodes settle on their own entity's side).
//
//   node test/layout.test.mjs ../../outputs/dashboard/data/<run>.json
//
// Any run record or view model works: both carry cases[].neighbourhood.

import { readFileSync } from 'node:fs'
import { edgeShapes, labelOverlaps, layout, PINS } from '../src/graph/layout.js'

const file = process.argv[2]
if (!file) {
  console.error('usage: node test/layout.test.mjs <view model or run record>')
  process.exit(2)
}
const data = JSON.parse(readFileSync(file, 'utf8'))
const every = Number(process.argv[3] || 9)
const sample = data.cases.filter((_, i) => i % every === 0)

let failures = 0
function check(label, ok, detail = '') {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `   [${detail}]`}`)
  if (!ok) failures += 1
}

const distance = (p, q) => Math.hypot(p.x - q.x, p.y - q.y)
const stats = { layouts: 0, unstable: 0, pinMoved: 0, notFinite: 0, sided: 0,
  wrongSide: 0, overlaps: 0, nodes: 0, shapeless: 0, slowest: 0, widened: 0,
  labelOverlaps: 0, labelOverlapsWithout: 0 }
// the labels the dashboard draws, from the same entity table
const labelOf = (id) => (data.entities[id] && data.entities[id].label) || id
const relationLabelOf = (id) => (data.relations[id] && data.relations[id].label) || id

for (const kase of sample) {
  for (const depth of [1, 2]) {
    const started = performance.now()
    const first = layout(kase.neighbourhood, { depth, seed: kase.id, labelOf, relationLabelOf })
    stats.slowest = Math.max(stats.slowest, performance.now() - started)
    const second = layout(kase.neighbourhood, { depth, seed: kase.id, labelOf, relationLabelOf })
    stats.labelOverlaps += labelOverlaps(first.positions)
    // the same layout without the label force, boxes measured the same way
    const plain = layout(kase.neighbourhood, { depth, seed: kase.id })
    for (const id of Object.keys(plain.positions)) plain.positions[id].box = first.positions[id].box
    stats.labelOverlapsWithout += labelOverlaps(plain.positions)
    stats.layouts += 1
    if (JSON.stringify(first.positions) !== JSON.stringify(second.positions)) stats.unstable += 1
    const pins = {}
    for (const node of first.nodes) {
      const p = first.positions[node.id]
      stats.nodes += 1
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) stats.notFinite += 1
      if (PINS[node.role]) {
        pins[node.role] = p
        if (p.x !== PINS[node.role][0] || p.y !== PINS[node.role][1]) stats.pinMoved += 1
      }
    }
    // a context node reached from one focus entity should sit nearer that
    // entity than the other two
    for (const node of first.nodes) {
      if (node.role !== 'context' || !pins[node.near]) continue
      const p = first.positions[node.id]
      const own = distance(p, pins[node.near])
      const others = Object.entries(pins).filter(([role]) => role !== node.near)
        .map(([, q]) => distance(p, q))
      stats.sided += 1
      if (others.some((d) => d < own)) stats.wrongSide += 1
    }
    const placed = first.nodes.map((node) => first.positions[node.id])
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        if (distance(placed[i], placed[j]) < placed[i].r + placed[j].r) stats.overlaps += 1
      }
    }
    const shapes = edgeShapes(first.edges, first.positions)
    if (shapes.length !== first.edges.length
        || shapes.some((s) => s.d.includes('NaN'))) stats.shapeless += 1
    if (first.viewBox[2] > 960 || first.viewBox[3] > 540) stats.widened += 1
  }
}

console.log(`layout over ${sample.length} cases x 2 depths (${stats.layouts} layouts, ${stats.nodes} nodes)`)
check('deterministic: a second run gives identical positions', stats.unstable === 0, `${stats.unstable} differ`)
check('anchor, true value and replacement stay on their pins', stats.pinMoved === 0, `${stats.pinMoved} moved`)
check('every position is finite', stats.notFinite === 0, `${stats.notFinite}`)
check('every edge gets a drawable shape', stats.shapeless === 0, `${stats.shapeless} layouts`)
const wrongShare = stats.wrongSide / Math.max(stats.sided, 1)
check(`context nodes on their own entity's side (${(100 * (1 - wrongShare)).toFixed(1)}%)`,
  wrongShare < 0.1, `${stats.wrongSide} of ${stats.sided}`)
check(`label boxes overlap far less with the label force (${stats.labelOverlaps} pairs vs ${stats.labelOverlapsWithout} without)`,
  stats.labelOverlaps * 4 <= stats.labelOverlapsWithout, `${stats.labelOverlaps} vs ${stats.labelOverlapsWithout}`)
console.log(`  info  overlapping node pairs: ${stats.overlaps} in ${stats.layouts} layouts;` +
  ` frame widened in ${stats.widened}; slowest layout ${stats.slowest.toFixed(1)} ms`)
console.log(failures ? `\n${failures} FAILED` : '\nALL PASS')
process.exit(failures ? 1 : 0)

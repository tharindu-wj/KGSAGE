// The neighbourhood graph: one case's subgraph, drawn as SVG in React and
// laid out by d3-force (layout.js) before first paint.
//
// Reading it: the three focus entities are pinned -- anchor left, true value
// upper right, replacement lower right. The solid edge is the true fact, the
// dashed one the corruption. Diamonds are shared neighbours: entities linked
// to both the anchor and the replacement, the graph's own corroboration.
// Grey nodes are context, one or two hops out.
//
// Interaction: hover or focus for what a node or edge is; drag a node to
// untangle; drag the background to pan; the buttons zoom, reset, switch
// depth, swap to the list view (the table twin), and export the picture.

import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { EDGE_LABEL, LABEL, edgeShapes, layout, truncate } from './layout.js'
import { usePalette } from '../palette.js'
import { ROLE_NAME, entity, label, relationLabel } from '../labels.js'

const EDGE_NAME = {
  true: 'the true fact',
  corrupted: 'the corruption',
  bridge: 'a fact through a shared neighbour',
  context: 'a fact of the graph',
}
const DRAW_ORDER = { context: 0, bridge: 1, true: 2, corrupted: 3 }
const NEAR_NAME = { anchor: 'the anchor', true_filler: 'the true value', candidate: 'the replacement' }

function download(blob, name) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// ?depth=2 in the URL opens every graph at two hops (a deep link to a view)
function initialDepth() {
  try {
    return new URLSearchParams(window.location.search).get('depth') === '2' ? 2 : 1
  } catch {
    return 1
  }
}

export default function Graph({ kase, data }) {
  const [depth, setDepth] = useState(initialDepth)
  const [listView, setListView] = useState(false)
  const [hover, setHover] = useState(null)
  const [moved, setMoved] = useState({})
  const [view, setView] = useState({ k: 1, x: 0, y: 0 })
  const svgRef = useRef(null)
  const viewportRef = useRef(null)
  const drag = useRef(null)
  const palette = usePalette()
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '')

  // a new case or a new depth starts again from the computed layout
  useEffect(() => {
    setMoved({})
    setView({ k: 1, x: 0, y: 0 })
    setHover(null)
  }, [kase.id, depth])

  const base = useMemo(
    () => layout(kase.neighbourhood, {
      depth, seed: kase.id,
      labelOf: (id) => label(data, id),
      relationLabelOf: (id) => relationLabel(data, id),
    }),
    [kase, depth, data])
  const positions = useMemo(() => {
    const merged = { ...base.positions }
    for (const [id, xy] of Object.entries(moved)) merged[id] = { ...merged[id], ...xy }
    return merged
  }, [base, moved])
  const roleOf = useMemo(
    () => Object.fromEntries(base.nodes.map((node) => [node.id, node.role])), [base])

  const classOf = (edge) => {
    if (edge.kind !== 'context') return edge.kind
    const roles = [roleOf[edge.h], roleOf[edge.t]]
    const bridgesShared = roles.includes('shared')
      && (roles.includes('anchor') || roles.includes('candidate'))
    const direct = roles.includes('anchor') && roles.includes('candidate')
    return bridgesShared || direct ? 'bridge' : 'context'
  }
  const shapes = useMemo(
    () => edgeShapes(base.edges, positions)
      .map((shape) => ({ ...shape, cls: classOf(shape.edge) }))
      .sort((a, b) => DRAW_ORDER[a.cls] - DRAW_ORDER[b.cls]),
    // classOf reads roleOf only
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [base, positions, roleOf])

  const colour = {
    anchor: palette['g-anchor'],
    true_filler: palette['g-true'],
    candidate: palette['g-replacement'],
    shared: palette['g-shared'],
  }
  const stroke = {
    true: palette['g-true'],
    corrupted: palette['g-replacement'],
    bridge: palette['g-bridge'],
    context: palette['g-edge'],
  }
  const fillOf = (node) => colour[node.role]
    || (node.hop >= 2 ? palette['g-hop2'] : palette['g-hop1'])

  // ---- pointer: drag a node, pan the background ----------------------------
  const toLayout = (event) => {
    const point = svgRef.current.createSVGPoint()
    point.x = event.clientX
    point.y = event.clientY
    const p = point.matrixTransform(viewportRef.current.getScreenCTM().inverse())
    return { x: p.x, y: p.y }
  }
  const startPan = (event) => {
    drag.current = { kind: 'pan', x: event.clientX, y: event.clientY, view }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const startNode = (event, id) => {
    event.stopPropagation()
    drag.current = { kind: 'node', id }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const onMove = (event) => {
    const current = drag.current
    if (!current) return
    if (current.kind === 'pan') {
      const scale = svgRef.current.getScreenCTM().a || 1
      setView({ ...current.view,
        x: current.view.x + (event.clientX - current.x) / scale,
        y: current.view.y + (event.clientY - current.y) / scale })
    } else {
      setMoved((m) => ({ ...m, [current.id]: toLayout(event) }))
    }
  }
  const endDrag = () => { drag.current = null }
  const zoom = (factor) => setView((v) => ({ ...v, k: Math.min(4, Math.max(0.4, v.k * factor)) }))
  const reset = () => { setMoved({}); setView({ k: 1, x: 0, y: 0 }) }

  const [vx, vy, vw, vh] = base.viewBox
  const cx = vx + vw / 2
  const cy = vy + vh / 2
  const viewportTransform =
    `translate(${view.x} ${view.y}) translate(${cx} ${cy}) scale(${view.k}) translate(${-cx} ${-cy})`

  // ---- export: the picture as a file ---------------------------------------
  const fileStem = `${data.run}_${kase.id}_${depth}hop`
  const exportableSvg = () => {
    const clone = svgRef.current.cloneNode(true)
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    clone.setAttribute('width', String(Math.round(vw)))
    clone.setAttribute('height', String(Math.round(vh)))
    clone.querySelectorAll('[data-hit]').forEach((element) => element.remove())
    clone.querySelector('[data-viewport]')?.removeAttribute('transform')
    return new XMLSerializer().serializeToString(clone)
  }
  const exportSvg = () => download(new Blob([exportableSvg()], { type: 'image/svg+xml' }), `${fileStem}.svg`)
  const exportPng = () => {
    const url = URL.createObjectURL(new Blob([exportableSvg()], { type: 'image/svg+xml' }))
    const image = new Image()
    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(vw * 2)
      canvas.height = Math.round(vh * 2)
      const context = canvas.getContext('2d')
      context.scale(2, 2)
      context.drawImage(image, 0, 0, vw, vh)
      URL.revokeObjectURL(url)
      canvas.toBlob((blob) => blob && download(blob, `${fileStem}.png`), 'image/png')
    }
    image.src = url
  }

  // ---- words for hover, focus and the screen reader -------------------------
  const nodeById = Object.fromEntries(base.nodes.map((node) => [node.id, node]))
  const describeNode = (node) => {
    const info = entity(data, node.id)
    const role = ROLE_NAME[node.role]
      || `${node.hop} hop${node.hop > 1 ? 's' : ''} from ${NEAR_NAME[node.near] || 'a focus entity'}`
    return {
      title: info.label || node.id,
      detail: [role, (info.kinds || []).slice(0, 3).join(', '),
        `${info.degree} fact${info.degree === 1 ? '' : 's'} in the graph`].filter(Boolean).join(' · '),
      description: info.description || '',
    }
  }
  const describeEdge = (shape) => ({
    title: `${label(data, shape.edge.h)} —${relationLabel(data, shape.edge.r)}→ ${label(data, shape.edge.t)}`,
    detail: EDGE_NAME[shape.cls],
    description: '',
  })
  const caption = hover
    ? (hover.node ? describeNode(nodeById[hover.node]) : describeEdge(hover.edge))
    : null
  const shared = kase.structure.shared_neighbours
  const ariaLabel = `Neighbourhood of ${label(data, kase.anchor)}: ${base.nodes.length} entities and ${base.edges.length} facts shown. `
    + `The corrupted edge links ${label(data, kase.anchor)} to ${label(data, kase.candidate)}, which `
    + (kase.structure.direct_edge ? 'is already directly linked to it'
      : shared ? `shares ${shared} neighbour${shared === 1 ? '' : 's'} with it` : 'shares no neighbour with it')
    + '. The list view gives every fact as text.'

  const markers = Object.entries(stroke).map(([cls, fill]) => (
    <marker key={cls} id={`${uid}-${cls}`} viewBox="0 0 10 10" refX="8.6" refY="5"
      markerWidth={cls === 'true' || cls === 'corrupted' ? 10 : 7}
      markerHeight={cls === 'true' || cls === 'corrupted' ? 10 : 7}
      markerUnits="userSpaceOnUse" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill={fill} />
    </marker>
  ))

  // labels wear a halo in the card colour so they stay legible over edges;
  // paint-order goes through style, which React maps to CSS for certain
  const halo = {
    stroke: palette['g-halo'], strokeWidth: 3.5, strokeLinejoin: 'round',
    style: { paintOrder: 'stroke', pointerEvents: 'none' },
  }
  const focusOrder = { context: 0, shared: 1, true_filler: 2, candidate: 2, anchor: 3 }
  const drawnNodes = [...base.nodes].sort(
    (a, b) => focusOrder[a.role] - focusOrder[b.role] || b.hop - a.hop)

  return (
    <div className="graph">
      <div className="graph-controls" role="toolbar" aria-label="Graph controls">
        <div className="segmented" role="radiogroup" aria-label="How far out">
          {[1, 2].map((d) => (
            <button key={d} role="radio" aria-checked={depth === d}
              className={depth === d ? 'on' : ''} onClick={() => setDepth(d)}>
              {d} hop{d > 1 ? 's' : ''}
            </button>
          ))}
        </div>
        <button className={`ghost${listView ? ' on' : ''}`} aria-pressed={listView}
          onClick={() => setListView((v) => !v)}>
          {listView ? 'graph view' : 'list view'}
        </button>
        {!listView && (
          <>
            <span className="control-gap" />
            <button className="ghost icon" onClick={() => zoom(1 / 1.25)} aria-label="Zoom out">{'−'}</button>
            <button className="ghost icon" onClick={() => zoom(1.25)} aria-label="Zoom in">+</button>
            <button className="ghost" onClick={reset}>reset</button>
            <button className="ghost" onClick={exportSvg}>SVG</button>
            <button className="ghost" onClick={exportPng}>PNG</button>
          </>
        )}
      </div>

      {listView ? (
        <GraphTable data={data} base={base} classOf={classOf} />
      ) : (
        <svg ref={svgRef} className="graph-svg" viewBox={base.viewBox.join(' ')}
          role="img" aria-label={ariaLabel}
          onPointerMove={onMove} onPointerUp={endDrag} onPointerCancel={endDrag}
          fontFamily='system-ui, -apple-system, "Segoe UI", sans-serif'>
          <defs>{markers}</defs>
          <rect x={vx} y={vy} width={vw} height={vh} fill={palette.card}
            onPointerDown={startPan} style={{ cursor: 'grab' }} />
          <g ref={viewportRef} data-viewport transform={viewportTransform}>
            {shapes.map((shape, i) => {
              const strong = shape.cls === 'true' || shape.cls === 'corrupted'
              return (
                <path key={`e${i}`} d={shape.d} fill="none" stroke={stroke[shape.cls]}
                  strokeWidth={strong ? 2.6 : shape.cls === 'bridge' ? 1.6 : 1.1}
                  strokeDasharray={shape.cls === 'corrupted' ? '7 5' : undefined}
                  strokeLinecap="round"
                  markerEnd={`url(#${uid}-${shape.cls})`} />
              )
            })}
            {shapes.map((shape, i) => (
              <path key={`h${i}`} data-hit d={shape.d} fill="none" stroke="transparent"
                strokeWidth="12" style={{ pointerEvents: 'stroke' }}
                onPointerEnter={() => setHover({ edge: shape })}
                onPointerLeave={() => setHover(null)} />
            ))}
            {shapes.filter((s) => s.cls === 'true' || s.cls === 'corrupted').map((shape, i) => (
              <text key={`el${i}`} x={shape.mx} y={shape.my - EDGE_LABEL.lift} textAnchor="middle"
                fontSize={EDGE_LABEL.size} fill={palette.muted} {...halo}>
                {truncate(relationLabel(data, shape.edge.r), EDGE_LABEL.max)}
              </text>
            ))}
            {drawnNodes.map((node) => {
              const p = positions[node.id]
              const focus = Boolean(ROLE_NAME[node.role]) && node.role !== 'shared'
              const common = {
                tabIndex: focus || node.role === 'shared' ? 0 : -1,
                onPointerEnter: () => setHover({ node: node.id }),
                onPointerLeave: () => setHover(null),
                onFocus: () => setHover({ node: node.id }),
                onBlur: () => setHover(null),
                onPointerDown: (event) => startNode(event, node.id),
                style: { cursor: 'grab' },
                'aria-label': describeNode(node).title,
              }
              if (node.role === 'shared') {
                const s = p.r + 2
                return (
                  <path key={node.id} {...common}
                    d={`M${p.x},${p.y - s} L${p.x + s},${p.y} L${p.x},${p.y + s} L${p.x - s},${p.y} Z`}
                    fill={palette['g-shared']} stroke={palette.card} strokeWidth="2" />
                )
              }
              return (
                <circle key={node.id} {...common} cx={p.x} cy={p.y} r={p.r}
                  fill={fillOf(node)} stroke={palette.card} strokeWidth={focus ? 2.5 : 1.5} />
              )
            })}
            {drawnNodes.map((node) => {
              const p = positions[node.id]
              const name = label(data, node.id)
              if (ROLE_NAME[node.role] && node.role !== 'shared') {
                return (
                  <g key={`l${node.id}`} style={{ pointerEvents: 'none' }}>
                    <text x={p.x} y={p.y + p.r + 15} textAnchor="middle" fontSize={LABEL.focus.size}
                      fontWeight="650" fill={palette.ink} {...halo}>{truncate(name, LABEL.focus.max)}</text>
                    <text x={p.x} y={p.y + p.r + 29} textAnchor="middle" fontSize="10.5"
                      letterSpacing="0.08em" fill={palette.muted} {...halo}>
                      {ROLE_NAME[node.role].toUpperCase()}
                    </text>
                  </g>
                )
              }
              if (node.role === 'shared' || node.hop <= 1) {
                return (
                  <text key={`l${node.id}`} x={p.x} y={p.y + p.r + 12} textAnchor="middle"
                    fontSize={LABEL.small.size} fill={palette.muted} {...halo}>
                    {truncate(name, LABEL.small.max)}
                  </text>
                )
              }
              return null
            })}
          </g>
        </svg>
      )}

      {!listView && (
        <p className="graph-caption" aria-live="polite">
          {caption ? (
            <>
              <b>{caption.title}</b> <span>{caption.detail}</span>
              {caption.description && <span className="caption-description"> {'—'} {caption.description}</span>}
            </>
          ) : (
            <span className="hint">Hover or focus a node or an edge to see what it is; drag to rearrange.</span>
          )}
        </p>
      )}
      <Legend palette={palette} />
    </div>
  )
}

function Legend({ palette }) {
  const dot = (fill, r = 6) => (
    <svg width="16" height="16" aria-hidden="true">
      <circle cx="8" cy="8" r={r} fill={fill} stroke={palette.card} strokeWidth="1.5" />
    </svg>
  )
  const line = (colour, dashed) => (
    <svg width="26" height="10" aria-hidden="true">
      <line x1="1" y1="5" x2="25" y2="5" stroke={colour} strokeWidth="2.6"
        strokeDasharray={dashed ? '6 4' : undefined} strokeLinecap="round" />
    </svg>
  )
  const diamond = (
    <svg width="16" height="16" aria-hidden="true">
      <path d="M8,2 L14,8 L8,14 L2,8 Z" fill={palette['g-shared']} />
    </svg>
  )
  return (
    <ul className="legend">
      <li>{dot(palette['g-anchor'])} anchor (kept in place)</li>
      <li>{dot(palette['g-true'])} true value</li>
      <li>{dot(palette['g-replacement'])} replacement</li>
      <li>{line(palette['g-true'], false)} true fact</li>
      <li>{line(palette['g-replacement'], true)} corruption</li>
      <li>{diamond} shared neighbour</li>
      <li>{line(palette['g-bridge'], false)} link through it</li>
      <li>{dot(palette['g-hop1'], 4.5)}{dot(palette['g-hop2'], 3.5)} context, 1 and 2 hops out</li>
    </ul>
  )
}

function GraphTable({ data, base, classOf }) {
  const rows = [...base.edges].sort(
    (a, b) => DRAW_ORDER[classOf(b)] - DRAW_ORDER[classOf(a)])
  return (
    <div className="graph-table-wrap">
      <p className="graph-table-note">
        {base.nodes.length} entities and {base.edges.length} facts in this view, the
        corruption and the true fact first.
      </p>
      <table className="graph-table">
        <thead>
          <tr><th>from</th><th>relation</th><th>to</th><th>what it is</th></tr>
        </thead>
        <tbody>
          {rows.map((edge, i) => (
            <tr key={i} className={`row-${classOf(edge)}`}>
              <td>{label(data, edge.h)}</td>
              <td>{relationLabel(data, edge.r)}</td>
              <td>{label(data, edge.t)}</td>
              <td>{EDGE_NAME[classOf(edge)]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

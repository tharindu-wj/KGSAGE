// What the graph records about the anchor and the replacement -- the same
// facts the LLM was shown, the corrupted relation's first.

import { label, relationLabel } from '../labels.js'

function FactList({ facts, focus, data, relation }) {
  return (
    <ul className="fact-list">
      {facts.map(([h, r, t], i) => (
        <li key={i} className={r === relation ? 'same-relation' : ''}>
          <span className={h === focus ? 'focus' : ''}>{label(data, h)}</span>
          <span className="fact-relation"> {'—'}{relationLabel(data, r)}{'→'} </span>
          <span className={t === focus ? 'focus' : ''}>{label(data, t)}</span>
        </li>
      ))}
    </ul>
  )
}

export default function Facts({ kase, data }) {
  const relation = kase.true.r
  return (
    <div className="facts">
      <div>
        <h3>About {label(data, kase.anchor)}, the anchor
          <small> {kase.facts.anchor.length} shown</small></h3>
        <FactList facts={kase.facts.anchor} focus={kase.anchor} data={data} relation={relation} />
      </div>
      <div>
        <h3>About {label(data, kase.candidate)}, the replacement
          <small> {kase.facts.candidate.length} shown</small></h3>
        <FactList facts={kase.facts.candidate} focus={kase.candidate} data={data} relation={relation} />
      </div>
    </div>
  )
}

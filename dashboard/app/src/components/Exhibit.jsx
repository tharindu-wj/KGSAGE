// What the graph itself measures about one corruption -- recomputed from
// the dataset by inference/corrupt.py, never read back from the generator.
// Four small panels: kind, corroboration, support, and how connected the
// three entities are.

import {
  LEVEL, count, entity, label, pct, plural, relationLabel, supportSentence, tripleText,
} from '../labels.js'

export default function Exhibit({ kase, data }) {
  const s = kase.structure
  const relation = relationLabel(data, kase.true.r)
  const replacement = entity(data, kase.candidate)
  const trueValue = entity(data, kase.true_filler)
  const sharedKinds = new Set((trueValue.kinds || []).filter((k) => (replacement.kinds || []).includes(k)))
  const support = s.support
  const sentence = supportSentence(kase, data)
  const baseline = s.shared_baseline
  const typical = baseline
    ? (Number.isInteger(baseline.typical) ? baseline.typical : baseline.typical.toFixed(1))
    : null
  const chains = new Set(s.chains || [])
  const degrees = [
    ['anchor', kase.anchor],
    ['true value', kase.true_filler],
    ['replacement', kase.candidate],
  ].map(([role, id]) => ({ role, id, degree: entity(data, id).degree || 0 }))
  const maxDegree = Math.max(1, ...degrees.map((d) => d.degree))

  return (
    <div className="exhibit">
      <section className="exhibit-panel">
        <h3>Kind</h3>
        <p className="exhibit-line">
          This slot of {'“'}{relation}{'”'} usually holds
        </p>
        <p className="pills">
          {s.slot_usual_kinds.map(([kind, n]) => (
            <span key={kind} className="pill faint">{kind} {'×'}{count(n)}</span>
          ))}
          {!s.slot_usual_kinds.length && <span className="muted">no kinds recorded</span>}
        </p>
        <p className="exhibit-line">{label(data, kase.candidate)} is</p>
        <p className="pills">
          {(replacement.kinds || []).map((kind) => (
            <span key={kind} className={`pill${sharedKinds.has(kind) ? ' hot' : ''}`}>{kind}</span>
          ))}
          {!(replacement.kinds || []).length && <span className="muted">no kinds recorded</span>}
        </p>
        <p className="exhibit-verdict">
          {s.kind_match === true && <><span className="tick">{'✓'}</span> shares a kind with the true value, {label(data, kase.true_filler)}</>}
          {s.kind_match === false && <><span className="cross">{'✗'}</span> shares no kind with the true value, {label(data, kase.true_filler)}</>}
          {s.kind_match == null && <>kinds unknown for one of them</>}
          {s.kind_peers != null && <><br />{count(s.kind_peers)} of this slot{'’'}s {count(s.slot_occupants)} occupants share a kind with it</>}
        </p>
      </section>

      <section className="exhibit-panel">
        <h3>Corroboration</h3>
        <p className="level-badge">
          <span className={`swatch lvl-${s.corroboration}`} aria-hidden="true" />
          {LEVEL[s.corroboration].label}
        </p>
        <p className="exhibit-line">{LEVEL[s.corroboration].means}.</p>
        {s.direct_facts.length > 0 && (
          <ul className="paths">
            {s.direct_facts.map((fact, i) => <li key={i}>{tripleText(data, fact)}</li>)}
          </ul>
        )}
        <p className="exhibit-verdict">
          {plural(s.shared_neighbours, 'shared neighbour')}
          {baseline && (
            <>; the typical one of this slot{'’'}s {count(baseline.compared)} other
              entities shares {typical}, and {pct(Math.round(baseline.as_many * 1000), 1000)} share
              as many or more</>
          )}.
        </p>
        {s.bridges.length > 0 && (
          <ul className="paths">
            {s.bridges.map((bridge) => (
              <li key={bridge.via}>
                <span className="via">via {label(data, bridge.via)}</span>{' '}
                <small className="muted">{count(entity(data, bridge.via).degree)} facts</small>
                {chains.has(bridge.via) && <span className="chain-tag">chain</span>}
                <span className="path-facts">
                  {[...bridge.anchor_side, ...bridge.candidate_side].map((fact) => tripleText(data, fact)).join('; ')}
                </span>
              </li>
            ))}
          </ul>
        )}
        {s.shared_neighbours > s.bridges.length && s.bridges.length > 0 && (
          <p className="exhibit-note">Chains first, then the least-connected: a small
            shared neighbour is stronger corroboration than a hub everything touches.
            {' '}{plural(s.shared_neighbours - s.bridges.length, 'more is', 'more are')} not shown.</p>
        )}
      </section>

      <section className="exhibit-panel">
        <h3>Support</h3>
        {support ? (
          <>
            <p className="exhibit-line">{sentence}</p>
            <div className="support-meter" role="img"
              aria-label={`support ${Math.round(100 * support.share)}%, guard at 50%`}>
              <span className="support-fill" style={{ width: `${100 * support.share}%` }} />
              <span className="support-guard" style={{ left: '50%' }} />
            </div>
            <p className="support-scale"><span>0</span><span>guard 50%</span><span>100%</span></p>
            <p className="exhibit-note">How strongly the graph{'’'}s own regularities predict
              the replacement. Above the guard it is probably a true fact the graph is missing.</p>
          </>
        ) : (
          <p className="exhibit-line muted">No value of the anchor is held by five or more
            entities, so the graph{'’'}s regularities cannot judge this one.</p>
        )}
      </section>

      <section className="exhibit-panel">
        <h3>Connectedness</h3>
        <ul className="degree-bars">
          {degrees.map((d) => (
            <li key={d.role}>
              <span className="degree-name"><span className={`swatch role-${d.role.replace(' ', '-')}`} aria-hidden="true" />
                {label(data, d.id)} <small>{d.role}</small></span>
              <span className="degree-track">
                <span className={`degree-bar role-${d.role.replace(' ', '-')}`}
                  style={{ width: `${(100 * d.degree) / maxDegree}%` }} />
                <span className="degree-value">{count(d.degree)}</span>
              </span>
            </li>
          ))}
        </ul>
        <p className="exhibit-note">Facts each entity takes part in, across all splits.</p>
      </section>
    </div>
  )
}

// One corruption per screen: the fact pair, its neighbourhood, the LLM's
// reasoning, what the graph measures, and what it records.

import Graph from '../graph/Graph.jsx'
import Criteria from './Criteria.jsx'
import Exhibit from './Exhibit.jsx'
import Facts from './Facts.jsx'
import { LEVEL, count, entity, label, relationLabel, relationMeaning } from '../labels.js'
import { isProbablyTrue } from '../decks.js'

function Fact({ kase, data, which }) {
  const triple = kase[which]
  const swapped = kase.slot === 'head' ? 'h' : 't'
  const role = which === 'true' ? 'true' : 'replacement'
  return (
    <h1 className="fact">
      {['h', 'r', 't'].map((slot) => (
        slot === 'r'
          ? <span key={slot} className="relation">{relationLabel(data, triple.r)}</span>
          : <span key={slot} className={`entity${slot === swapped ? ` swap swap-${role}` : ''}`}>
              {label(data, triple[slot])}
            </span>
      ))}
    </h1>
  )
}

export default function CaseCard({ data, deck, index, onPrev, onNext, onHome }) {
  const kase = deck.cases[index]
  const links = [kase.anchor, kase.true_filler, kase.candidate]
    .map((id) => ({ id, wiki: entity(data, id).wiki }))
    .filter((item) => item.wiki)

  return (
    <div className="case-shell">
      <nav className="case-nav">
        <button onClick={onHome} className="ghost">{'←'} overview</button>
        <span>{deck.title} {'·'} case {count(index + 1)} of {count(deck.cases.length)}</span>
        <span className="chips">
          <span className="chip">{kase.slot} replaced</span>
          <span className="chip">{relationLabel(data, kase.true.r)}</span>
          <span className="chip">corroboration: {LEVEL[kase.structure.corroboration].label.toLowerCase()}</span>
          {isProbablyTrue(kase) && <span className="chip chip-warn">{'⚠'} probably true</span>}
        </span>
      </nav>

      <article className="case-card">
        <div className="fact-pair">
          <div className="fact-row">
            <span className="fact-tag tag-true"><span aria-hidden="true">{'✓'}</span> true fact</span>
            <Fact kase={kase} data={data} which="true" />
          </div>
          <div className="fact-row">
            <span className="fact-tag tag-corrupted"><span aria-hidden="true">{'✗'}</span> corruption</span>
            <Fact kase={kase} data={data} which="corrupted" />
          </div>
          <p className="fact-meaning">
            {'“'}{relationLabel(data, kase.true.r)}{'”'}: {relationMeaning(data, kase.true.r)}
          </p>
          {links.length > 0 && (
            <p className="wiki-links">Look up:{' '}
              {links.map((item, i) => (
                <span key={item.id}>
                  {i > 0 && ' · '}
                  <a href={item.wiki} target="_blank" rel="noreferrer">{label(data, item.id)} {'↗'}</a>
                </span>
              ))}
            </p>
          )}
        </div>

        <section className="panel">
          <h2>The neighbourhood</h2>
          <p className="panel-sub">
            {label(data, kase.anchor)} kept its place; {label(data, kase.candidate)} replaced{' '}
            {label(data, kase.true_filler)}. The dashed edge is the corruption.
          </p>
          <Graph kase={kase} data={data} />
        </section>

        <section className="panel">
          <h2>Does it do what KGSAGE intends?</h2>
          <Criteria kase={kase} data={data} />
        </section>

        <section className="panel">
          <h2>What the graph measures</h2>
          <Exhibit kase={kase} data={data} />
        </section>

        <section className="panel">
          <h2>What the graph records</h2>
          <Facts kase={kase} data={data} />
        </section>

        <footer className="provenance">
          {data.run} {'·'} {kase.id} {'·'} {kase.true.h} {kase.true.r} {kase.true.t}
          {' → '}{kase.corrupted.h} {kase.corrupted.r} {kase.corrupted.t}
        </footer>
      </article>

      <nav className="pager">
        <button onClick={onPrev} disabled={index === 0}>{'←'} previous</button>
        <span className="pager-hint">{'←'} {'→'} to page {'·'} Esc for the overview</span>
        <button onClick={onNext} disabled={index === deck.cases.length - 1}>next {'→'}</button>
      </nav>
    </div>
  )
}

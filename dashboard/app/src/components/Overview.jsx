// The setup screen: what was made, how it was judged, and the decks. Every
// number on it is read from the loaded view model, never written here.

import { useState } from 'react'
import { DIMENSIONS } from '../decks.js'
import {
  CRITERION, GOALS, LEVEL, LEVELS, VERDICT, count, pct, plural, relationLabel,
} from '../labels.js'

const GUARD_WORDS = {
  kind: () => 'same kind',
  support: (value) => `support ≤ ${value}`,
  unique: () => 'unique',
}

export default function Overview({ data, decks, onOpenDeck }) {
  const s = data.stats
  const p = data.params
  const explained = s.explained
  const [hoverKey, setHoverKey] = useState(null)
  const hovered = decks.find((d) => d.key === hoverKey)
  const lit = hovered ? new Set(hovered.cases.map((c) => c.id)) : null
  const share = p.ratio != null ? `${Math.round(p.ratio * 1000) / 10}%` : null

  return (
    <div className="overview">
      <header>
        <p className="eyebrow">{data.dataset.name} {'·'} {data.run}</p>
        <h1>KGSAGE corruptions, explained</h1>
        <p className="lede">
          A trained KGSAGE generator took {count(s.cases)} true facts
          {share && <> ({share} of the graph{'’'}s {count(data.dataset.triples)})</>}
          {' '}and replaced one entity in each. An LLM then judged every corruption
          against the generator{'’'}s three goals {'—'} type-valid, plausible,
          uncorroborated by the neighbourhood {'—'} and checked separately whether
          it is false in the world.
        </p>
      </header>

      <div className="pipeline">
        {[
          ['corrupt', `${count(data.generation.emitted)} unique corruptions, every one false in the graph; ${plural(data.generation.nulls, 'null')} and ${plural(data.generation.repeats, 'repeat')} dropped`],
          ['describe', `each with its neighbourhood, the graph’s facts about it, and measured corroboration, kind and support`],
          ['explain', explained
            ? `${count(explained)} judged by ${data.reasoning.models.join(', ')} (prompt ${data.reasoning.prompt_versions.join(', ')})`
            : 'not yet judged: run inference/explain.py'],
        ].map(([step, text], i) => (
          <div key={step} className="pipeline-step">
            <span className="step-number">{i + 1}</span>
            <b>{step}</b>
            <small>{text}</small>
          </div>
        ))}
      </div>

      <div className="stat-row">
        <Stat label="Corruptions" value={count(s.cases)}
          sub={`head ${count(s.slot.head)} · tail ${count(s.slot.tail)}`} />
        <Stat label="Aligned with all three goals" value={count(s.verdicts.aligned)}
          sub={explained ? `${pct(s.verdicts.aligned, explained)} of ${count(explained)} judged` : 'not yet judged'} accent />
        <Stat label="No shared neighbour, by measure" value={count(s.measured.uncorroborated)}
          sub={`${pct(s.measured.uncorroborated, s.cases)} of corruptions`} />
        <Stat label="Probably true in the world" value={count(s.probably_true)}
          sub={explained ? `${pct(s.probably_true, explained)} of judged · triage for a Wikidata check` : 'not yet judged'} warn />
      </div>

      <div className="chart-row">
        <figure className="chart-card">
          <figcaption>
            <b>Verdicts</b>
            <small>three goals, aligned = all hold</small>
          </figcaption>
          <VerdictBar stats={s} />
        </figure>
        <figure className="chart-card">
          <figcaption>
            <b>What fails</b>
            <small>corruptions failing each check, of {count(explained)} judged</small>
          </figcaption>
          <FailBars stats={s} />
        </figure>
        <figure className="chart-card">
          <figcaption>
            <b>Corroboration, by measure</b>
            <small>what the graph itself says links anchor and replacement</small>
          </figcaption>
          <CorroborationBar stats={s} />
        </figure>
      </div>

      <Breakdown data={data} />

      <h2 className="decks-title">The case files</h2>
      <p className="decks-sub">
        One docket of {count(s.cases)} corruptions, opened from different
        sides{' — '}the same corruption sits in several files. Hover a file to
        light up what it holds.
      </p>
      <div className="dot-strip" aria-hidden="true">
        {data.cases.map((c) => (
          <span key={c.id} className={`dot${!lit || lit.has(c.id) ? ' on' : ''}`} />
        ))}
      </div>
      {DIMENSIONS.map(([dimension, caption]) => {
        const slice = decks.filter((d) => d.dimension === dimension)
        if (!slice.length) return null
        return (
          <div key={dimension} className="deck-dim">
            {caption && <p className="deck-dim-label">{caption}</p>}
            <div className="deck-grid">
              {slice.map((deck) => (
                <button key={deck.key}
                  className={`deck-button${deck.minor ? ' minor' : ''}${deck.tone ? ` tone-${deck.tone}` : ''}`}
                  onClick={() => onOpenDeck(deck.key)}
                  onMouseEnter={() => setHoverKey(deck.key)}
                  onMouseLeave={() => setHoverKey(null)}
                  onFocus={() => setHoverKey(deck.key)}
                  onBlur={() => setHoverKey(null)}>
                  <b>{deck.title}</b>
                  <span className="deck-count">
                    {deck.dimension === 'all' ? count(deck.cases.length)
                      : `${count(deck.cases.length)} of ${count(s.cases)}`}
                  </span>
                  {!deck.minor && <small>{deck.blurb}</small>}
                </button>
              ))}
            </div>
          </div>
        )
      })}

      <footer className="fine-print">
        Checkpoint {data.checkpoint.path} ({data.checkpoint.sha256}) {'·'} seed {p.seed}
        {' · '}guards {p.guards
          ? Object.entries(p.guards).filter(([, v]) => v !== false && v != null)
            .map(([k, v]) => (GUARD_WORDS[k] ? GUARD_WORDS[k](v) : k)).join(', ')
          : 'off'}
        {' · '}corroboration mask {p.support_max == null ? 'off (measured, not enforced)' : `on at ${p.support_max}`}
        {' · '}record {data.source}, generated {data.generated}, exported {data.exported}.
        {' '}The verdict is derived from the three goals; {'“'}false in the world{'”'} is
        triage for a Wikidata check, not a measurement.
      </footer>
    </div>
  )
}

function Stat({ label, value, sub, accent, warn }) {
  return (
    <div className={`stat${accent ? ' accent' : ''}${warn ? ' warn' : ''}`}>
      <small className="stat-label">{label}</small>
      <b>{value}</b>
      <small className="stat-sub">{sub}</small>
    </div>
  )
}

// Part-to-whole: one thin stacked bar, a 2px gap between segments, every
// segment named and counted below it -- colour never carries it alone.
function VerdictBar({ stats }) {
  const [hover, setHover] = useState(null)
  const total = stats.cases
  const segments = [
    ...Object.keys(VERDICT).map((key) => ({
      key, label: VERDICT[key].label, icon: VERDICT[key].icon, value: stats.verdicts[key] })),
    { key: 'unexplained', label: 'Not yet judged', icon: '…', value: total - stats.explained },
  ].filter((segment) => segment.value > 0)
  return (
    <div className="verdict-chart">
      <div className="stacked" role="img"
        aria-label={segments.map((g) => `${g.label} ${g.value}`).join(', ')}>
        {segments.map((segment) => (
          <span key={segment.key} className={`segment seg-${segment.key}`}
            style={{ flexGrow: segment.value }}
            onMouseEnter={() => setHover(segment.key)} onMouseLeave={() => setHover(null)} />
        ))}
      </div>
      <ul className="segment-legend">
        {segments.map((segment) => (
          <li key={segment.key} className={hover === segment.key ? 'hot' : ''}>
            <span className={`swatch seg-${segment.key}`} aria-hidden="true" />
            <span className="segment-icon" aria-hidden="true">{segment.icon}</span>
            <b>{count(segment.value)}</b> {segment.label.toLowerCase()}
            <small> {pct(segment.value, total)}</small>
          </li>
        ))}
      </ul>
    </div>
  )
}

// An ordered scale, weakest to strongest corroboration: one hue, light to
// dark, every level named and counted beside its swatch.
function CorroborationBar({ stats }) {
  const total = stats.cases
  const levels = LEVELS.map((key) => ({
    key, label: LEVEL[key].label, value: stats.measured.corroboration[key] || 0 }))
  const shown = levels.filter((level) => level.value > 0)
  return (
    <div className="verdict-chart">
      <div className="stacked" role="img"
        aria-label={levels.map((l) => `${l.label} ${l.value}`).join(', ')}>
        {shown.map((level) => (
          <span key={level.key} className={`segment lvl-${level.key}`}
            style={{ flexGrow: level.value }} title={`${level.label}: ${level.value}`} />
        ))}
      </div>
      <ul className="segment-legend">
        {levels.map((level) => (
          <li key={level.key} title={LEVEL[level.key].means}>
            <span className={`swatch lvl-${level.key}`} aria-hidden="true" />
            <b>{count(level.value)}</b> {level.label.toLowerCase()}
            <small> {pct(level.value, total)}</small>
          </li>
        ))}
      </ul>
    </div>
  )
}

// One series (cases failing a check), so one colour; values at the tips.
function FailBars({ stats }) {
  const explained = stats.explained || 0
  const rows = [
    ...GOALS.map((goal) => ({ key: goal, label: CRITERION[goal].name, value: stats.fails[goal] })),
    { key: 'false', label: 'False in the world', value: stats.probably_true, note: 'probably true' },
  ]
  const max = Math.max(1, ...rows.map((row) => row.value))
  return (
    <ul className="bars">
      {rows.map((row) => (
        <li key={row.key} title={`${row.value} of ${explained} judged`}>
          <span className="bar-label">{row.label}</span>
          <span className="bar-track">
            <span className="bar" style={{ width: `${(100 * row.value) / max}%` }} />
            <span className="bar-value">{count(row.value)} <small>{pct(row.value, explained)}</small></span>
          </span>
        </li>
      ))}
    </ul>
  )
}

// More than seven classes carry meaning, so a table: by slot, then by the
// most frequent relations.
function Breakdown({ data }) {
  const s = data.stats
  const rows = [
    ...['head', 'tail'].map((slot) => ({ key: slot, name: `${slot} replaced`, group: 'slot', ...s.by_slot[slot] })),
    ...s.relations.slice(0, 10).map((row) => ({ key: row.relation, name: relationLabel(data, row.relation), group: 'relation', ...row })),
  ]
  return (
    <figure className="chart-card breakdown">
      <figcaption>
        <b>Where it works</b>
        <small>share of judged corruptions that are aligned, and what fails, by slot and by relation</small>
      </figcaption>
      <table>
        <thead>
          <tr>
            <th scope="col"></th>
            <th scope="col" className="num">cases</th>
            <th scope="col">aligned</th>
            <th scope="col" className="num">out of character</th>
            <th scope="col" className="num">corroborated</th>
            <th scope="col" className="num">probably true</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.key} className={i === 2 ? 'group-start' : ''}>
              <th scope="row">{row.name}</th>
              <td className="num">{count(row.cases)}</td>
              <td>
                <span className="meter" aria-hidden="true">
                  <span style={{ width: row.explained ? `${(100 * row.verdicts.aligned) / row.explained}%` : 0 }} />
                </span>
                {row.explained ? pct(row.verdicts.aligned, row.explained) : '–'}
              </td>
              <td className="num">{count(row.fails.plausible)}</td>
              <td className="num">{count(row.fails.uncorroborated)}</td>
              <td className="num">{count(row.probably_true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}

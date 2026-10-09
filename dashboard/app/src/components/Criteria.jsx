// The LLM's reasoning for one corruption: the derived verdict and its
// summary, the three goals as a checklist, and the separate world check.
// Every stamp pairs an icon and a word with its colour.

import { CRITERION, GOALS, VERDICT } from '../labels.js'

function Row({ name, block }) {
  const holds = block.holds
  const isFalseCheck = name === 'false'
  const stamp = holds ? 'HOLDS' : isFalseCheck ? 'PROBABLY TRUE' : 'FAILS'
  return (
    <li className={`criterion ${holds ? 'holds' : 'fails'}${isFalseCheck ? ' world' : ''}`}>
      <span className={`stamp ${holds ? 'stamp-holds' : isFalseCheck ? 'stamp-true' : 'stamp-fails'}`}>
        <span aria-hidden="true">{holds ? '✓' : isFalseCheck ? '⚠' : '✗'}</span> {stamp}
      </span>
      <div className="criterion-body">
        <p className="criterion-name">
          <b>{CRITERION[name].name}</b>
          <small> {'—'} {CRITERION[name].means}</small>
          {isFalseCheck && <span className={`confidence c-${block.confidence}`}>{block.confidence} confidence</span>}
        </p>
        <p className="why">{block.why}</p>
      </div>
    </li>
  )
}

export default function Criteria({ kase, data }) {
  const reasoning = kase.reasoning
  if (!reasoning) {
    return (
      <div className="empty-state">
        <b>Not yet explained.</b> Run{' '}
        <code>python inference/explain.py --run {data.run}</code>, then export
        the dashboard again.
      </div>
    )
  }
  const verdict = VERDICT[reasoning.verdict]
  return (
    <div className="criteria-panel">
      <div className={`verdict verdict-${reasoning.verdict}`}>
        <span className="verdict-badge">
          <span aria-hidden="true">{verdict.icon}</span> {verdict.label}
        </span>
        <span className="verdict-means">{verdict.means}</span>
        <p className="summary">{reasoning.summary}</p>
      </div>
      <ol className="criteria">
        {GOALS.map((goal) => <Row key={goal} name={goal} block={reasoning.criteria[goal]} />)}
      </ol>
      <p className="separate-label">A separate check, never part of the verdict</p>
      <ol className="criteria">
        <Row name="false" block={reasoning.criteria.false} />
      </ol>
      <p className="judged-by">
        Judged by {reasoning.model} {'·'} prompt {reasoning.prompt_version} {'·'} {reasoning.at}
      </p>
    </div>
  )
}

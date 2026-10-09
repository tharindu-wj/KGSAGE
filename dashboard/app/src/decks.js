// Decks: pure filters over the exported cases. They overlap by design --
// one corruption sits in several -- and each carries a `dimension` so the
// overview can group them and count them as "n of <total>". Nothing here
// computes a graph statistic; it only groups fields export.py established.

import { relationLabel } from './labels.js'

export const verdictOf = (c) => (c.reasoning ? c.reasoning.verdict : null)
export const isProbablyTrue = (c) => Boolean(c.reasoning) && !c.reasoning.criteria.false.holds
const failsGoal = (c, goal) => Boolean(c.reasoning) && c.reasoning.fails.includes(goal)

export const DIMENSIONS = [
  ['all', null],
  ['verdict', 'by the LLM’s verdict on KGSAGE’s three goals'],
  ['criterion', 'by what went wrong'],
  ['measure', 'by what the graph measures'],
  ['slot', 'by which slot was replaced'],
  ['relation', 'by relation, the ten most frequent'],
]

export function buildDecks(data) {
  const cases = data.cases
  const decks = [
    { key: 'all', dimension: 'all', title: 'Every corruption',
      blurb: 'The whole run, in the order the generator made it.', cases },
    { key: 'aligned', dimension: 'verdict', tone: 'good', title: 'Aligned',
      blurb: 'Type-valid, plausible and uncorroborated: what KGSAGE is for.',
      cases: cases.filter((c) => verdictOf(c) === 'aligned') },
    { key: 'partial', dimension: 'verdict', tone: 'warning', title: 'Partial',
      blurb: 'Misses exactly one of the three goals.',
      cases: cases.filter((c) => verdictOf(c) === 'partial') },
    { key: 'misaligned', dimension: 'verdict', tone: 'critical', title: 'Misaligned',
      blurb: 'Misses two or more of the goals.',
      cases: cases.filter((c) => verdictOf(c) === 'misaligned') },
    { key: 'unexplained', dimension: 'verdict', title: 'Not yet explained',
      blurb: 'No LLM reasoning yet: run inference/explain.py, then export again.',
      cases: cases.filter((c) => !c.reasoning) },
    { key: 'probably-true', dimension: 'criterion', tone: 'warning', title: 'Probably true in the world',
      blurb: 'The LLM believes the corrupted fact is real. Check these against Wikidata first.',
      cases: cases.filter(isProbablyTrue) },
    { key: 'out-of-character', dimension: 'criterion', title: 'Out of character',
      blurb: 'Fails plausible: absurd, anachronistic, or at odds with what the entity is.',
      cases: cases.filter((c) => failsGoal(c, 'plausible')) },
    { key: 'corroborated', dimension: 'criterion', title: 'Corroborated after all',
      blurb: 'Fails uncorroborated: the graph’s own facts point at the replacement.',
      cases: cases.filter((c) => failsGoal(c, 'uncorroborated')) },
    { key: 'wrong-kind', dimension: 'criterion', title: 'Wrong kind',
      blurb: 'Fails type-valid: not the kind of thing this slot holds.',
      cases: cases.filter((c) => failsGoal(c, 'type_valid')) },
    { key: 'direct', dimension: 'measure', title: 'Directly linked',
      blurb: 'The graph already links the anchor to the replacement by another relation.',
      cases: cases.filter((c) => c.structure.corroboration === 'direct') },
    { key: 'chain', dimension: 'measure', title: 'A chain of the same relation',
      blurb: 'The corrupted relation runs twice through one entity: the graph predicts the fact.',
      cases: cases.filter((c) => c.structure.corroboration === 'chain') },
    { key: 'notable', dimension: 'measure', title: 'Notable sharing',
      blurb: 'More shared neighbours than the slot’s alternatives usually have.',
      cases: cases.filter((c) => c.structure.corroboration === 'notable') },
    { key: 'ordinary', dimension: 'measure', title: 'Ordinary sharing',
      blurb: 'Shared neighbours, but no more than is usual for the slot.',
      cases: cases.filter((c) => c.structure.corroboration === 'ordinary') },
    { key: 'no-shared', dimension: 'measure', title: 'No shared neighbour',
      blurb: 'Nothing links to both the anchor and the replacement.',
      cases: cases.filter((c) => c.structure.corroboration === 'none') },
    { key: 'head', dimension: 'slot', title: 'Head replaced',
      blurb: 'The anchor is the tail, often a hub such as a country or an occupation.',
      cases: cases.filter((c) => c.slot === 'head') },
    { key: 'tail', dimension: 'slot', title: 'Tail replaced',
      blurb: 'The anchor is the subject, whose own facts judge the replacement.',
      cases: cases.filter((c) => c.slot === 'tail') },
  ]
  for (const relation of data.stats.top_relations) {
    decks.push({ key: `rel:${relation}`, dimension: 'relation', minor: true,
      title: relationLabel(data, relation), blurb: '',
      cases: cases.filter((c) => c.true.r === relation) })
  }
  return decks.filter((deck) => deck.cases.length > 0)
}

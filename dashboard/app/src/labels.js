// Ids -> words, from the tables the exporter put in the view model. The
// app never opens a dataset file; everything it can say about an entity
// came out of dashboard/export.py.

export const entity = (data, id) => data.entities[id] || { label: id, kinds: [], degree: 0 }
export const label = (data, id) => entity(data, id).label || id
export const relationLabel = (data, id) => (data.relations[id] || {}).label || id
export const relationMeaning = (data, id) => (data.relations[id] || {}).description || ''

export const ROLE_NAME = {
  anchor: 'anchor',
  true_filler: 'true value',
  candidate: 'replacement',
  shared: 'shared neighbour',
}

export const GOALS = ['type_valid', 'plausible', 'uncorroborated']

export const CRITERION = {
  type_valid: {
    name: 'Type-valid',
    means: 'the replacement is the kind of thing this slot of the relation holds',
  },
  plausible: {
    name: 'Plausible',
    means: 'in character: the corrupted fact fits what the entities are',
  },
  uncorroborated: {
    name: 'Uncorroborated',
    means: "nothing the graph records about the anchor points to the replacement",
  },
  false: {
    name: 'False in the world',
    means: 'by world knowledge, the corrupted fact is actually false',
  },
}

export const VERDICT = {
  aligned: { label: 'Aligned', icon: '✓', means: 'meets all three goals' },
  partial: { label: 'Partial', icon: '◐', means: 'misses one goal' },
  misaligned: { label: 'Misaligned', icon: '✗', means: 'misses two or more goals' },
}

// The measured corroboration, weakest to strongest (inference/
// neighbourhood.py, Graph.corroboration). An ordered scale: it wears one
// hue, light to dark.
export const LEVELS = ['none', 'ordinary', 'notable', 'chain', 'direct']
export const LEVEL = {
  none: { label: 'None', means: 'nothing links to both the anchor and the replacement' },
  ordinary: { label: 'Ordinary', means: 'they share neighbours, but no more than the slot’s alternatives usually do' },
  notable: { label: 'Notable', means: 'they share more neighbours than the slot’s alternatives usually do' },
  chain: { label: 'Chain', means: 'the corrupted relation runs twice through one entity, so the graph predicts the fact' },
  direct: { label: 'Direct', means: 'the graph already links the two by another relation' },
}

export const pct = (n, d) => (d ? `${Math.round((100 * n) / d)}%` : '–')
export const count = (n) => Number(n || 0).toLocaleString('en')
export const plural = (n, word, many = `${word}s`) => `${count(n)} ${n === 1 ? word : many}`

// The support measurement in words, phrased for the corrupted slot -- the
// same sentence the LLM was shown (inference/prompts.py support_sentence).
export function supportSentence(kase, data) {
  const support = kase.structure.support
  if (!support) return null
  const name = relationLabel(data, kase.true.r)
  const value = label(data, support.value)
  const candidate = label(data, kase.candidate)
  const share = `${Math.round(100 * support.share)}%`
  if (kase.slot === 'tail') {
    return `Of the ${support.holders} entities whose “${name}” includes ${value}, ${share} also include ${candidate}.`
  }
  return `Of the ${support.holders} entities ${value} links to by “${name}”, ${share} are also linked from ${candidate}.`
}

export function tripleText(data, [h, r, t]) {
  return `${label(data, h)} —${relationLabel(data, r)}→ ${label(data, t)}`
}

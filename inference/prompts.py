"""What the LLM is told, and the shape its answer must take. Versioned.

Every case explain.py writes carries PROMPT_VERSION, so a record can say
which wording produced which reasoning. Change the instruction, the case
block or the schema below and the version must change with it -- a
reasoning block judged under one wording is not comparable with one
judged under another.

The model sees only what the run record holds: the case's two triples,
the three entities' labels, kinds and descriptions, the graph's own facts
about the anchor and the replacement, and the structure corrupt.py
measured. It is never told the generator's scores, and it never decides
the verdict: explain.py derives that from the three goals, so the verdict
cannot disagree with the criteria it summarises.
"""

#: explain/1 (9 Oct 2026) let a borderline "why" argue one way and "holds"
#: say the other; explain/2 made holds follow the sentence and defined
#: plausible as "in character". explain/3 (10 Oct 2026): under explain/2
#: "uncorroborated" never failed in 990 cases -- the model read it as "no
#: direct link", which is true of every corruption by construction. The
#: case block now shows the paths through shared neighbours and any direct
#: link by another relation, and the criterion is judged on those paths.
#: explain/4 (10 Oct 2026, the same day, after an 18-case probe): explain/3
#: failed every direct link but still waved two-step paths away because
#: "they do not directly link"; the rule now forbids arguing from the
#: missing link and says outright that a specific two-step path counts.
#: explain/5 (10 Oct 2026, after a 28-case probe of explain/4): the swing
#: went too far -- any non-hub path failed the criterion, including the
#: diplomatic triangles most country pairs share. The case block now gives
#: a baseline (how many neighbours the slot's typical alternative shares
#: with the anchor), and sharing no more than typical is not support.
#: explain/6 (10 Oct 2026, after the explain/5 probe): the model was still
#: inconsistent at the threshold (a direct "continent" link waved through,
#: 48% called strong). The block now states the measured corroboration as
#: one word -- DIRECT, NOTABLE, ORDINARY, NONE -- and the rule keys on it.
#: explain/7 (10 Oct 2026, after the explain/6 probe: direct 5/5 failed,
#: ordinary and none 14/14 held, notable mostly overridden on semantic
#: grounds, which is the LLM's job, but an influenced-by chain through
#: Pascal was still dismissed as "not direct"): a chain that repeats the
#: corrupted relation through a specific entity always counts as support.
#: explain/8 (10 Oct 2026): explain/7 still called the Pascal chain "not
#: direct" -- no wording moved this model -- so the chain is now measured
#: (neighbourhood.Graph.chains) and stated as a level of its own, CHAIN,
#: which the rule treats like DIRECT.
PROMPT_VERSION = "explain/8"

#: KGSAGE's three goals, in the paper's order. A corruption that meets
#: all three is "aligned".
GOALS = ("type_valid", "plausible", "uncorroborated")

#: the goals plus the separate world-knowledge check. "false" is triage
#: for a Wikidata verification, never part of the verdict.
CRITERIA = GOALS + ("false",)

#: what a run's dataset is, in one line the model can use. A dataset not
#: listed here is introduced by its name alone.
DATASET_NOTES = {
    "codex-s": ("CoDEx-S, an encyclopedic knowledge graph drawn from "
                "Wikidata about notable people, organisations and places"),
}

INSTRUCTION = """\
You review corruptions made by KGSAGE, a generator of false-but-plausible
knowledge-graph facts used to test anomaly detectors.

A corruption takes a TRUE fact (head --relation-- tail) from the graph and
replaces exactly one entity, the head or the tail, with another entity: the
REPLACEMENT. The relation never changes. The entity that kept its place is
the ANCHOR. KGSAGE aims for corruptions with three properties, its goals.
Judge each case against them, then make a fourth, separate check:

1. type_valid: the replacement is the right KIND of thing for this slot of
   this relation -- a language where a language belongs, a person where a
   person belongs. Judge by what the relation means and by the kinds shown.
   This is not about truth.
2. plausible: the corrupted fact is in character -- it fits what the
   entities are, as their kinds, descriptions and recorded facts show, so a
   reader could believe it. Untrue but in character holds (a novelist listed
   as a playwright). Out of character, anachronistic or absurd does not hold
   (a chemist listed as a singer-songwriter; an empire as a member of an
   organisation founded after it ended). Decide this without what you know
   of the real entities' lives: whether it IS true is the fourth check.
3. uncorroborated: the anchor's NEIGHBOURHOOD in the graph gives no support
   to the replacement. Judge only what is listed under "measured in the
   graph", starting from its measured corroboration. The corrupted fact
   itself is never in the graph, so never argue from its absence: "they are
   not directly linked" decides nothing.
     DIRECT   -- the graph already links the two by another relation: it
                 does NOT hold.
     CHAIN    -- the corrupted relation runs twice through one entity (the
                 anchor --influenced by-- Pascal --influenced by-- the
                 replacement), so the graph itself predicts the corrupted
                 fact: it does NOT hold.
     NOTABLE  -- they share more neighbours than the slot's alternatives
                 usually do: it does NOT hold, unless every listed path runs
                 through a generic entity (a continent, a country, a broad
                 field, an organisation most countries join) or none of the
                 paths bears on the corrupted relation.
     ORDINARY -- they share no more than is usual for the slot: it HOLDS,
                 unless a listed path makes the corrupted fact expected.
     NONE     -- nothing links to both: it HOLDS.
   Name what decides it, and never use "they are not directly linked" as a
   reason.
4. false: by your own knowledge of the world, the corrupted fact is actually
   false. If it is true or probably true (the graph may simply be missing
   it), it does not hold. Give your confidence: high, medium or low.

For every criterion write "why" first: one sentence of at most 25 words,
grounded in this case's entities and facts, that reaches a decision. Then
"holds", which must agree with that sentence -- true when the sentence finds
the criterion met, false when it finds it not met. Do not restate the
triple; say what decides the criterion.

End each case with a "summary" of at most 30 words: which of the three goals
the corruption meets or misses, and whether it is actually false.

Return a JSON array with one object per case, in the order given, each
carrying the case's id.
"""


def _criterion(*extra):
    """One criterion's object: the reason first, then the decision."""
    properties = {"why": {"type": "STRING"}, "holds": {"type": "BOOLEAN"}}
    order = ["why", "holds"]
    for name, schema in extra:
        properties[name] = schema
        order.append(name)
    return {"type": "OBJECT", "properties": properties,
            "required": list(order), "propertyOrdering": order}


#: the structured-output schema (google-genai response_schema). "why"
#: precedes "holds" on purpose: the model reasons, then decides.
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "type_valid": _criterion(),
            "plausible": _criterion(),
            "uncorroborated": _criterion(),
            "false": _criterion(("confidence", {
                "type": "STRING", "enum": ["high", "medium", "low"]})),
            "summary": {"type": "STRING"},
        },
        "required": ["id", *CRITERIA, "summary"],
        "propertyOrdering": ["id", *CRITERIA, "summary"],
    },
}


def verdict_of(criteria):
    """aligned (all three goals hold), partial (one fails) or misaligned
    (two or more fail) -- derived, never asked for."""
    fails = [goal for goal in GOALS if not criteria[goal]["holds"]]
    return ("aligned" if not fails else
            "partial" if len(fails) == 1 else "misaligned"), fails


def case_block(case, record):
    """One case as the model reads it: the facts, then the measurements."""
    entities, relations = record["entities"], record["relations"]

    def label(entity):
        return (entities.get(entity) or {}).get("label", entity)

    def kinds(entity):
        shown = (entities.get(entity) or {}).get("kinds", [])[:4]
        return ", ".join(shown) or "no kind recorded"

    def described(role, entity):
        description = (entities.get(entity) or {}).get("description", "")
        tail = f": {description}" if description else ""
        return f"  {role:11s} {label(entity)} [{kinds(entity)}]{tail}"

    def said(triple):
        head, relation, tail = triple
        name = (relations.get(relation) or {}).get("label", relation)
        return f"{label(head)} --{name}-- {label(tail)}"

    def fact(triple):
        return f"    - {said(triple)}"

    relation = case["true"]["r"]
    name = (relations.get(relation) or {}).get("label", relation)
    meaning = (relations.get(relation) or {}).get("description", "")
    anchor, candidate = case["anchor"], case["candidate"]
    structure = case["structure"]
    lines = [
        f"CASE {case['id']}: the {case['slot'].upper()} was replaced; "
        f"the anchor is {label(anchor)}",
        f"  true fact:      {case['text']['true']}",
        f"  corrupted fact: {case['text']['corrupted']}",
        f'  relation "{name}"' + (f": {meaning}" if meaning else ""),
        described("anchor", anchor),
        described("true value", case["true_filler"]),
        described("replacement", candidate),
    ]
    anchor_facts = case["facts"]["anchor"]
    lines.append(f"  the graph's facts about {label(anchor)} "
                 f"({len(anchor_facts)} shown, \"{name}\" first):")
    lines.extend(fact(triple) for triple in anchor_facts)
    candidate_facts = case["facts"]["candidate"]
    if candidate_facts:
        lines.append(f"  the graph's facts about {label(candidate)} "
                     f"({len(candidate_facts)} shown):")
        lines.extend(fact(triple) for triple in candidate_facts)

    lines.append("  measured in the graph:")
    shared = structure["shared_neighbours"]
    lines.append(f"    - measured corroboration: {structure['corroboration'].upper()}")
    if structure["direct_facts"]:
        lines.append("    - the graph already links anchor and replacement directly: "
                     + "; ".join(said(triple) for triple in structure["direct_facts"]))
    else:
        lines.append("    - no direct link between anchor and replacement by any relation")
    if shared:
        lines.append(f"    - {shared} shared neighbour{'' if shared == 1 else 's'} "
                     f"(entities linked to both); the least-connected, with the "
                     f"paths through them:")
        chains = set(structure.get("chains", []))
        for bridge in structure["bridges"]:
            via = bridge["via"]
            degree = (entities.get(via) or {}).get("degree", 0)
            sides = "; ".join(said(triple) for triple
                              in bridge["anchor_side"] + bridge["candidate_side"])
            tag = " [a chain of the corrupted relation]" if via in chains else ""
            lines.append(f"        via {label(via)} ({degree} facts in the graph)"
                         f"{tag}: {sides}")
    else:
        lines.append("    - no shared neighbour: nothing links to both the anchor "
                     "and the replacement")
    baseline = structure.get("shared_baseline")
    if baseline:
        typical = baseline["typical"]
        typical = int(typical) if float(typical).is_integer() else typical
        lines.append(
            f"    - compared with the slot: the typical one of the "
            f"{baseline['compared']} other entities that fill this slot shares "
            f"{typical} neighbour{'' if typical == 1 else 's'} with the anchor; "
            f"{baseline['as_many']:.0%} of them share {shared} or more")
    usual = ", ".join(f"{kind} x{count}"
                      for kind, count in structure["slot_usual_kinds"])
    if usual:
        peers = structure["kind_peers"]
        lines.append(
            f"    - this slot of \"{name}\" is usually held by: {usual}"
            + (f"; {peers} of its {structure['slot_occupants']} occupants "
               f"share a kind with the replacement" if peers is not None else ""))
    support = support_sentence(case, record)
    if support:
        lines.append(f"    - {support}")
    return "\n".join(lines)


def support_sentence(case, record):
    """The support measurement in words, phrased for the corrupted slot.
    None when no anchor value is held widely enough to judge by."""
    support = case["structure"]["support"]
    if not support:
        return None
    entities, relations = record["entities"], record["relations"]

    def label(entity):
        return (entities.get(entity) or {}).get("label", entity)

    relation = case["true"]["r"]
    name = (relations.get(relation) or {}).get("label", relation)
    value, candidate = label(support["value"]), label(case["candidate"])
    if case["slot"] == "tail":
        return (f"of the {support['holders']} entities whose \"{name}\" "
                f"includes {value}, {support['share']:.0%} also include "
                f"{candidate}")
    return (f"of the {support['holders']} entities {value} links to by "
            f"\"{name}\", {support['share']:.0%} are also linked from "
            f"{candidate}")


def batch_prompt(cases, record):
    """The user turn for one call: the dataset line, then the case blocks."""
    dataset = record["dataset"]["name"]
    note = DATASET_NOTES.get(dataset, f"the knowledge graph '{dataset}'")
    blocks = "\n\n".join(case_block(case, record) for case in cases)
    return (f"Dataset: {note}.\n"
            f"Judge these {len(cases)} corruption"
            f"{'' if len(cases) == 1 else 's'}.\n\n{blocks}\n")

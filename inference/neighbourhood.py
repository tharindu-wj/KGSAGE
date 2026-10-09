"""The dataset's graph, indexed once, and the questions asked of it per case.

Every number here is recomputed from the dataset's own splits, never read
back from the generator, so a record's measurements can be reproduced from
the TSVs alone -- by anyone, without the checkpoint. Two of them mirror the
package's guards on purpose (same definition, recomputed, not imported --
the package keeps those helpers private):

    support     max over the anchor's true values v of P(candidate | v):
                the share of the entities holding v in this slot that
                also hold the candidate. The SUPPORT guard bans > 0.5.
    kind_match  the candidate shares a kind with the value it replaced:
                CoDEx's own criterion for a hard negative, and the KIND
                guard.

The rest measure corroboration -- what the anchor's surroundings say
about the candidate -- and they are MEASURED, never enforced: the
corroboration mask stays off by default (docs/KGSAGE_COLUMN.md, option C).

Iteration order is fixed everywhere (lists and insertion-ordered dicts,
never a set walked in hash order), so the same seed writes the same
record in any process.
"""
import random
from collections import Counter, defaultdict, deque

#: shared neighbours a case's subgraph always keeps, least-connected
#: first: a small shared neighbour is stronger corroboration than a
#: country hub every entity touches
SHARED_KEPT = 3
#: shared neighbours named in a case's structure
SHARED_LISTED = 5
#: an anchor value held by fewer entities than this is too thin to
#: support anything -- the support guard's own threshold (min_holders)
MIN_HOLDERS = 5
#: kinds listed for a slot's usual occupants
USUAL_KINDS = 3
#: sharing is NOTABLE when the replacement shares more neighbours with the
#: anchor than the slot's typical alternative AND no more than this share
#: of the alternatives match it -- the top fifth of the slot
NOTABLE_SHARE = 0.2
#: a CHAIN (anchor --r-- X --r-- replacement) counts only for a directed
#: relation: above this share of facts recorded both ways ("diplomatic
#: relation"), every pair shares such a triangle and it supports nothing
CHAIN_MAX_RECIPROCITY = 0.25

#: the three entities a case is about, in tie-break order
FOCUS = ("anchor", "true_filler", "candidate")


class Graph:
    """The all-splits graph of one dataset, as string triples."""

    def __init__(self, triples, context=None):
        self.triples = list(triples)
        self.context = context
        self.adjacency = defaultdict(list)  # node -> [(other, relation, outgoing)]
        self.neighbours = defaultdict(set)  # node -> undirected neighbours
        self.incident = defaultdict(list)   # node -> triple indices
        self.degree = Counter()             # node -> incident triples
        self.tails_of = defaultdict(list)   # (head, relation) -> tails
        self.heads_of = defaultdict(list)   # (relation, tail) -> heads
        self.by_relation = defaultdict(list)
        #: (relation, slot) -> occupants, insertion-ordered (a dict, not a
        #: set, so counting over it is deterministic)
        self.occupants = defaultdict(dict)
        for index, (head, relation, tail) in enumerate(self.triples):
            self.adjacency[head].append((tail, relation, True))
            self.adjacency[tail].append((head, relation, False))
            self.neighbours[head].add(tail)
            self.neighbours[tail].add(head)
            self.incident[head].append(index)
            if tail != head:
                self.incident[tail].append(index)
            self.degree[head] += 1
            self.degree[tail] += 1
            self.tails_of[(head, relation)].append(tail)
            self.heads_of[(relation, tail)].append(head)
            self.by_relation[relation].append((head, relation, tail))
            self.occupants[(relation, "head")][head] = None
            self.occupants[(relation, "tail")][tail] = None
        self.triple_set = set(self.triples)
        #: relation -> share of its facts whose reverse is also a fact
        self.reciprocity = {
            relation: sum(1 for h, r, t in facts if (t, r, h) in self.triple_set)
            / len(facts)
            for relation, facts in self.by_relation.items()}
        self._cooccurrence_cache = {}
        self._usual_cache = {}
        self._peers_cache = {}
        # Neighbour sets as integer bitsets, for the corroboration baseline:
        # shared-neighbour counts against a slot's every occupant are one
        # AND and one popcount each, fast enough in plain Python.
        self._bit = {}
        for head, _, tail in self.triples:
            for node in (head, tail):
                if node not in self._bit:
                    self._bit[node] = 1 << len(self._bit)
        self._mask = {node: sum(self._bit[other] for other in others)
                      for node, others in self.neighbours.items()}

    # ---- kinds -------------------------------------------------------------

    def kind_ids(self, entity):
        return self.context.kind_ids(entity) if self.context else []

    def kind_label(self, kind):
        return self.context.kind_label(kind) if self.context else kind

    def usual_kinds(self, relation, slot):
        """The most common kinds among this slot's occupants, with counts."""
        key = (relation, slot)
        if key not in self._usual_cache:
            counts = Counter()
            for occupant in self.occupants.get(key, {}):
                for kind in self.kind_ids(occupant):
                    counts[kind] += 1
            self._usual_cache[key] = [[self.kind_label(kind), count]
                                      for kind, count
                                      in counts.most_common(USUAL_KINDS)]
        return self._usual_cache[key]

    def kind_peers(self, relation, slot, candidate):
        """How many OTHER occupants of this slot share a kind with the
        candidate. Zero is the odd-types argument: nothing of its kind has
        ever sat here. None when the candidate has no kinds to compare."""
        kinds = frozenset(self.kind_ids(candidate))
        if not kinds:
            return None
        key = (relation, slot, kinds)
        if key not in self._peers_cache:
            self._peers_cache[key] = [
                occupant for occupant in self.occupants.get((relation, slot), {})
                if kinds & set(self.kind_ids(occupant))]
        return sum(1 for occupant in self._peers_cache[key]
                   if occupant != candidate)

    # ---- values and support ------------------------------------------------

    def values_of(self, anchor, relation, slot):
        """The anchor's true values in the corrupted slot: its tails under
        the relation for a tail corruption, the heads that point at it for
        a head corruption."""
        if slot == "tail":
            return list(self.tails_of.get((anchor, relation), []))
        return list(self.heads_of.get((relation, anchor), []))

    def _cooccurrence(self, relation, slot):
        """holders[v] = entities holding v in this slot; cooc[v][u] = how
        many of those also hold u. The package's _cooccurrence, rebuilt."""
        key = (relation, slot)
        if key in self._cooccurrence_cache:
            return self._cooccurrence_cache[key]
        groups = {}
        for head, _, tail in self.by_relation.get(relation, []):
            if slot == "tail":
                groups.setdefault(head, []).append(tail)
            else:
                groups.setdefault(tail, []).append(head)
        holders, cooc = {}, {}
        for held in groups.values():
            for value in held:
                holders[value] = holders.get(value, 0) + 1
                row = cooc.setdefault(value, {})
                for other in held:
                    if other != value:
                        row[other] = row.get(other, 0) + 1
        self._cooccurrence_cache[key] = (holders, cooc)
        return holders, cooc

    def support(self, anchor, relation, slot, candidate):
        """How strongly the graph's own regularities PREDICT the candidate
        for this anchor: the best P(candidate | v) over the anchor's values
        v held by at least MIN_HOLDERS entities. None when no value is
        thick enough to judge by."""
        holders, cooc = self._cooccurrence(relation, slot)
        best = None
        for value in self.values_of(anchor, relation, slot):
            count = holders.get(value, 0)
            if count < MIN_HOLDERS:
                continue
            together = cooc.get(value, {}).get(candidate, 0)
            if best is None or together / count > best[0]:
                best = (together / count, value, count, together)
        if best is None:
            return None
        share, value, count, together = best
        return {"share": round(share, 4), "value": value,
                "holders": count, "together": together}

    # ---- corroboration ------------------------------------------------------

    def shared_neighbours(self, anchor, candidate):
        """Entities linked to both, least-connected first."""
        shared = ((self.neighbours.get(anchor, set())
                   & self.neighbours.get(candidate, set()))
                  - {anchor, candidate})
        return sorted(shared, key=lambda entity: (self.degree[entity], entity))

    def chains(self, anchor, relation, slot, candidate, shared):
        """Shared neighbours through which the corrupted relation runs twice
        in the corrupted fact's own direction -- anchor --r--> X --r-->
        replacement for a tail corruption, replacement --r--> X --r-->
        anchor for a head one. Such a chain makes the corrupted fact
        expected, so it is support however the counts look; for a relation
        recorded both ways it is a mere triangle and is ignored."""
        if self.reciprocity.get(relation, 0) > CHAIN_MAX_RECIPROCITY:
            return []
        start, end = (anchor, candidate) if slot == "tail" else (candidate, anchor)
        return [via for via in shared
                if (start, relation, via) in self.triple_set
                and (via, relation, end) in self.triple_set]

    def ranked_shared(self, anchor, relation, slot, candidate):
        """Shared neighbours in the order a case shows them: chains first,
        then the least-connected."""
        shared = self.shared_neighbours(anchor, candidate)
        chained = self.chains(anchor, relation, slot, candidate, shared)
        return chained + [via for via in shared if via not in chained], chained

    def facts_between(self, a, b, limit=2):
        """The graph's own facts that link two entities, either direction."""
        found = []
        for index in self.incident.get(a, []):
            head, relation, tail = self.triples[index]
            if {head, tail} == {a, b}:
                found.append([head, relation, tail])
                if len(found) == limit:
                    break
        return found

    def bridges(self, anchor, candidate, shared):
        """For the least-connected shared neighbours, the facts on each side:
        how the anchor actually reaches the replacement. A shared neighbour
        is only as strong as the path through it, and the path is what a
        judge needs to see -- the count alone says nothing."""
        return [{"via": via,
                 "anchor_side": self.facts_between(anchor, via),
                 "candidate_side": self.facts_between(via, candidate)}
                for via in shared[:SHARED_KEPT]]

    def corroboration_baseline(self, anchor, relation, slot, candidate):
        """How unusual the replacement's sharing is: the shared-neighbour
        count between the anchor and every OTHER entity that fills this
        slot (not the anchor's own true values, not the replacement), as
        the typical count and the share that match or beat the
        replacement's. Countries share partners, people share fields; a
        count is support only if it stands out from the slot's own."""
        values = set(self.values_of(anchor, relation, slot))
        anchor_mask = self._mask.get(anchor, 0)
        own = len(self.shared_neighbours(anchor, candidate))
        counts = []
        for other in self.occupants.get((relation, slot), {}):
            if other in values or other in (anchor, candidate):
                continue
            both = (anchor_mask & self._mask.get(other, 0)
                    & ~(self._bit[anchor] | self._bit[other]))
            counts.append(both.bit_count())
        if not counts:
            return None
        counts.sort()
        middle = len(counts) // 2
        typical = (counts[middle] if len(counts) % 2
                   else (counts[middle - 1] + counts[middle]) / 2)
        return {"typical": typical,
                "as_many": round(sum(1 for c in counts if c >= own) / len(counts), 4),
                "compared": len(counts)}

    @staticmethod
    def corroboration(direct, shared, baseline, chained=()):
        """The measured corroboration in one word, strongest first:
            direct    the graph already links anchor and replacement
            chain     the corrupted relation runs twice through a shared
                      neighbour (chains), on a directed relation
            notable   shared neighbours above the slot's typical count,
                      in its top fifth (NOTABLE_SHARE)
            ordinary  shared neighbours, but no more than the slot's
                      alternatives usually have
            none      no shared neighbour and no link
        """
        if direct:
            return "direct"
        if chained:
            return "chain"
        if not shared:
            return "none"
        if (baseline and shared > baseline["typical"]
                and baseline["as_many"] <= NOTABLE_SHARE):
            return "notable"
        return "ordinary"

    def structure(self, anchor, relation, slot, true_filler, candidate):
        """The measured facts about one corruption (CONTRACTS.md)."""
        shared, chained = self.ranked_shared(anchor, relation, slot, candidate)
        direct = candidate in self.neighbours.get(anchor, set())
        baseline = self.corroboration_baseline(anchor, relation, slot, candidate)
        candidate_kinds = set(self.kind_ids(candidate))
        filler_kinds = set(self.kind_ids(true_filler))
        occupants = self.occupants.get((relation, slot), {})
        return {
            "candidate_in_pool": candidate in occupants,
            "kind_match": (bool(candidate_kinds & filler_kinds)
                           if candidate_kinds and filler_kinds else None),
            "kind_peers": self.kind_peers(relation, slot, candidate),
            "slot_occupants": len(occupants),
            "slot_usual_kinds": self.usual_kinds(relation, slot),
            "shared_neighbours": len(shared),
            "shared_sample": shared[:SHARED_LISTED],
            "shared_baseline": baseline,
            "chains": chained,
            "corroboration": self.corroboration(direct, len(shared), baseline,
                                                chained),
            "bridges": self.bridges(anchor, candidate, shared),
            "direct_edge": direct,
            "direct_facts": (self.facts_between(anchor, candidate, limit=3)
                             if direct else []),
            "candidate_hop": 1 if direct else (2 if shared else None),
            "uncorroborated": not direct and not shared,
            "support": self.support(anchor, relation, slot, candidate),
            "anchor_values": self.values_of(anchor, relation, slot),
        }

    # ---- facts and the drawable neighbourhood ------------------------------

    def facts_about(self, entity, relation, limit):
        """The entity's own triples, the corrupted relation's first, then
        the rest in dataset order -- what an LLM is shown about it."""
        first, rest = [], []
        for index in self.incident.get(entity, []):
            triple = self.triples[index]
            (first if triple[1] == relation else rest).append(list(triple))
        return (first + rest)[:limit]

    def ego(self, centre, hops, max_neighbours, rng, keep, outer_neighbours=None):
        """Breadth-first out to `hops`. The centre expands at most
        `max_neighbours` context entries, chosen at random, and every node
        further out at most `outer_neighbours` (context of context needs
        less room); entries that reach a `keep` node always come too, ON
        TOP of the cap. The walk of the retired cli/ego_from_csv.py with
        two changes: there the kept entries counted against the cap, which
        was harmless for its three kept entities but, once shared
        neighbours are kept as well, filled the cap on a well-connected
        anchor and left no context at all; and there one cap served every
        depth."""
        if outer_neighbours is None:
            outer_neighbours = max_neighbours
        hop = {centre: 0}
        edges = []
        queue = deque([centre])
        while queue:
            node = queue.popleft()
            depth = hop[node]
            if depth >= hops:
                continue
            entries = self.adjacency.get(node, [])
            kept = [entry for entry in entries if entry[0] in keep]
            rest = [entry for entry in entries if entry[0] not in keep]
            cap = max_neighbours if depth == 0 else outer_neighbours
            if len(rest) > cap:
                rng.shuffle(rest)
                rest = rest[:cap]
            entries = kept + rest
            for other, relation, outgoing in entries:
                edges.append((node, relation, other) if outgoing
                             else (other, relation, node))
                if other not in hop:
                    hop[other] = depth + 1
                    queue.append(other)
        return hop, edges

    def neighbourhood(self, anchor, true_filler, candidate, true_triple,
                      corrupted_triple, hops, candidate_hops, max_neighbours,
                      seed, outer_neighbours=None):
        """The subgraph a case is drawn from: the anchor's and the true
        value's ego to `hops`, the candidate's to `candidate_hops`, the
        three of them and the least-connected shared neighbours never cut,
        plus the corrupted edge itself. `seed` (any string) fixes which
        neighbours a capped node keeps."""
        rng = random.Random(seed)
        relation = true_triple[1]
        slot = "head" if corrupted_triple[0] != true_triple[0] else "tail"
        shared = self.ranked_shared(anchor, relation, slot, candidate)[0][:SHARED_KEPT]
        keep = {anchor, true_filler, candidate, *shared}
        focus = {anchor: "anchor", true_filler: "true_filler",
                 candidate: "candidate"}
        caps = (max_neighbours, rng, keep, outer_neighbours)
        egos = (("anchor", self.ego(anchor, hops, *caps)),
                ("true_filler", self.ego(true_filler, hops, *caps)),
                ("candidate", self.ego(candidate, candidate_hops, *caps)))

        # Each node's distance to the nearest focus entity, and which one.
        nearest = {}
        for name, (hop, _) in egos:
            for node, depth in hop.items():
                if node not in nearest or depth < nearest[node][0]:
                    nearest[node] = (depth, name)
        ordered = ([anchor, true_filler, candidate]
                   + [node for node in shared if node not in focus]
                   + [node for node in nearest
                      if node not in focus and node not in shared])
        nodes = []
        for node in ordered:
            if node in focus:
                nodes.append({"id": node, "role": focus[node], "hop": 0,
                              "near": focus[node]})
            elif node in shared:
                nodes.append({"id": node, "role": "shared", "hop": 1,
                              "near": "shared"})
            else:
                depth, name = nearest[node]
                nodes.append({"id": node, "role": "context", "hop": depth,
                              "near": name})

        edges = [{"h": true_triple[0], "r": true_triple[1],
                  "t": true_triple[2], "kind": "true"},
                 {"h": corrupted_triple[0], "r": corrupted_triple[1],
                  "t": corrupted_triple[2], "kind": "corrupted"}]
        seen = {tuple(true_triple), tuple(corrupted_triple)}
        for _, (_, walk) in egos:
            for triple in walk:
                if triple not in seen:
                    seen.add(triple)
                    edges.append({"h": triple[0], "r": triple[1],
                                  "t": triple[2], "kind": "context"})
        return {"nodes": nodes, "edges": edges}

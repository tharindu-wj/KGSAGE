"""Phase 3 — Corruption Generation (the paper's final pipeline phase).

Loads a trained KGSAGE checkpoint and produces one corruption per input
triple. This is the public generation API of the package — a downstream
anomaly detector calls it (directly, or through its own thin bridge module)
every time it needs a batch of negatives. Everything stays in-process; no
intermediate files. PyG is never needed here: the checkpoint carries the
frozen context table E' and the membership sketches.

Vocabulary note: inside kgsage/ the emitted false triple is always a
CORRUPTION. The detector side is what re-labels it a "negative" (training) or
an "anomaly" (evaluation) — that is a deliberate seam, not two names for one
thing drifting apart. It is also why the entry point below is called
`generate_negatives` even though this package only ever says "corruption":
that name faces the detector and is frozen.

The pipeline, one corruption per real triple:

  STEP 1  Translate the caller's integer ids -> strings -> the generator's
          integer ids (the caller and the generator may number the same
          entity differently; strings are the shared language).
  STEP 2  Pick the slot to corrupt: head or tail, 50/50 (or the one the
          caller forces with `slot=`). The entity that
          keeps its slot is the ANCHOR; the value being replaced is the
          TRUE_FILLER. The relation slot is never corrupted — with head and
          tail fixed there is rarely a coherent alternative relation, so
          relation corruptions come out type-incoherent (confirmed in
          downstream-detector logs).
  STEP 3  Score every CANDIDATE in the relation's FULL type pool with the
          generator, conditioned on the frozen E' rows of the triple and the
          anchor's membership sketch.
  STEP 4  Mask out every candidate that must never be picked: the
          true_filler, every KNOWN-TRUE filler of the query across ALL
          splits (this is the falseness guarantee), the other_entity of the
          triple (self-loop), everything outside the relation's type pool,
          and — optionally — every candidate the anchor's neighbourhood
          corroborates (the corroboration mask, see
          corroborated_entities_mask).
  STEP 5  Choose the PICKED_CANDIDATE: argmax over the masked scores plus
          Gumbel noise, drawn from a torch.Generator seeded from the
          caller's numpy rng — reproducible per seed.
  STEP 6  Check the picked_candidate (no self-loop, not a real fact); up to
          `max_resample` redraws. If every redraw fails, the ORIGINAL triple
          is emitted as a null corruption, counted in `used_original` and
          listed in `null_indices` — a null is a real fact and callers must
          drop or replace it before training on it. There is NO hidden
          random fallback.
  STEP 7  Translate the corruption back to the caller's integer ids.
"""
import numpy as np
import torch


def load_checkpoint(ckpt_path, device=None, types_path=None):
    """Reconstruct the trained generator and its lookup tables from a .pt file.

    types_path (optional): a JSON {entity: [type_id, ...]} in the generator's
    entity vocabulary, attached as the payload's entity KINDS for the guards
    of generate_negatives (see attach_entity_types). The checkpoint itself
    carries no types.

    Only candidate_v2 payloads (CandidateScoringGenerator + membership
    sketches) are loadable. Older checkpoints remain useful only as
    --init_context_from E' donors for the trainer, which reads their tensors
    directly and never calls this function.

    `load_checkpoint` is a frozen public name — downstream detector code
    re-exports it (e.g. as `load_gan`), so renaming it breaks those callers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # frozen arch literal; "candidate_v2" = the dual-discriminator architecture
    if payload.get("arch") == "candidate_v2":
        loaded = _load_candidate_v2_payload(payload, ckpt_path, device)
        if types_path is not None:
            typed = attach_entity_types(loaded, types_path)
            print(f"[v2] {ckpt_path}: kinds attached for {typed:,}/"
                  f"{loaded['n_ent']:,} entities from {types_path}",
                  flush=True)
        return loaded

    raise ValueError(
        f"Checkpoint {ckpt_path!r} is not a candidate_v2 payload "
        f"(arch={payload.get('arch')!r}). Legacy checkpoints are no longer "
        "loadable; use one of the canonical generator_*.pt artifacts or "
        "retrain with `python -m kgsage.gan.train`."
    )


def _load_candidate_v2_payload(saved, ckpt_path, device):
    """Build the runtime payload dict from a candidate_v2 checkpoint.

    The keys of the returned dict are a contract — knockout_eval, the CLI
    scripts and any downstream detector code all read them. Do not rename them.

    `saved` holds the ON-DISK payload keys, which are frozen: renaming any of
    them would make every archived .pt unloadable. The runtime dict this
    returns uses the module's own vocabulary instead (see the Vocabulary note
    at the top of this file).
    """
    from kgsage.gan.generator import CandidateScoringGenerator

    generator = CandidateScoringGenerator(
        dim=saved["dim"], sketch_bits=saved["sketch_bits"],
        n_rel=saved["n_rel"]).to(device)
    # frozen payload key; generator_state = the trained G's state_dict
    generator.load_state_dict(saved["generator_state"])
    generator.eval()

    # frozen payload key; context_embeddings = the frozen context table E'
    # (the RGCN encoder's OUTPUT, not the entity_embeddings that fed into it)
    context_table = saved["context_embeddings"].to(device)

    # Known-true fillers per query, across ALL splits — the falseness bans.
    real_triple_set = set(tuple(t) for t in saved["real_triples"])
    true_tails, true_heads = {}, {}
    for h, r, t in real_triple_set:
        true_tails.setdefault((h, r), []).append(t)
        true_heads.setdefault((r, t), []).append(h)

    # frozen payload key; pool_masks = the per-relation type pools
    pool_masks = saved.get("pool_masks")
    if pool_masks is None:
        # Early smoke checkpoints predate pool storage: derive pools from the
        # all-splits triples (slightly more permissive than train-only pools).
        print(f"[v2] {ckpt_path}: no pool_masks in payload -- deriving from "
              "real_triples (all splits)", flush=True)
        pool_masks = torch.zeros(2, saved["n_rel"], saved["n_ent"],
                                 dtype=torch.bool)
        for h, r, t in real_triple_set:
            pool_masks[0, r, h] = True
            pool_masks[1, r, t] = True

    return {
        "arch": "candidate_v2",
        "generator": generator,
        "device": device,
        "context_table": context_table,
        # frozen payload key; sketches = the Bloom membership sketches
        "sketches": saved["sketches"].float(),
        "ent2id": saved["ent2id"], "rel2id": saved["rel2id"],
        "id2ent": saved["id2ent"], "id2rel": saved["id2rel"],
        "real_triple_set": real_triple_set,
        "true_tails": true_tails, "true_heads": true_heads,
        "pool_masks": pool_masks,
        "n_ent": saved["n_ent"], "n_rel": saved["n_rel"],
        "tau": saved.get("tau", 0.5),
    }


def corroborated_entities_mask(payload, anchor, support_max=0):
    """Bool [n_ent] row: candidates the anchor's neighbourhood CORROBORATES.

    A candidate x counts as corroborated when the anchor's surroundings vouch
    for it:
      - x is a direct (1-hop) neighbour of the anchor, or
      - x shares more than `support_max` neighbours with the anchor.

    `support_max` keeps its old spelling because it is a frozen public kwarg
    of generate_negatives; read it as "the corroboration tolerance".

    The CORROBORATION MASK bans exactly these, so an emitted corruption is
    contradicted by the anchor's neighbourhood BY CONSTRUCTION ("no one
    around this entity points at the picked_candidate") instead of relying on
    the learned scores alone. support_max=0 is the strict reading: one shared
    neighbour already counts as corroboration.

    The undirected adjacency is built once per payload from the checkpoint's
    all-splits triples and cached; per-anchor rows are cached too.
    """
    import scipy.sparse as sp

    cache = payload.setdefault("_corroboration_cache", {})
    cache_key = (anchor, support_max)
    if cache_key in cache:
        return cache[cache_key]

    adjacency = payload.get("_corroboration_adj")
    if adjacency is None:
        n_ent = payload["n_ent"]
        rows, cols = [], []
        for h, _, t in payload["real_triple_set"]:
            rows.append(h); cols.append(t)
            rows.append(t); cols.append(h)
        adjacency = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, cols)),
            shape=(n_ent, n_ent))
        adjacency.data[:] = 1              # collapse parallel edges to 0/1
        payload["_corroboration_adj"] = adjacency

    anchor_neighbours = adjacency.getrow(anchor)     # 1 x n: N(anchor)
    shared_counts = anchor_neighbours @ adjacency    # 1 x n: |N(anchor) ∩ N(x)|
    mask_numpy = ((anchor_neighbours.toarray()[0] > 0)
                  | (shared_counts.toarray()[0] > support_max))
    mask = torch.from_numpy(mask_numpy).to(payload["device"])
    if len(cache) < 20000:
        cache[cache_key] = mask
    return mask


def attach_entity_types(payload, types_path):
    """Attach entity KINDS to a loaded payload, for the KIND guard.

    `types_path` is a JSON file {entity_string: [type_id, ...]} in the
    generator's own entity vocabulary (CoDEx ships one as
    types/entity2types.json). The checkpoint does not carry types: its
    "type pool" only records who has occupied a slot before, which is how a
    country gets decoded as a member of a language academy. This adds the
    missing table. Returns how many generator entities received kinds.
    """
    import json
    with open(types_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    n_ent = payload["n_ent"]
    kinds = [frozenset() for _ in range(n_ent)]
    by_kind = {}
    typed = 0
    for name, type_ids in raw.items():
        gid = payload["ent2id"].get(name)
        if gid is None or not type_ids:
            continue
        kinds[gid] = frozenset(type_ids)
        typed += 1
        for kind in type_ids:
            by_kind.setdefault(kind, []).append(gid)
    payload["kinds"] = kinds
    payload["_ents_by_kind"] = {
        kind: np.array(sorted(ids), dtype=np.int64)
        for kind, ids in by_kind.items()}
    return typed


def _cooccurrence(payload, relation, slot):
    """Per (relation, slot), built lazily: how often two values are held
    TOGETHER by the same entity.

    slot 2 (tail corrupted): group the true tails of every head under the
    relation; slot 0 (head corrupted): group the true heads of every tail.
    Returns (holders, cooc): holders[v] = how many entities hold v in this
    slot, cooc[v][u] = how many of those also hold u. So
    cooc[v][u] / holders[v] is P(u | v), the share of v's holders that also
    hold u, which is what the SUPPORT guard reads.
    """
    cache = payload.setdefault("_cooccurrence_cache", {})
    key = (relation, slot)
    if key in cache:
        return cache[key]
    groups = {}
    for h, r, t in payload["real_triple_set"]:
        if r != relation:
            continue
        if slot == 2:
            groups.setdefault(h, []).append(t)
        else:
            groups.setdefault(t, []).append(h)
    holders, cooc = {}, {}
    for held in groups.values():
        for v in held:
            holders[v] = holders.get(v, 0) + 1
            row = cooc.setdefault(v, {})
            for u in held:
                if u != v:
                    row[u] = row.get(u, 0) + 1
    cache[key] = (holders, cooc)
    return cache[key]


def _support_banned(payload, anchor_values, relation, slot, tau,
                    min_holders=5):
    """Candidates the graph's own regularities PREDICT for this slot.

    For each true value v the anchor already holds, every u that more than
    `tau` of v's holders also hold is banned: emitting it would label as
    false a fact the graph expects to be true ("actor" -> "film actor").
    Values held by fewer than `min_holders` entities are too thin to judge.
    """
    holders, cooc = _cooccurrence(payload, relation, slot)
    banned = set()
    for v in anchor_values:
        count = holders.get(v, 0)
        if count < min_holders:
            continue
        for u, together in cooc.get(v, {}).items():
            if together / count > tau:
                banned.add(u)
    return banned


def _kind_allowed(payload, replaced):
    """Entities sharing at least one kind with the entity being replaced.

    CoDEx's own criterion for a hard negative: the replacement matches the
    type of the value it replaces. It holds for every hand-verified CoDEx
    negative at this granularity. None means "cannot judge" (no types
    attached, or the replaced entity is untyped) and imposes nothing.
    """
    kinds = payload.get("kinds")
    if kinds is None or not kinds[replaced]:
        return None
    parts = [payload["_ents_by_kind"][k] for k in kinds[replaced]
             if k in payload["_ents_by_kind"]]
    if not parts:
        return None
    return np.unique(np.concatenate(parts))


def build_guard_mask(payload, h, r, t, slot, guards):
    """Bool [n_ent] on the payload's device, True = banned by a GUARD.

    guards: {"kind": bool, "support": float | None}. The KIND guard keeps
    only replacements of the same kind as the true filler; the SUPPORT guard
    bans values the graph predicts for the anchor. Returns None when no
    guard has anything to say for this row.
    """
    n_ent = payload["n_ent"]
    true_filler = h if slot == 0 else t
    ban = None
    if guards.get("kind", True):
        allowed = _kind_allowed(payload, true_filler)
        if allowed is not None:
            ban = torch.ones(n_ent, dtype=torch.bool)
            ban[torch.from_numpy(allowed)] = False
    tau = guards.get("support")
    if tau is not None:
        anchor_values = (payload["true_heads"].get((r, t), []) if slot == 0
                         else payload["true_tails"].get((h, r), []))
        predicted = _support_banned(payload, anchor_values, r, slot, tau)
        if predicted:
            if ban is None:
                ban = torch.zeros(n_ent, dtype=torch.bool)
            ban[torch.tensor(sorted(predicted), dtype=torch.long)] = True
    return None if ban is None else ban.to(payload["device"])


def _pick_candidate_index(logits, true_filler_index, torch_rng,
                          banned=None, pool_row=None, other_entity=None,
                          corroboration_mask=None, guard_mask=None):
    """STEPS 4-5 for one row: apply the masks, then Gumbel-pick a candidate.

    Masks applied (each optional beyond the true_filler):
      true_filler_index  : the original value — forces the slot to change.
      banned             : every known-true filler of this query (all splits).
      other_entity       : the triple's other entity (self-loop ban).
      pool_row           : bool [n_ent] type pool — bans every candidate
                           outside the relation's observed slot fillers.
      corroboration_mask : bool [n_ent] — bans every candidate the anchor's
                           neighbourhood corroborates (see
                           corroborated_entities_mask), making the pick
                           neighbourhood-contradicting by construction.
      guard_mask         : bool [n_ent] — bans every candidate a GUARD
                           rejects (build_guard_mask: wrong kind,
                           graph-predicted, or already emitted). Treated
                           like a correctness ban.

    Picking adds Gumbel noise at temperature 0.5 (mostly argmax) drawn from
    the caller's seeded torch.Generator, so generation is reproducible.

    If a row has nothing left to pick, masks are relaxed in order: the
    corroboration mask is lifted first, then the type pool. The correctness
    bans (true_filler, known-true, self-loop) and the guard mask are NEVER
    lifted: a row with nothing left becomes a null corruption, which the
    eval role drops.

    Returns (index, lifted_corroboration): index is -1 if nothing is
    pickable; lifted_corroboration flags that the corroboration mask had to
    be dropped.
    """
    def correctness_masked():
        masked = logits.clone()
        masked[true_filler_index] = float("-inf")
        if banned:
            masked[banned] = float("-inf")
        if other_entity is not None:
            masked[other_entity] = float("-inf")
        if guard_mask is not None:
            masked[guard_mask] = float("-inf")
        return masked

    lifted_corroboration = False
    masked = correctness_masked()
    if pool_row is not None:
        masked[~pool_row] = float("-inf")
    if corroboration_mask is not None:
        masked[corroboration_mask] = float("-inf")
        if torch.isinf(masked).all():     # degenerate: lift corroboration first
            lifted_corroboration = True
            masked = correctness_masked()
            if pool_row is not None:
                masked[~pool_row] = float("-inf")
    if pool_row is not None and torch.isinf(masked).all():
        masked = correctness_masked()          # degenerate pool: lift pool ban
        if corroboration_mask is not None and not lifted_corroboration:
            masked[corroboration_mask] = float("-inf")
            if torch.isinf(masked).all():
                lifted_corroboration = True
                masked = correctness_masked()
    if torch.isinf(masked).all():
        return -1, lifted_corroboration

    uniform_noise = torch.empty_like(masked)
    uniform_noise.uniform_(generator=torch_rng).clamp_(1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    return (int((masked + gumbel_noise * 0.5).argmax().item()),
            lifted_corroboration)


#: what `slot=` accepts -> the internal slot code (0 = head, 2 = tail)
_SLOT_CODES = {"head": 0, "tail": 2, 0: 0, 2: 2}


def generate_negatives(triples, payload, id_maps, rng=None,
                       batch_size=256, max_resample=8, support_max=None,
                       guards=None, slot=None):
    """Generate one corruption per input triple. Main entry point.

    The name `generate_negatives` is frozen — downstream detector pipelines
    call it directly. Inside kgsage/ the emitted object is a CORRUPTION; it is
    the detector side that presents it as a "negative".

    triples      : list of (h, r, t) in the CALLER's integer id space.
    payload      : the dict returned by load_checkpoint().
    id_maps      : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' for the
                   caller's vocabulary (translation goes through strings).
    rng          : numpy random Generator, seeded by the caller.
    max_resample : redraws before a row degrades to a null corruption.
    support_max  : frozen public kwarg name for the corroboration tolerance.
                   None = off. An integer >= 0 turns on the CORROBORATION
                   MASK: candidates the anchor's neighbourhood corroborates
                   (direct neighbours, or more than support_max shared
                   neighbours) become unpickable, so every emitted corruption
                   contradicts the neighbourhood by construction. Degenerate
                   rows lift this mask first (counted in stats).
    guards       : None (default) = exactly the behaviour every recorded run
                   had. A dict switches on the EVAL-COLUMN GUARDS:
                     {"kind": True}    keep only replacements of the same
                                       kind as the value replaced (CoDEx's
                                       own criterion; needs types attached,
                                       see load_checkpoint(types_path=...))
                     {"support": 0.5}  ban values the graph's own
                                       regularities predict for the anchor
                                       (more than half of the entities that
                                       share one of its values hold this
                                       one too) -- these were coming out
                                       labelled false while probably true
                     {"unique": True}  never emit the same corruption twice
                                       in one call
                   A row nothing survives becomes a null corruption. Raise
                   max_resample (32) with guards on. Guards are for test
                   columns, not for training negatives.
    slot         : None (default) = head or tail, 50/50 -- every recorded
                   run. "head" / "tail" (or 0 / 2) corrupts that slot on
                   every row. The 50/50 coin is drawn either way, so a
                   forced slot leaves the caller's rng exactly where None
                   would: same seed, same sources, only the slot differs.

    Returns (corruptions, stats). stats['null_indices'] lists the positions
    whose emitted corruption is the original triple — callers training on
    these must drop or replace those rows.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if slot is not None and slot not in _SLOT_CODES:
        raise ValueError(
            f"slot={slot!r}: corrupt 'head' or 'tail' (or 0 / 2), or pass "
            "None for the 50/50 default. The relation slot is never "
            "corrupted.")
    forced_slot = None if slot is None else _SLOT_CODES[slot]

    if payload.get("arch") == "candidate_v2":
        return _generate_negatives_candidate_v2(triples, payload, id_maps,
                                                rng, max_resample, support_max,
                                                guards, forced_slot)

    raise ValueError(
        "generate_negatives requires a candidate_v2 payload; legacy "
        "checkpoints are no longer supported."
    )


def _generate_negatives_candidate_v2(triples, payload, id_maps, rng,
                                     max_resample, support_max, guards=None,
                                     forced_slot=None):
    """The candidate_v2 decode: score every candidate in the relation's FULL
    type pool per row, scatter those scores into an n_ent-wide vector, then
    run the shared mask ladder (_pick_candidate_index)."""
    generator = payload["generator"]
    device = payload["device"]
    context_table = payload["context_table"]
    sketches = payload["sketches"]
    ent2id_gen, rel2id_gen = payload["ent2id"], payload["rel2id"]
    id2ent_gen, id2rel_gen = payload["id2ent"], payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    true_tails = payload.get("true_tails", {})
    true_heads = payload.get("true_heads", {})
    pool_masks = payload["pool_masks"]
    n_ent = payload["n_ent"]

    guards = guards or {}
    unique = bool(guards.get("unique"))
    seen_fillers = {}     # (anchor, relation, slot) -> fillers emitted so far

    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))

    # STEP 1: caller ids -> strings -> generator ids.
    generator_triples = []
    for h_caller, r_caller, t_caller in triples:
        generator_triples.append(
            (ent2id_gen[id_maps["id2ent"][h_caller]],
             rel2id_gen[id_maps["id2rel"][r_caller]],
             ent2id_gen[id_maps["id2ent"][t_caller]]))

    corruptions = []
    # used_original / null_indices are frozen stats keys; downstream callers
    # read them to find and replace null corruptions before training.
    stats = {"processed": 0, "used_original": 0, "null_indices": [],
             "resampled": 0, "type_valid": 0, "slot_h": 0, "slot_r": 0,
             "slot_t": 0, "corroboration_lifted": 0,
             # guard stats: stay zero unless generate_negatives got guards
             "guard_null": 0, "unique_masked": 0}

    for row_index, (h, r, t) in enumerate(generator_triples):
        # STEP 2: pick the corrupted slot, 50/50 head or tail. The anchor is
        # whichever entity keeps its slot; the true_filler is what we replace.
        # The coin is drawn even when the caller forced a slot, so the rng
        # stream (and every later draw from it) is the same either way.
        coin_slot = 2 if rng.random() < 0.5 else 0
        slot = coin_slot if forced_slot is None else forced_slot
        if slot == 0:                                  # corrupt the HEAD
            true_filler, other_entity, anchor = h, t, t
            banned = true_heads.get((r, t))
            pool_row = pool_masks[0, r]
        else:                                          # corrupt the TAIL
            true_filler, other_entity, anchor = t, h, h
            banned = true_tails.get((h, r))
            pool_row = pool_masks[1, r]

        # STEP 3: score every candidate in the relation's type pool.
        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(n_ent)
        with torch.no_grad():
            head_id = torch.tensor([h], device=device)
            relation_id = torch.tensor([r], device=device)
            tail_id = torch.tensor([t], device=device)
            pool_scores = generator(
                head_id, relation_id, tail_id, context_table,
                sketches[[anchor]].to(device),
                pool_ids.unsqueeze(0).to(device),
                torch.zeros(1, len(pool_ids), device=device), slot)[0]
        # Scatter candidate scores into a full-vocabulary vector (everything
        # else -inf) so the mask ladder applies uniformly.
        full_scores = torch.full((n_ent,), float("-inf"), device=device)
        full_scores[pool_ids.to(device)] = pool_scores

        corroboration_mask = (
            corroborated_entities_mask(payload, anchor, support_max)
            if support_max is not None else None)

        # GUARDS (None unless generate_negatives got guards): wrong-kind and
        # graph-predicted candidates, plus every filler already emitted for
        # this (anchor, relation, slot) when uniqueness is on. A hard ban,
        # never lifted -- a row nothing survives becomes a null corruption.
        guard_mask = (build_guard_mask(payload, h, r, t, slot, guards)
                      if guards else None)
        if unique:
            prior = seen_fillers.get((anchor, r, slot))
            if prior:
                if guard_mask is None:
                    guard_mask = torch.zeros(n_ent, dtype=torch.bool,
                                             device=device)
                guard_mask[torch.tensor(sorted(prior), dtype=torch.long,
                                        device=device)] = True
                stats["unique_masked"] += 1

        # STEPS 4-6: mask, pick, check; bounded redraws.
        # emit_* is the triple actually returned; it stays == (h, r, t) if
        # every redraw fails, which is what makes the row a null corruption.
        # (Deliberately NOT named corr_*: that prefix is reserved for the
        # frozen eval-CSV columns and the corr-pick= training-log token.)
        emit_h, emit_r, emit_t = h, r, t
        emitted = False
        row_lifted = False
        for _ in range(max_resample):
            picked_index, lifted = _pick_candidate_index(
                full_scores, true_filler, torch_rng, banned=banned,
                pool_row=pool_row.to(device), other_entity=other_entity,
                corroboration_mask=corroboration_mask, guard_mask=guard_mask)
            row_lifted = row_lifted or lifted
            if picked_index < 0:
                break
            proposed_corruption = ((picked_index, r, t) if slot == 0
                                   else (h, r, picked_index))
            if (proposed_corruption[0] != proposed_corruption[2]
                    and proposed_corruption not in real_triple_set):
                emit_h, emit_r, emit_t = proposed_corruption
                emitted = True
                if unique:
                    seen_fillers.setdefault((anchor, r, slot),
                                            set()).add(picked_index)
                if bool(pool_row[picked_index]):
                    stats["type_valid"] += 1
                break
            stats["resampled"] += 1

        if not emitted:
            stats["used_original"] += 1
            stats["null_indices"].append(row_index)
            if guard_mask is not None:
                stats["guard_null"] += 1
        if row_lifted:
            stats["corroboration_lifted"] += 1
        stats["slot_h" if slot == 0 else "slot_t"] += 1

        # STEP 7: generator ids -> strings -> caller ids.
        corruptions.append((id_maps["ent2id"][id2ent_gen[emit_h]],
                            id_maps["rel2id"][id2rel_gen[emit_r]],
                            id_maps["ent2id"][id2ent_gen[emit_t]]))
        stats["processed"] += 1

    return corruptions, stats


def render_stats(stats):
    """Human-readable one-line summary of one batch of generation.

    Frozen public name AND frozen output shape: downstream training logs print
    this line verbatim, so the field names below stay as they are for log
    comparability with runs already collected.
    """
    total = stats["slot_h"] + stats["slot_r"] + stats["slot_t"]
    if total > 0:
        slot_pct = (
            f"head={stats['slot_h']}/{total}({stats['slot_h']/total:.1%}) "
            f"rel={stats['slot_r']}/{total}({stats['slot_r']/total:.1%}) "
            f"tail={stats['slot_t']}/{total}({stats['slot_t']/total:.1%})"
        )
    else:
        slot_pct = "no slots"
    processed = stats["processed"]
    used = stats["used_original"]
    fail_pct = (f"{used:,}/{processed:,}({used / processed:.1%})"
                if processed else "n/a")
    extras = ""
    if "type_valid" in stats and processed:
        extras = (f"  type_valid={stats['type_valid']:,}/{processed:,}"
                  f"({stats['type_valid'] / processed:.1%})"
                  f"  resampled={stats.get('resampled', 0):,}")
    return (
        f"processed={processed:,}  "
        f"used_original(gen_failed)={fail_pct}{extras}  "
        f"slot_distribution: {slot_pct}"
    )

"""The definitions store: what every id in a dataset MEANS, loaded once.

A dataset stores triples as opaque ids -- Q42, P106 -- which are exact but
unreadable. A CoDEx-style dataset folder carries four definition files
beside its splits, and they are what a person or an LLM can judge by:

    entities/en/entities.json    entity id -> {label, description, wiki}
    relations/en/relations.json  relation id -> {label, description}
    types/entity2types.json      entity id -> [type id, ...]   (its KINDS)
    types/en/types.json          type id -> {label, description, wiki}

KGSAGE's data/codex-s/ holds copies of the CoDEx release's files, so this
repository runs alone. A dataset without them still runs: every id stands
for itself, nothing has a kind, and the run record says so
("labels": "ids").

Translation runs one way, ids outward. Nothing here accepts a label as
input, so labels need not be unique.
"""
import json
from pathlib import Path

ENTITIES = Path("entities/en/entities.json")
RELATIONS = Path("relations/en/relations.json")
ENTITY_TYPES = Path("types/entity2types.json")
TYPES = Path("types/en/types.json")


class DatasetContext:
    """Labels, descriptions and kinds for one dataset folder."""

    def __init__(self, folder):
        folder = Path(folder)
        found = {name: (folder / name).exists()
                 for name in (ENTITIES, RELATIONS, ENTITY_TYPES, TYPES)}
        #: what the record's dataset.labels says: the full CoDEx set, some
        #: of it, or nothing (ids only)
        self.source = ("codex-json" if all(found.values())
                       else "partial" if any(found.values()) else "ids")
        self.missing = [name.as_posix() for name, ok in found.items() if not ok]
        self._entities = _read(folder / ENTITIES)
        self._relations = _read(folder / RELATIONS)
        self._entity_types = _read(folder / ENTITY_TYPES)
        self._types = _read(folder / TYPES)
        #: the kind table the kind guard attaches to a checkpoint, if any
        self.types_path = (folder / ENTITY_TYPES
                           if found[ENTITY_TYPES] else None)

    # ---- entities --------------------------------------------------------

    def entity_label(self, entity):
        return (self._entities.get(entity) or {}).get("label") or entity

    def entity_description(self, entity):
        return (self._entities.get(entity) or {}).get("description") or ""

    def entity_wiki(self, entity):
        return (self._entities.get(entity) or {}).get("wiki") or ""

    def kind_ids(self, entity):
        """The entity's kinds as type ids, in the file's order."""
        return list(self._entity_types.get(entity) or [])

    def kind_label(self, kind):
        return (self._types.get(kind) or {}).get("label") or kind

    def kind_labels(self, entity):
        return [self.kind_label(kind) for kind in self.kind_ids(entity)]

    # ---- relations -------------------------------------------------------

    def relation_label(self, relation):
        return (self._relations.get(relation) or {}).get("label") or relation

    def relation_description(self, relation):
        return (self._relations.get(relation) or {}).get("description") or ""

    # ---- triples ---------------------------------------------------------

    def triple_text(self, triple):
        """One triple as readable text: '<head> --<relation>-- <tail>'."""
        head, relation, tail = triple
        return (f"{self.entity_label(head)} --{self.relation_label(relation)}"
                f"-- {self.entity_label(tail)}")


def _read(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

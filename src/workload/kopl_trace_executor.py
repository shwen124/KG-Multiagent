"""KoPL executor with provenance tracing for KQA Pro workload extraction."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "third_party" / "KQAPro_Baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from utils.value_class import ValueClass, comp  # noqa: E402


EdgeTriple = Tuple[str, str, str]
ProgramStep = dict


QUERY_FUNCTIONS = {
    "What",
    "Count",
    "SelectBetween",
    "SelectAmong",
    "QueryAttr",
    "QueryAttrUnderCondition",
    "VerifyStr",
    "VerifyNum",
    "VerifyYear",
    "VerifyDate",
    "QueryRelation",
    "QueryAttrQualifier",
    "QueryRelationQualifier",
}


class TraceRuleExecutor:
    """Rule-based KoPL executor adapted from KQAPro_Baselines with trace collection."""

    def __init__(self, kb_json: Path):
        kb = json.loads(kb_json.read_text(encoding="utf-8"))
        self.concepts = kb["concepts"]
        self.entities = kb["entities"]

        for info in self.concepts.values():
            info["name"] = " ".join(info["name"].split())
        for info in self.entities.values():
            info["name"] = " ".join(info["name"].split())

        self.entity_name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for ent_id, ent_info in self.entities.items():
            self.entity_name_to_ids[ent_info["name"]].append(ent_id)

        self.concept_name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for con_id, con_info in self.concepts.items():
            self.concept_name_to_ids[con_info["name"]].append(con_id)

        self.concept_to_entity: Dict[str, List[str]] = defaultdict(set)
        for ent_id in self.entities:
            for concept_id in self._get_all_concepts(ent_id):
                self.concept_to_entity[concept_id].add(ent_id)
        self.concept_to_entity = {k: list(v) for k, v in self.concept_to_entity.items()}

        self.key_type: Dict[str, str] = {}
        for ent_info in self.entities.values():
            for attr_info in ent_info["attributes"]:
                self.key_type[attr_info["key"]] = attr_info["value"]["type"]
                for qk, qvs in attr_info["qualifiers"].items():
                    for qv in qvs:
                        self.key_type[qk] = qv["type"]
            for rel_info in ent_info["relations"]:
                for qk, qvs in rel_info["qualifiers"].items():
                    for qv in qvs:
                        self.key_type[qk] = qv["type"]
        self.key_type = {k: v if v != "year" else "date" for k, v in self.key_type.items()}

        for ent_info in self.entities.values():
            for attr_info in ent_info["attributes"]:
                attr_info["value"] = self._parse_value(attr_info["value"])
                for qk, qvs in attr_info["qualifiers"].items():
                    attr_info["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]
            for rel_info in ent_info["relations"]:
                for qk, qvs in rel_info["qualifiers"].items():
                    rel_info["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]

        for ent_id in self.entities:
            for rel_info in self.entities[ent_id]["relations"]:
                obj_id = rel_info["object"]
                if obj_id in self.concepts:
                    self.concepts[obj_id].setdefault("relations", []).append(
                        {
                            "predicate": rel_info["predicate"],
                            "direction": "forward"
                            if rel_info["direction"] == "backward"
                            else "backward",
                            "object": ent_id,
                            "qualifiers": rel_info["qualifiers"],
                        }
                    )

        self.reset_trace()

    def reset_trace(self) -> None:
        self.visited_vertices: Set[str] = set()
        self.visited_edges: Set[EdgeTriple] = set()
        self.relate_steps: List[List[EdgeTriple]] = []
        self.find_seeds: List[str] = []

    def execute_with_trace(self, program: List[ProgramStep], gold_answer: Optional[str] = None) -> dict:
        self.reset_trace()
        memory: List = []
        try:
            for step in program:
                func_name = step["function"]
                deps = [memory[i] for i in step["dependencies"]]
                func = getattr(self, func_name)
                result = func(deps, step["inputs"])
                memory.append(result)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "vertices": [],
                "edges": [],
                "witnesses": [],
            }

        witnesses = self._extract_witnesses(program, memory, gold_answer)
        return {
            "success": True,
            "vertices": sorted(self.visited_vertices),
            "edges": [list(edge) for edge in sorted(self.visited_edges)],
            "witnesses": witnesses,
            "answer": memory[-1] if memory else None,
        }

    def _extract_witnesses(
        self,
        program: List[ProgramStep],
        memory: List,
        gold_answer: Optional[str],
    ) -> List[List[str]]:
        final_idx = len(program) - 1
        while final_idx >= 0 and program[final_idx]["function"] in QUERY_FUNCTIONS:
            final_idx -= 1
        if final_idx < 0:
            return []

        entity_ids, _ = self._as_entity_output(memory[final_idx])
        active = set(entity_ids)
        active = self._filter_active_by_answer(active, memory[-1], gold_answer)

        path_vertices = set(active)
        for step_idx in range(final_idx, -1, -1):
            step = program[step_idx]
            fn = step["function"]
            if fn == "Relate":
                parent_idx = step["dependencies"][0]
                heads, _ = self._as_entity_output(memory[parent_idx])
                if len(heads) != 1:
                    continue
                head = heads[0]
                predicate, direction = step["inputs"]
                matched_tails = {
                    tail
                    for tail in active
                    if (head, predicate, tail) in self.visited_edges
                    or self._has_relation(head, predicate, tail, direction)
                }
                if matched_tails:
                    path_vertices.add(head)
                    path_vertices.update(matched_tails)
                    active.add(head)
            elif fn == "Find":
                for dep_idx in step["dependencies"]:
                    dep_entities, _ = self._as_entity_output(memory[dep_idx])
                    active.update(dep_entities)
                name = step["inputs"][0]
                for ent_id in self.entity_name_to_ids.get(name, []):
                    if ent_id in active or ent_id in path_vertices:
                        path_vertices.add(ent_id)

        path_vertices.update(self.find_seeds)
        if not path_vertices:
            return []
        return [sorted(path_vertices)]

    def _filter_active_by_answer(
        self,
        active: Set[str],
        result,
        gold_answer: Optional[str],
    ) -> Set[str]:
        if not active:
            return active
        if gold_answer is None:
            return active

        if isinstance(result, str):
            if result == gold_answer:
                return active
            narrowed = {
                ent_id
                for ent_id in active
                if self.entities.get(ent_id, {}).get("name") == gold_answer
            }
            return narrowed or active

        if isinstance(result, ValueClass):
            if str(result) == gold_answer:
                return active
            return active

        if isinstance(result, int):
            if str(result) == str(gold_answer):
                return active
            return active

        return active

    @staticmethod
    def _as_entity_output(value) -> Tuple[List[str], Optional[list]]:
        if isinstance(value, tuple) and len(value) == 2:
            entity_ids, facts = value
            return list(entity_ids), facts
        return [], None

    def _record_entities(self, entity_ids: Iterable[str]) -> None:
        for ent_id in entity_ids:
            self.visited_vertices.add(ent_id)

    def _record_edge(self, head: str, relation: str, tail: str) -> None:
        self.visited_vertices.add(head)
        self.visited_vertices.add(tail)
        self.visited_edges.add((head, relation, tail))

    def _has_relation(self, head: str, predicate: str, tail: str, direction: str) -> bool:
        rel_infos = (
            self.entities[head]["relations"]
            if head in self.entities
            else self.concepts[head]["relations"]
        )
        for rel_info in rel_infos:
            if (
                rel_info["predicate"] == predicate
                and rel_info["direction"] == direction
                and rel_info["object"] == tail
            ):
                return True
        return False

    def _parse_value(self, value: dict) -> ValueClass:
        if value["type"] == "date":
            x = value["value"]
            p1, p2 = x.find("/"), x.rfind("/")
            y, m, d = int(x[:p1]), int(x[p1 + 1 : p2]), int(x[p2 + 1 :])
            return ValueClass("date", date(y, m, d))
        if value["type"] == "year":
            return ValueClass("year", value["value"])
        if value["type"] == "string":
            return ValueClass("string", value["value"])
        if value["type"] == "quantity":
            return ValueClass("quantity", value["value"], value["unit"])
        raise ValueError(f"unsupported value type: {value['type']}")

    def _get_direct_concepts(self, ent_id: str) -> List[str]:
        if ent_id in self.entities:
            return self.entities[ent_id]["instanceOf"]
        return self.concepts[ent_id]["instanceOf"]

    def _get_all_concepts(self, ent_id: str) -> List[str]:
        ancestors = []
        q: Queue[str] = Queue()
        for concept_id in self._get_direct_concepts(ent_id):
            q.put(concept_id)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for parent in self.concepts[con_id]["instanceOf"]:
                q.put(parent)
        return ancestors

    def _parse_key_value(self, key: Optional[str], value: str, typ: Optional[str] = None) -> ValueClass:
        if typ is None:
            typ = self.key_type[key]
        if typ == "string":
            return ValueClass("string", value)
        if typ == "quantity":
            if " " in value:
                parts = value.split()
                unit = " ".join(parts[1:])
                return ValueClass("quantity", float(parts[0]), unit)
            return ValueClass("quantity", float(value), "1")
        if "/" in value or ("-" in value and value[0] != "-"):
            split_char = "/" if "/" in value else "-"
            p1, p2 = value.find(split_char), value.rfind(split_char)
            y, m, d = int(value[:p1]), int(value[p1 + 1 : p2]), int(value[p2 + 1 :])
            return ValueClass("date", date(y, m, d))
        return ValueClass("year", int(value))

    def FindAll(self, dependencies, inputs):
        entity_ids = list(self.entities.keys())
        self._record_entities(entity_ids)
        return entity_ids, None

    def Find(self, dependencies, inputs):
        name = inputs[0]
        entity_ids = list(self.entity_name_to_ids.get(name, []))
        for concept_id in self.concept_name_to_ids.get(name, []):
            entity_ids.append(concept_id)
        self._record_entities(entity_ids)
        self.find_seeds.extend(entity_ids)
        return entity_ids, None

    def FilterConcept(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        concept_name = inputs[0]
        concept_ids = self.concept_name_to_ids.get(concept_name, [])
        allowed = set()
        for concept_id in concept_ids:
            allowed.update(self.concept_to_entity.get(concept_id, []))
            for ent in self.concept_to_entity.get(concept_id, []):
                self._record_edge(ent, "instanceOf", concept_id)
        filtered = list(set(entity_ids) & allowed)
        self._record_entities(filtered)
        return filtered, None

    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        tgt_value_obj = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        for ent_id in entity_ids:
            for attr_info in self.entities[ent_id]["attributes"]:
                if (
                    attr_info["key"] == tgt_key
                    and attr_info["value"].can_compare(tgt_value_obj)
                    and comp(attr_info["value"], tgt_value_obj, op)
                ):
                    res_ids.append(ent_id)
                    res_facts.append(attr_info)
        self._record_entities(res_ids)
        return res_ids, res_facts

    def FilterStr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return self._filter_attribute(entity_ids, inputs[0], inputs[1], "=", "string")

    def FilterNum(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return self._filter_attribute(entity_ids, inputs[0], inputs[1], inputs[2], "quantity")

    def FilterYear(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return self._filter_attribute(entity_ids, inputs[0], inputs[1], inputs[2], "year")

    def FilterDate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return self._filter_attribute(entity_ids, inputs[0], inputs[1], inputs[2], "date")

    def _filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, op, typ):
        tgt_value_obj = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        for ent_id, fact in zip(entity_ids, facts):
            for qk, qvs in fact["qualifiers"].items():
                if qk != tgt_key:
                    continue
                for qv in qvs:
                    if qv.can_compare(tgt_value_obj) and comp(qv, tgt_value_obj, op):
                        res_ids.append(ent_id)
                        res_facts.append(fact)
        self._record_entities(res_ids)
        return res_ids, res_facts

    def QFilterStr(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        return self._filter_qualifier(entity_ids, facts, inputs[0], inputs[1], "=", "string")

    def QFilterNum(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        return self._filter_qualifier(entity_ids, facts, inputs[0], inputs[1], inputs[2], "quantity")

    def QFilterYear(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        return self._filter_qualifier(entity_ids, facts, inputs[0], inputs[1], inputs[2], "year")

    def QFilterDate(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        return self._filter_qualifier(entity_ids, facts, inputs[0], inputs[1], inputs[2], "date")

    def Relate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if not entity_ids:
            self.relate_steps.append([])
            return [], None
        head = entity_ids[0]
        predicate, direction = inputs[0], inputs[1]
        rel_infos = (
            self.entities[head]["relations"]
            if head in self.entities
            else self.concepts[head]["relations"]
        )
        res_ids = []
        res_facts = []
        step_edges: List[EdgeTriple] = []
        for rel_info in rel_infos:
            if rel_info["predicate"] == predicate and rel_info["direction"] == direction:
                tail = rel_info["object"]
                res_ids.append(tail)
                res_facts.append(rel_info)
                edge = (head, predicate, tail)
                step_edges.append(edge)
                self._record_edge(head, predicate, tail)
        self.relate_steps.append(step_edges)
        return res_ids, res_facts

    def And(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        result = list(set(entity_ids_1) & set(entity_ids_2))
        self._record_entities(result)
        return result, None

    def Or(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        result = list(set(entity_ids_1) | set(entity_ids_2))
        self._record_entities(result)
        return result, None

    def What(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        self._record_entities(entity_ids[:1])
        return self.entities[entity_ids[0]]["name"]

    def Count(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return len(entity_ids)

    def SelectBetween(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        key, op = inputs[0], inputs[1]
        v1 = v2 = None
        for attr_info in self.entities[entity_id_1]["attributes"]:
            if key == attr_info["key"]:
                v1 = attr_info["value"]
        for attr_info in self.entities[entity_id_2]["attributes"]:
            if key == attr_info["key"]:
                v2 = attr_info["value"]
        chosen = (
            entity_id_1
            if ((op == "greater" and v1 > v2) or (op == "less" and v1 < v2))
            else entity_id_2
        )
        self._record_entities([chosen])
        return self.entities[chosen]["name"]

    def SelectAmong(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, op = inputs[0], inputs[1]
        candidates = []
        for ent_id in entity_ids:
            for attr_info in self.entities[ent_id]["attributes"]:
                if key == attr_info["key"]:
                    candidates.append((ent_id, attr_info["value"]))
        candidates.sort(key=lambda item: item[1])
        chosen = candidates[0][0] if op == "smallest" else candidates[-1][0]
        self._record_entities([chosen])
        return self.entities[chosen]["name"]

    def QueryAttr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key = inputs[0]
        self._record_entities([entity_id])
        for attr_info in self.entities[entity_id]["attributes"]:
            if key == attr_info["key"]:
                return attr_info["value"]
        return None

    def QueryAttrUnderCondition(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key, qual_key, qual_value = inputs[0], inputs[1], inputs[2]
        qual_value_obj = self._parse_key_value(qual_key, qual_value)
        self._record_entities([entity_id])
        for attr_info in self.entities[entity_id]["attributes"]:
            if attr_info["key"] != key:
                continue
            for qk, qvs in attr_info["qualifiers"].items():
                if qk != qual_key:
                    continue
                for qv in qvs:
                    if qv.can_compare(qual_value_obj) and comp(qv, qual_value_obj, "="):
                        return attr_info["value"]
        return None

    def _verify(self, dependencies, value, op, typ):
        attr_value = dependencies[0]
        target = self._parse_key_value(None, value, typ)
        return "yes" if attr_value.can_compare(target) and comp(attr_value, target, op) else "no"

    def VerifyStr(self, dependencies, inputs):
        return self._verify(dependencies, inputs[0], "=", "string")

    def VerifyNum(self, dependencies, inputs):
        return self._verify(dependencies, inputs[0], inputs[1], "quantity")

    def VerifyYear(self, dependencies, inputs):
        return self._verify(dependencies, inputs[0], inputs[1], "year")

    def VerifyDate(self, dependencies, inputs):
        return self._verify(dependencies, inputs[0], inputs[1], "date")

    def QueryRelation(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        rel_infos = (
            self.entities[entity_id_1]["relations"]
            if entity_id_1 in self.entities
            else self.concepts[entity_id_1]["relations"]
        )
        for rel_info in rel_infos:
            if rel_info["object"] == entity_id_2 and rel_info["direction"] == "forward":
                self._record_edge(entity_id_1, rel_info["predicate"], entity_id_2)
                return rel_info["predicate"]
        return None

    def QueryAttrQualifier(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key, value, qual_key = inputs[0], inputs[1], inputs[2]
        value_obj = self._parse_key_value(key, value)
        self._record_entities([entity_id])
        for attr_info in self.entities[entity_id]["attributes"]:
            if (
                attr_info["key"] == key
                and attr_info["value"].can_compare(value_obj)
                and comp(attr_info["value"], value_obj, "=")
            ):
                for qk, qvs in attr_info["qualifiers"].items():
                    if qk == qual_key:
                        return qvs[0]
        return None

    def QueryRelationQualifier(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        predicate, qual_key = inputs[0], inputs[1]
        rel_infos = (
            self.entities[entity_id_1]["relations"]
            if entity_id_1 in self.entities
            else self.concepts[entity_id_1]["relations"]
        )
        for rel_info in rel_infos:
            if (
                rel_info["object"] == entity_id_2
                and rel_info["direction"] == "forward"
                and rel_info["predicate"] == predicate
            ):
                self._record_edge(entity_id_1, predicate, entity_id_2)
                for qk, qvs in rel_info["qualifiers"].items():
                    if qk == qual_key:
                        return qvs[0]
        return None

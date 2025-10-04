from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio


@dataclass
class AgentConfig:
    name: str
    personality: str


@dataclass
class AgentOutput:
    agent: str
    facts: List[str]
    classification: str
    rationale: str
    raw: str


def _load_personality_files(files: List[str]) -> List[AgentConfig]:
    configs: List[AgentConfig] = []
    for idx, f in enumerate(files):
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Personality file not found: {f}")
        text = p.read_text(encoding="utf-8").strip()
        name = p.stem or f"agent_{idx+1}"
        configs.append(AgentConfig(name=name, personality=text))
    return configs


def _get_default_llm(model: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    # Allows overriding via env; defaults to gpt-4o-mini or gpt-4o if available.
    mdl = model or os.getenv("JURY_OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=mdl, temperature=temperature)


def _make_agent_chain() -> Runnable:
    # A minimal template; we inject system and user messages dynamically per call.
    # We require strict JSON output with keys: facts, classification, rationale.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_instructions}"),
            ("human", "{user_instructions}"),
        ]
    )
    llm = _get_default_llm()
    return prompt | llm


def _build_system_instructions(agent_name: str, personality: str, allowed_classes: List[str]) -> str:
    classes_str = ", ".join(allowed_classes)
    return (
        f"You are {agent_name}, a juror with the following enduring personality and priors:\n"
        f"---\n{personality}\n---\n"
        "Debate protocol:\n"
        "- Always ground your beliefs in your stated personality and the given statement.\n"
        "- Extract and list 2-4 concrete facts you believe to be true about the statement.\n"
        f"- Provide exactly one classification, chosen strictly from: [{classes_str}].\n"
        "- You may revise your classification each round after reading others' positions.\n"
        "- Be concise and focused on verifiable or strongly believed facts.\n"
        "Output format:\n"
        '- Reply with a single minified JSON object only (no code fences), with keys:\n'
        '  {"facts": string[], "classification": string, "rationale": string}\n'
        f'- Ensure "classification" is one of: [{classes_str}].\n'
    )


def _build_user_instructions(
    statement: str,
    allowed_classes: List[str],
    round_idx: int,
    total_rounds: int,
    other_positions: List[Dict[str, Any]],
) -> str:
    # Include others' last positions as compact JSON.
    others_json = json.dumps(other_positions, ensure_ascii=False)
    classes_str = ", ".join(allowed_classes)
    return (
        f"Round {round_idx+1}/{total_rounds}\n"
        f"Statement: {statement}\n"
        f"Allowed classes: [{classes_str}]\n"
        f"Other agents' latest positions (JSON): {others_json}\n"
        "Respond now with the strict JSON object described above."
    )


def _safe_parse_agent_json(
    text: str, allowed_classes: List[str], default_class: str
) -> Tuple[List[str], str, str]:
    # Try strict parse; if not, attempt to extract a JSON object heuristically.
    data: Dict[str, Any]
    try:
        data = json.loads(text)
    except Exception:
        # Heuristic: grab the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except Exception:
                data = {}
        else:
            data = {}
    facts = data.get("facts", [])
    if not isinstance(facts, list):
        facts = [str(facts)]
    facts = [str(f).strip() for f in facts if str(f).strip()]
    raw_cls = str(data.get("classification", "")).strip()

    # Normalize classification choice
    cls_norm = raw_cls.lower()
    allowed_norm = {c.lower(): c for c in allowed_classes}
    if cls_norm in allowed_norm:
        final_cls = allowed_norm[cls_norm]
    else:
        # Try simple containment/starts-with matching
        candidates = [c for c in allowed_classes if c.lower().startswith(cls_norm) or cls_norm in c.lower()]
        final_cls = candidates[0] if candidates else default_class

    rationale = str(data.get("rationale", "")).strip()
    return facts, final_cls, rationale


def _agent_step(
    chain: Runnable,
    agent: AgentConfig,
    statement: str,
    allowed_classes: List[str],
    round_idx: int,
    total_rounds: int,
    others: List[AgentOutput],
    default_class: str,
) -> AgentOutput:
    system_txt = _build_system_instructions(agent.name, agent.personality, allowed_classes)
    other_positions = [
        {
            "agent": o.agent,
            "classification": o.classification,
            "facts": o.facts[:2],  # keep short
        }
        for o in others
    ]
    user_txt = _build_user_instructions(statement, allowed_classes, round_idx, total_rounds, other_positions)

    resp = chain.invoke({"system_instructions": system_txt, "user_instructions": user_txt})
    content = getattr(resp, "content", str(resp))
    facts, classification, rationale = _safe_parse_agent_json(content, allowed_classes, default_class=default_class)
    return AgentOutput(
        agent=agent.name,
        facts=facts,
        classification=classification,
        rationale=rationale,
        raw=content,
    )


def judge(
    statement: str,
    class_labels: List[str],
    personality_files: List[str],
    rounds: int = 3,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Run a jury debate among N= len(personality_files) agents for up to K=rounds rounds.

    Args:
        statement: The input statement to classify.
        class_labels: Predefined set of allowed classification labels.
        personality_files: Paths to N files with system prompts (one per agent).
        rounds: Max number of debate rounds K.
        model: Optional LLM model name for ChatOpenAI.
        temperature: Sampling temperature.

    Returns:
        A dictionary containing:
        - final_classification: the unanimous or majority decision.
        - unanimous_round: int or None when unanimity occurred.
        - per_round: list of lists with AgentOutput dicts per round.
        - final_votes: mapping label -> count from the final round considered.
    """
    if not class_labels:
        raise ValueError("class_labels must be a non-empty list.")
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    agents = _load_personality_files(personality_files)
    if not agents:
        raise ValueError("No agents provided (personality_files is empty).")

    # Build a shared chain with overridable model params
    llm = ChatOpenAI(model=model or os.getenv("JURY_OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_instructions}"),
            ("human", "{user_instructions}"),
        ]
    )
    chain: Runnable = prompt | llm

    per_round: List[List[Dict[str, Any]]] = []
    last_outputs: List[AgentOutput] = []

    unanimous_round: Optional[int] = None
    default_class = class_labels[0]

    for r in range(rounds):
        round_outputs: List[AgentOutput] = []
        for i, agent in enumerate(agents):
            others = [o for o in last_outputs if o.agent != agent.name] if last_outputs else []
            out = _agent_step(
                chain=chain,
                agent=agent,
                statement=statement,
                allowed_classes=class_labels,
                round_idx=r,
                total_rounds=rounds,
                others=others,
                default_class=default_class,
            )
            print("[Round %d] Agent %s classified as %s" % (r + 1, agent.name, out.classification))
            round_outputs.append(out)
        per_round.append(
            [
                {
                    "agent": o.agent,
                    "facts": o.facts,
                    "classification": o.classification,
                    "rationale": o.rationale,
                }
                for o in round_outputs
            ]
        )
        last_outputs = round_outputs

        # Early stop on unanimity
        labels = {o.classification for o in round_outputs}
        if len(labels) == 1:
            unanimous_round = r + 1  # 1-based
            break

    # Determine final decision
    final_labels = [o["classification"] for o in per_round[-1]]
    counts = Counter(final_labels)
    top_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == top_count]
    if len(tied) == 1:
        final_decision = tied[0]
    else:
        # Deterministic tie-break: by order in class_labels
        order = {c: i for i, c in enumerate(class_labels)}
        final_decision = sorted(tied, key=lambda x: order.get(x, 10**9))[0]

    return {
        "final_classification": final_decision,
        "unanimous_round": unanimous_round,
        "per_round": per_round,
        "final_votes": dict(counts),
    }


async def astream_judge(
    statement: str,
    class_labels: List[str],
    personality_files: List[str],
    rounds: int = 3,
    model: Optional[str] = None,
    temperature: float = 0.2,
):
    """
    Async streaming variant of judge(). Yields dict events:
      - {"type": "agent_output", "round": int, "agent": str, "facts": string[], "classification": str, "rationale": str}
      - {"type": "final_decision", "final_classification": str, "unanimous_round": Optional[int], "final_votes": Dict[str,int]}
    """
    if not class_labels:
        raise ValueError("class_labels must be a non-empty list.")
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    agents = _load_personality_files(personality_files)
    if not agents:
        raise ValueError("No agents provided (personality_files is empty).")

    # Build shared chain
    llm = ChatOpenAI(model=model or os.getenv("JURY_OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_instructions}"),
            ("human", "{user_instructions}"),
        ]
    )
    chain: Runnable = prompt | llm

    per_round: List[List[Dict[str, Any]]] = []
    last_outputs: List[AgentOutput] = []

    unanimous_round: Optional[int] = None
    default_class = class_labels[0]

    for r in range(rounds):
        round_outputs: List[AgentOutput] = []
        for agent in agents:
            others = [o for o in last_outputs if o.agent != agent.name] if last_outputs else []
            # Offload the blocking LLM call to a thread
            out: AgentOutput = await asyncio.to_thread(
                _agent_step,
                chain,
                agent,
                statement,
                class_labels,
                r,
                rounds,
                others,
                default_class,
            )
            round_outputs.append(out)
            # Stream this agent's result immediately
            yield {
                "type": "agent_output",
                "round": r + 1,
                "agent": out.agent,
                "facts": out.facts,
                "classification": out.classification,
                "rationale": out.rationale,
            }

        per_round.append(
            [
                {
                    "agent": o.agent,
                    "facts": o.facts,
                    "classification": o.classification,
                    "rationale": o.rationale,
                }
                for o in round_outputs
            ]
        )
        last_outputs = round_outputs

        labels = {o.classification for o in round_outputs}
        if len(labels) == 1:
            unanimous_round = r + 1
            break

    # Determine final decision (same logic as judge)
    final_labels = [o["classification"] for o in per_round[-1]]
    counts = Counter(final_labels)
    top_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == top_count]
    if len(tied) == 1:
        final_decision = tied[0]
    else:
        order = {c: i for i, c in enumerate(class_labels)}
        final_decision = sorted(tied, key=lambda x: order.get(x, 10**9))[0]

    yield {
        "type": "final_decision",
        "final_classification": final_decision,
        "unanimous_round": unanimous_round,
        "final_votes": dict(counts),
    }

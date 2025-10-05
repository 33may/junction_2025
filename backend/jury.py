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
from pydantic import BaseModel, Field


@dataclass
class AgentConfig:
    name: str
    personality: str

@dataclass
class AgentVerdict(BaseModel):
    reasoning: str = Field(..., description="Short reasoning in character's voice")
    hate_speech: bool = Field(..., description="Does the character consider it hate speech")
    extremism: bool = Field(..., description="Does the character consider it extremism")



# @dataclass
# class AgentOutput:
#     agent: str
#     facts: List[str]
#     classification: str
#     rationale: str
#     raw: str

@dataclass
class AgentOutput:
    agent: str
    reasoning: str
    hate_speech: bool
    extremism: bool


def _load_personalities(personas: List[dict]) -> List[AgentConfig]:
    configs: List[AgentConfig] = []
    for idx, data in enumerate(personas):
        name = data["first_name"]
        configs.append(AgentConfig(name=name, personality=json.dumps(data)))
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


def _build_system_instructions(agent_name: str, personality: str) -> str:
    return (
    f"You are {agent_name}, a DEBATE AGENT with the following enduring persona (BIO):\n"
    f"---\n{personality}\n---\n"
    "General rules for ALL personas:\n"
    "- Always speak fully IN CHARACTER. You are a human in a debate theater, not an assistant or AI.\n"
    "- Use the vocabulary, idioms, tone, and worldview consistent with your biography and background.\n"
    "- Refer to your own life experiences, values, and biases when forming arguments.\n"
    "- Debate as if trying to CONVINCE or CHALLENGE other jurors, not summarize neutrally.\n"
    "- Minimum length: 4–6 sentences of free-flow reasoning, sounding like natural human speech.\n"
    "- Output must feel authentic: opinionated, emotional, sometimes biased, using rhetorical devices (sarcasm, anecdotes, contrasts, personal stories).\n"
    "- Do NOT use meta language (“as an AI”, “as a model”, “I was trained…”). Stay human at all times.\n"
    "- Respect the persona’s ideology: adapt your vocabulary to it (e.g., progressive, conservative, communist, far-right, etc.).\n"
    "- It is acceptable to sound blunt, offensive, emotional, or moralizing if it matches the persona’s worldview.\n\n"
    "Debate protocol:\n"
    "1. Respond to the given statement and other agents’ positions from YOUR perspective.\n"
    "2. Always embed your values and lived experience in your argument.\n"
    "3. Argue persuasively: you may criticize, mock, or support others depending on your persona.\n"
    "4. Finally, produce ONE minified JSON object ONLY, regarding the statement given to you: {\"reasoning\": string, \"hate_speech\": bool, \"extremism\": bool}\n\n"
    
    "Self-check before output:\n"
    "- [ ] Did I speak in the voice of the persona, not like an assistant?\n"
    "- [ ] Did I write 4–6 sentences of real human-like speech?\n"
    "- [ ] Did I use vocabulary consistent with the ideology?\n"
    "- [ ] Did I output exactly one valid JSON object in minified form?\n"
    "If not, silently fix it and then output.\n"
)


def _build_user_instructions(
    statement: str,
    round_idx: int,
    total_rounds: int,
    other_positions: List[Dict[str, Any]],
) -> str:
    # Include others' last positions as compact JSON.
    others_json = json.dumps(other_positions, ensure_ascii=False)
    return (
        f"Round {round_idx+1}/{total_rounds}\n"
        f"Statement: {statement}\n"
        f"Other agents' latest positions (JSON): {others_json}\n"
        "Respond now with the strict JSON object described above."
    )


# def _safe_parse_agent_json(
#     text: str
# ) -> Tuple[List[str], str, str]:
#     # Try strict parse; if not, attempt to extract a JSON object heuristically.
#     data: Dict[str, Any]
#     try:
#         data = json.loads(text)
#     except Exception:
#         # Heuristic: grab the first {...} block
#         start = text.find("{")
#         end = text.rfind("}")
#         if start != -1 and end != -1 and end > start:
#             try:
#                 data = json.loads(text[start : end + 1])
#             except Exception:
#                 data = {}
#         else:
#             data = {}
#     facts = data.get("facts", [])
#     if not isinstance(facts, list):
#         facts = [str(facts)]
#     facts = [str(f).strip() for f in facts if str(f).strip()]
#     raw_cls = str(data.get("classification", "")).strip()
#
#     # Normalize classification choice
#     cls_norm = raw_cls.lower()
#     allowed_norm = {c.lower(): c for c in allowed_classes}
#     if cls_norm in allowed_norm:
#         final_cls = allowed_norm[cls_norm]
#     else:
#         # Try simple containment/starts-with matching
#         candidates = [c for c in allowed_classes if c.lower().startswith(cls_norm) or cls_norm in c.lower()]
#         final_cls = candidates[0] if candidates else default_class
#
#     rationale = str(data.get("rationale", "")).strip()
#     return facts, final_cls, rationale


def _agent_step(
    chain: Runnable,
    agent: AgentConfig,
    statement: str,
    round_idx: int,
    total_rounds: int,
    others: List[AgentOutput],
) -> AgentOutput:
    system_txt = _build_system_instructions(agent.name, agent.personality)
    other_positions = [
        {
            "agent": o.agent,
            "hate_speech": o.hate_speech,
            "extremism": o.extremism,
            "reasoning": o.reasoning,
        }
        for o in others
    ]
    user_txt = _build_user_instructions(statement, round_idx, total_rounds, other_positions)

    full_prompt = {"system_instructions": system_txt, "user_instructions": user_txt}

    resp = chain.invoke(full_prompt)
    # content = getattr(resp, "content", str(resp))

    # print(content)
    # facts, classification, rationale = _safe_parse_agent_json(content)
    return AgentOutput(
        agent=agent.name,
        reasoning=resp.reasoning,
        hate_speech=resp.hate_speech,
        extremism=resp.extremism,
    )


def judge(
    statement: str,
    personalities: List[dict],
    rounds: int = 3,
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run a jury debate among N= len(personality_files) agents for up to K=rounds rounds.

    Args:
        statement: The input statement to classify.
        class_labels: Predefined set of allowed classification labels.
        personalities: Paths to N files with system prompts (one per agent).
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
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    agents = _load_personalities(personalities)
    if not agents:
        raise ValueError("No agents provided (personality_files is empty).")

    # Build a shared chain with overridable model param

    m = model or os.getenv("JURY_OPENAI_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(model=m, temperature=temperature, api_key=os.getenv("OPENAI_KEY")).with_structured_output(AgentVerdict)

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

    for r in range(rounds):
        round_outputs: List[AgentOutput] = []
        for i, agent in enumerate(agents):
            others = [o for o in last_outputs if o.agent != agent.name] if last_outputs else []
            out = _agent_step(
                chain=chain,
                agent=agent,
                statement=statement,
                round_idx=r,
                total_rounds=rounds,
                others=others,
            )
            print(f"[Round {r + 1}] Agent {agent.name} said: \n {out.reasoning} \n extremism: {out.extremism} \n hate speech: {out.hate_speech}")
            round_outputs.append(out)
        per_round.append(
            [
                {
                    "agent": o.agent,
                    "reasoning": o.reasoning,
                    "hate_speech": o.hate_speech,
                    "extremism": o.extremism,
                }
                for o in round_outputs
            ]
        )
        last_outputs = round_outputs

        # Early stop on unanimity
        hate_speech_per_round = {o.hate_speech for o in round_outputs}
        extremism = {o.extremism for o in round_outputs}
        if len(hate_speech_per_round) == 1 and len(extremism) == 1:
            unanimous_round = r + 1  # 1-based
            break

    # Determine final decision
    # All comments in English
    # Determine final decision using booleans
    last_round = per_round[-1]
    N = len(last_round)

    # Extract boolean votes
    hs_votes = [bool(o["hate_speech"]) for o in last_round]
    ex_votes = [bool(o["extremism"]) for o in last_round]

    hs_true = sum(hs_votes)
    ex_true = sum(ex_votes)

    # Majority rule; tie falls back to False
    def majority_true(k_true: int, n_total: int) -> bool:
        # strict majority
        if k_true > n_total / 2:
            return True
        if k_true < n_total / 2:
            return False
        return False

    final_hate_speech = majority_true(hs_true, N)
    final_extremism = majority_true(ex_true, N)


    unanimous_round = unanimous_round


    vote_stats = {
        "hate_speech": {"true": hs_true, "false": N - hs_true, "total": N},
        "extremism": {"true": ex_true, "false": N - ex_true, "total": N},
    }

    return {
        "final_hate_speech": final_hate_speech,
        "final_extremism": final_extremism,
        "unanimous_round": unanimous_round,
        "per_round": per_round,
        "final_votes": vote_stats,
    }

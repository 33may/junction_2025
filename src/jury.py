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
        f"You are {agent_name}, a juror with the following enduring personality and priors:\n"
        f"---\n{personality}\n---\n"
        "Debate protocol:\n"
        # "- Always ground your beliefs in your stated personality and the given statement.\n"
        # "- Extract and list 2-4 concrete facts you believe to be true about the statement.\n"
        # f"- Provide exactly one classification, chosen strictly from: [{classes_str}].\n"
        # "- You may revise your classification each round after reading others' positions.\n"
        # "- Be concise and focused on verifiable or strongly believed facts.\n"
        # "Output format:\n"
        
        "Agent’s Task Instruction: When presented with a text, the agent should: Read and interpret the statement. Relate it to their worldview and life experience."
        "Express an opinion on whether the text constitutes hate speech or extremism or is simply a “legitimate opinion.” Provide a justification and reasoning that is based on your personality and not facts that are objective."
        "Be the real character, use the style of talking and level of education of specified character, The main goal is to share opinions and model sociology groups and their opinions on statements."
        "at the end give answer, as if CHARACTER consider it hate speech(yes/no), do CHARACTER consider it extremism(yes/no) structured as verdict, no sentence around just"
        
        ' Reply with a single minified JSON object only (no code fences), with keys:\n'
        '  {"reasoning": string, "hate_speech": bool, "extremism": bool}\n'
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
    default_class: str,
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

    resp = chain.invoke({"system_instructions": system_txt, "user_instructions": user_txt})
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
    class_labels: List[str],
    personalities: List[dict],
    rounds: int = 3,
    model: Optional[str] = None,
    temperature: float = 0.2,
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
    if not class_labels:
        raise ValueError("class_labels must be a non-empty list.")
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    agents = _load_personalities(personalities)
    if not agents:
        raise ValueError("No agents provided (personality_files is empty).")

    # Build a shared chain with overridable model params
    llm = ChatOpenAI(model=model or os.getenv("JURY_OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature, api_key=os.getenv("OPENAI_KEY")).with_structured_output(AgentVerdict)

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
                round_idx=r,
                total_rounds=rounds,
                others=others,
                default_class=default_class,
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

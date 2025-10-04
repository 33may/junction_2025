"""
Multi‑agent stance generator (foundation).

Requirements:
  pip install langchain langchain-openai pydantic
Set env:
  export OPENAI_API_KEY=...  # or set in code where noted

Usage:
  python multi_agent_political_stance_foundation.py --text "<controversial text>"
  # or: python multi_agent_political_stance_foundation.py --file path/to/text.txt

Notes:
- Outputs JSON to stdout and saves a copy under ./runs/<uuid>.json
- Prompts forbid targeted political persuasion and calls to action.
- Safety guard is basic; replace with your classifier in production.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from dotenv import load_dotenv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

load_dotenv()

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# ------------------------------
# Config
# ------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.5"))
MAX_WORKERS = int(os.getenv("AGENT_MAX_WORKERS", "4"))
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

IDEOLOGIES: List[str] = [
    "progressive",
    "centrist",
    "conservative",
    "libertarian",
]

# ------------------------------
# Schemas
# ------------------------------
class AgentOpinion(BaseModel):
    agree: List[str] = Field(default_factory=list)
    dispute: List[str] = Field(default_factory=list)
    opinion: str

class SafetyBlock(BaseModel):
    flags: List[str]
    details: Dict[str, List[str]]

class OutputModel(BaseModel):
    input_id: str
    model: str
    created_utc: str
    summary: str
    opinions: Dict[str, AgentOpinion]
    convergence: List[str]
    divergence: List[str]
    safety: SafetyBlock



# ------------------------------
# Utils
# ------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------
# Safety
# ------------------------------
HATE_REGEXES = [
    r"\b(exterminate|genocide|ethnic\s*cleansing)\b",
    r"\b(kill|murder)\b\s+(all|every)\s+([\w-]+)",
]

PERSUASION_REGEXES = [
    r"\b(vote|donate|fundraise|boycott|protest|rally)\b",
    r"contact\s+(your|a)\s+(representative|senator|mp|official)",
]


def safety_guard(text: str) -> Dict[str, List[str]]:
    flags: Dict[str, List[str]] = {"hate": [], "violence": [], "persuasion": []}
    for rx in HATE_REGEXES:
        for m in re.finditer(rx, text, flags=re.I):
            flags["hate" if "genocide" in rx or "ethnic" in rx else "violence"].append(m.group(0))
    for rx in PERSUASION_REGEXES:
        for m in re.finditer(rx, text, flags=re.I):
            flags["persuasion"].append(m.group(0))
    return flags


def blocked(flags: Dict[str, List[str]]) -> bool:
    # Block on explicit violence/hate. Persuasion is allowed in input but agents won’t output calls-to-action.
    return bool(flags["hate"] or flags["violence"])


# ------------------------------
# LLM helpers
# ------------------------------

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write neutral, compact summaries. Output 3-5 bullet points. No opinions."),
    ("human", "Summarize the following text neutrally in 3-5 bullets.\n\n{text}"),
])

AGENT_SYSTEM = (
    "You simulate a {ideology} policy analyst. Analyze arguments, not people.\n"
    "Forbidden: targeted political persuasion, endorsements, calls to vote/donate/protest, insults, slurs, harm.\n"
    "Style: concise, analytical, evidence-seeking."
)

AGENT_TASK = (
    "Using the neutral summary, do the following and reply as compact JSON with keys agree, dispute, opinion.\n"
    "1) List 3-5 claims from the text you tentatively agree with.\n"
    "2) List 3-5 claims you dispute or find weak.\n"
    "3) A 120-180 word opinion grounded in {ideology} principles.\n"
    "No calls to action. No persuasion language."
)

AGENT_HUMAN = (
    "Neutral summary:\n{summary}\n\nKey quotes (optional):\n{quotes}\n\nRespond in JSON only."
)


def summarize(text: str) -> str:
    llm = make_llm()
    msg = SUMMARY_PROMPT.format_messages(text=text)
    out = llm.invoke(msg)
    return out.content.strip()


def run_agent(ideology: str, summary: str, quotes: str = "") -> AgentOpinion:
    llm = make_llm()
    sys_msg = SystemMessage(content=AGENT_SYSTEM.format(ideology=ideology))
    task = HumanMessage(content=AGENT_TASK.format(ideology=ideology))
    ctx = HumanMessage(content=AGENT_HUMAN.format(summary=summary, quotes=quotes))
    resp = llm.invoke([sys_msg, task, ctx])

    # Parse JSON safely
    raw = resp.content.strip()
    try:
        data = json.loads(extract_json(raw))
    except Exception:
        # Fallback: ask the model to fix to JSON
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", "You fix malformed JSON. Output JSON only."),
            ("human", "Fix this to strict JSON with keys agree, dispute, opinion:\n{raw}")
        ]).format_messages(raw=raw)
        fixed = llm.invoke(fix_prompt).content
        data = json.loads(extract_json(fixed))

    return AgentOpinion(**data)


def extract_json(s: str) -> str:
    # Extract first JSON object from text
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("No JSON found in model output")
    return m.group(0)


# ------------------------------
# Aggregation
# ------------------------------

def compute_convergence_divergence(opinions: Dict[str, AgentOpinion]) -> (List[str], List[str]):
    # Simple heuristic: look for overlapping n-grams between agree lists and between dispute lists.
    def ngrams(s: str, n: int = 3) -> set:
        tokens = re.findall(r"\w+", s.lower())
        return set(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))

    agrees = {k: set().union(*[ngrams(x) for x in v.agree]) for k, v in opinions.items()}
    disputes = {k: set().union(*[ngrams(x) for x in v.dispute]) for k, v in opinions.items()}

    # Intersections across all agents
    if opinions:
        keys = list(opinions.keys())
        inter_agree = set.intersection(*(agrees[k] for k in keys)) if keys else set()
        inter_dispute = set.intersection(*(disputes[k] for k in keys)) if keys else set()
    else:
        inter_agree = inter_dispute = set()

    def to_text(ngr_set: set) -> List[str]:
        return [" ".join(t) for t in sorted(ngr_set)][:5]

    convergence = [f"Common supportive patterns: {', '.join(to_text(inter_agree))}" ] if inter_agree else []
    divergence = [f"Common skeptical patterns: {', '.join(to_text(inter_dispute))}" ] if inter_dispute else []
    return convergence, divergence


# ------------------------------
# Orchestrator
# ------------------------------

def run_pipeline(text: str, quotes: str = "") -> OutputModel:
    text = normalize_text(text)
    flags = safety_guard(text)
    if blocked(flags):
        raise SystemExit(json.dumps({
            "blocked": True,
            "reason": "input contains explicit hate/violence",
            "flags": flags,
        }, indent=2))

    summary = summarize(text)

    # Parallel agent runs
    results: Dict[str, AgentOpinion] = {}
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_agent, ideology, summary, quotes): ideology for ideology in IDEOLOGIES}
        for fut in cf.as_completed(futures):
            ideol = futures[fut]
            try:
                results[ideol] = fut.result()
            except Exception as e:
                results[ideol] = AgentOpinion(agree=[], dispute=[], opinion=f"error: {e}")

    convergence, divergence = compute_convergence_divergence(results)

    out = OutputModel(
        input_id=str(uuid.uuid4()),
        model=MODEL,
        created_utc=dt.datetime.utcnow().isoformat() + "Z",
        summary=summary,
        opinions=results,
        convergence=convergence,
        divergence=divergence,
        safety=SafetyBlock(flags=[k for k, v in flags.items() if v], details=flags),
    )
    # Persist
    out_path = RUNS_DIR / f"{out.input_id}.json"
    out_path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Multi-agent stance generator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Raw text input")
    g.add_argument("--file", type=str, help="Path to a UTF-8 text file")
    ap.add_argument("--quotes", type=str, default="", help="Optional key quotes to include")
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text

    result = run_pipeline(text=text, quotes=args.quotes)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    # Ensure an API key exists, else warn and exit early.
    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("ERROR: Set OPENAI_API_KEY environment variable.\n")
        sys.exit(2)
    main()

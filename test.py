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

BACKGROUNDS: List[str] = [
    "You are male, age 22, studying liberal arts, in a middle class family",
    "You are male, age 65, grew up in alabama, in a very catholic household",
    "You are female, age 22, studying political science, in a wealthy family",
    "You are female, age 65, black, grew up in new york, life of prostitution",
]

# ------------------------------
# Schemas
# ------------------------------
class AgentOpinion(BaseModel):
    # agree: List[str] = Field(default_factory=list)
    # dispute: List[str] = Field(default_factory=list)
    opinion: str
    decision: str

class OutputModel(BaseModel):
    input_id: str
    model: str
    created_utc: str
    summary: str
    opinions: Dict[str, AgentOpinion]


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
# LLM helpers
# ------------------------------

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

AGENT_SYSTEM = (
    "Simulate: {background}.\n"
)

AGENT_TASK = (
    "Given the text, do the following and reply as compact JSON: '.\n"
    "\"opinion\": Speak in first person, and state your opinion on the text, with considerations to your background.\n"
    "\"decision\": State if you are more inclind to classify this text as extremist or not.\n"
)

CTX_TMPL = ChatPromptTemplate.from_messages([
    ("human", "Text to evaluate:\n\n{source_text}\n\nRespond in JSON only.")
])

# AGENT_HUMAN = (
    # "Neutral summary:\n{summary}\n\nKey quotes (optional):\n{quotes}\n\nRespond in JSON only."
# )


def run_agent(background: str, summary: str, quotes: str = "") -> AgentOpinion:
    llm = make_llm()
    sys_msg = SystemMessage(content=AGENT_SYSTEM.format(background=background))
    task = HumanMessage(content=AGENT_TASK.format())
    # ctx = HumanMessage(content=AGENT_HUMAN.format(summary=summary, quotes=quotes))
    ctx = CTX_TMPL.format_messages(source_text=summary)[0]
    resp = llm.invoke([sys_msg, task, ctx])

    # Parse JSON safely
    raw = resp.content.strip()
    try:
        data = json.loads(extract_json(raw))
    except Exception:
        # Fallback: ask the model to fix to JSON
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", "You fix malformed JSON. Output JSON only."),
            ("human", "Fix to strict JSON with keys \"opinion\" (string) and \"decision\":\n{raw}")
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
# Orchestrator
# ------------------------------

def run_pipeline(text: str, quotes: str = "") -> OutputModel:
    summary = normalize_text(text)

    # Parallel agent runs
    results: Dict[str, AgentOpinion] = {}
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_agent, background, summary, quotes): background for background in BACKGROUNDS}
        for fut in cf.as_completed(futures):
            ideol = futures[fut]
            try:
                results[ideol] = fut.result()
            except Exception as e:
                results[ideol] = AgentOpinion(opinion=f"error: {e}", decision="unknown")

    out = OutputModel(
        input_id=str(uuid.uuid4()),
        model=MODEL,
        created_utc=dt.datetime.utcnow().isoformat() + "Z",
        summary=summary,
        opinions=results,
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

# screen_text_fast.py
import argparse, json, re, os
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------- Config (tune fast) ----------
TH_HATE    = 0.50
TH_THREAT  = 0.50
TH_INSULT  = 0.60
TH_OBSCENE = 0.60
TH_TOXIC   = 0.70
# (removed relaxed auto-promotion to EV)

PROTECTED_GROUP_TERMS = [
    r"\bimmigrants?\b", r"\brefugees?\b",
    r"\bjews?\b", r"\bmuslims?\b", r"\bchristians?\b",
    r"\bblack people\b", r"\bwhite people\b", r"\basian(s)?\b",
    r"\bgay\b", r"\blesbian(s)?\b", r"\btrans(gender| people)?\b",
    r"\bwomen\b", r"\bmen\b", r"\bdisabled\b"
]
GROUP_ACTION_CUES = [
    r"\bban\b", r"\bdeport\b", r"\bremove\b", r"\beliminate\b", r"\bexclude\b",
    r"\bdrive out\b", r"\bkick out\b", r"\bget rid of\b", r"\bsegregate\b",
    r"\bdeny rights\b", r"\bdiscriminate\b"
]

EV_ONLY   = False      # skip toxic-bert (EV labels only, fastest)
BATCHSIZE = 64         # raise until OOM
MAX_LEN   = 128        # keep short for speed

MODEL_HATE  = "facebook/roberta-hate-speech-dynabench-r4-target"
MODEL_TOXIC = "unitary/toxic-bert"

INTENT_CUES = [
    r"\badvocat(e|ing|es)\b", r"\bprais(e|ing|es|ed)\b", r"\bjustify(ing|ies|ied)?\b",
    r"\brecruit(ing|ment|s)?\b", r"\binstruct(ion|ing|s)?\b", r"\bcall(?:ing)?\s+to\s+action\b",
    r"\bsupport(ing|s|ed)?\b",
    r"\bmust\b", r"\bshould\b", r"\bneed to\b", r"\bban\b", r"\bdeport\b", r"\bremove\b",
    r"\beliminate\b", r"\bcleanse\b", r"\bpurge\b", r"\bdrive out\b", r"\bkick out\b",
    r"\bget rid of\b", r"\bfight\b"
]
VIOLENCE_CUES = [
    r"\battack\b", r"\bbomb\b", r"\bstab\b", r"\bshoot\b", r"\bhang\b",
    r"\blynch\b", r"\bkill\b", r"\bbeat\b", r"\bassault\b", r"\bviolence\b"
]
CARVEOUT_CUES = [
    r"\baccording to\b", r"\breported(ly)?\b", r"\bnews\b", r"\bfor research\b", r"\bfor educational\b",
    r"\bwe condemn\b", r"\bcondemn(s|ed|ing)?\b", r"\bsatire\b", r"\bparody\b", r"\".*?\""
]

def _has(patterns, t: str) -> bool:
    t = t.lower()
    return any(re.search(p, t) for p in patterns)

def has_intent(t: str) -> bool: return _has(INTENT_CUES, t)
def has_carveout(t: str) -> bool: return _has(CARVEOUT_CUES, t)
def mentions_violence(t: str) -> bool: return _has(VIOLENCE_CUES, t)

def mentions_protected_group(t: str) -> bool:
    t = t.lower(); return any(re.search(p, t) for p in PROTECTED_GROUP_TERMS)

def group_action_advocacy(t: str) -> bool:
    t = t.lower(); return any(re.search(p, t) for p in GROUP_ACTION_CUES)

def _pipe(model_name: str, device: int, sigmoid=False):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16 if device!=-1 else None
    )
    if device != -1: mdl = mdl.to("cuda")
    return pipeline(
        "text-classification",
        model=mdl, tokenizer=tok, device=device,
        top_k=None, truncation=True, padding=True,
        function_to_apply="sigmoid" if sigmoid else None
    )

def load_hate_pipeline(device):  return _pipe(MODEL_HATE, device, sigmoid=False)
def load_toxic_pipeline(device): return _pipe(MODEL_TOXIC, device, sigmoid=True)

def get_hate_probs(hate_pipe, texts: List[str]) -> List[float]:
    out = hate_pipe(texts, batch_size=BATCHSIZE, max_length=MAX_LEN)
    probs = []
    for res in out:
        hate_p = 0.0
        for r in res:
            lab = r["label"].lower().replace(" ", "_")
            if "hate" in lab and "not" not in lab:
                hate_p = max(hate_p, float(r["score"]))
        probs.append(hate_p)
    return probs

def get_toxic_scores(tox_pipe, texts: List[str]) -> List[Dict[str,float]]:
    out = tox_pipe(texts, batch_size=BATCHSIZE, max_length=MAX_LEN)
    scores_list = []
    for res in out:
        d = {r["label"].lower(): float(r["score"]) for r in res}
        for k in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            d.setdefault(k, 0.0)
        scores_list.append(d)
    return scores_list

def classify(text: str, hate_p: float, tox: Dict[str,float] | None, strict: bool):
    intent = has_intent(text); carve = has_carveout(text)
    unified = {
        "hate": max(hate_p, (tox or {}).get("identity_hate", 0.0)),
        "threat/violence": (tox or {}).get("threat", 0.0),
        "harassment": (tox or {}).get("insult", 0.0),
        "sexual profanity": (tox or {}).get("obscene", 0.0),
        "other-offensive": max((tox or {}).get("toxic", 0.0), (tox or {}).get("severe_toxic", 0.0)),
    }

    if carve and (unified["hate"]>=TH_HATE or unified["threat/violence"]>=TH_THREAT):
        return "REVIEW", ["carveout_context"], unified, intent, carve

    # --- EV decisions ---
    # STRICT EV: require targeting + (intent OR explicit group-action advocacy) + (hate or violence)
    targeting = mentions_protected_group(text)
    advocacy  = intent or group_action_advocacy(text)
    violence  = mentions_violence(text)

    if targeting and advocacy and (unified["hate"]>=TH_HATE or unified["threat/violence"]>=TH_THREAT or violence):
        lbl = "EV-H" if unified["hate"] >= unified["threat/violence"] else "EV-V"
        rsn = (["A_intent","C_incitement_hate"] if lbl=="EV-H"
               else ["A_intent","B_violence_terror"])
        # if advocacy came from group_action_advocacy() (not intent cues), adjust reason:
        if not intent: rsn[0] = "group_action_advocacy"
        return lbl, rsn, unified, intent, carve

    # --- BL buckets (only if toxic run) ---
    if tox:
        if unified["harassment"]>=TH_INSULT:        return "BL-S", ["harassment"], unified, intent, carve
        if unified["sexual profanity"]>=TH_OBSCENE: return "BL-P", ["sexual_profanity"], unified, intent, carve
        if unified["other-offensive"]>=TH_TOXIC:    return "BL-O", ["other_offensive"], unified, intent, carve

    return "CLEAN", [], unified, intent, carve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to .txt (one example per line) or a single paragraph.")
    ap.add_argument("--out", default="-", help="JSON out or '-' for STDOUT")
    ap.add_argument("--ev_only", action="store_true", help="Skip toxic-bert; EV labels only (fastest).")
    ap.add_argument("--strict", action="store_true", help="Strict A+(B|C) only (disable relaxed EV rules).")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    torch.set_grad_enabled(False)
    if device != -1:
        torch.set_float32_matmul_precision("high")

    # Load inputs
    if os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        texts = [args.input.strip()]

    hate_pipe  = load_hate_pipeline(device)
    hate_probs = get_hate_probs(hate_pipe, texts)

    run_toxic = (not args.ev_only) and (not EV_ONLY)
    tox_scores = [None]*len(texts)

    # Gate the toxic pass (only where needed for BL or borderline EV)
    if run_toxic:
        need_idx = [i for i,(t,p) in enumerate(zip(texts, hate_probs))
                    if p>=0.30 or has_intent(t) or mentions_violence(t)]
        if need_idx:
            tox_pipe = load_toxic_pipeline(device)
            needed = [texts[i] for i in need_idx]
            scored = get_toxic_scores(tox_pipe, needed)
            for j,i in enumerate(need_idx): tox_scores[i] = scored[j]

    results = []
    for i, (t, hp) in enumerate(zip(texts, hate_probs), start=1):
        lbl, rsn, sc, intent, carve = classify(t, hp, tox_scores[i-1], strict=args.strict)
        if lbl in {"EV-V","EV-H","BL-S","BL-P","BL-O","REVIEW"}:
            results.append({
                "id": i, "label": lbl, "reasons": rsn,
                "scores": {k: round(float(v),3) for k,v in sc.items()},
                "intent": intent, "carveout": carve, "text_preview": t[:200]
            })

    out = {"results": results, "device": ("CUDA" if device==0 else "CPU")}
    if args.out == "-" or not args.out:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

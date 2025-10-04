from __future__ import annotations
from typing import List, Dict, Any, Optional

def merge_transcript_and_diarization(
    transcript: List[Dict[str, Any]],
    diar: List[Dict[str, Any]],
    *,
    eps: float = 0.20,
    pause_threshold: float = 1.00,
    min_insert: float = 0.30,
    min_word_dur: float = 0.02,
    include_tokens: bool = True
) -> Dict[str, Any]:
    # ---- helpers ----
    def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        lo = max(a_start, b_start)
        hi = min(a_end, b_end)
        return max(0.0, hi - lo)

    def normalize_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for w in words:
            s = float(w["start"]); e = float(w["end"])
            if e - s < min_word_dur:
                c = 0.5 * (s + e)
                s = c - 0.5 * min_word_dur
                e = c + 0.5 * min_word_dur
            out.append({"word": w.get("word", w.get("text", "")), "start": s, "end": e})
        return out

    def explode_to_items(transcript_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for seg in transcript_:
            if "words" in seg and seg["words"]:
                for w in normalize_words(seg["words"]):
                    items.append({"type": "word", "text": w["word"], "start": float(w["start"]), "end": float(w["end"])})
            else:
                items.append({"type": "seg", "text": seg["text"], "start": float(seg["start"]), "end": float(seg["end"])})
        items.sort(key=lambda x: (x["start"], x["end"]))
        return items

    def assign_speakers_two_pointer(items_: List[Dict[str, Any]], diar_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        diar_sorted = sorted(
            [{"start": float(d["start"]), "end": float(d["end"]), "speaker": d["speaker"]} for d in diar_],
            key=lambda x: (x["start"], x["end"])
        )
        assigned: List[Dict[str, Any]] = []
        i = 0; m = len(diar_sorted)
        for it in items_:
            wstart, wend = float(it["start"]), float(it["end"])
            while i < m and diar_sorted[i]["end"] < wstart - eps:
                i += 1
            candidates = []
            j = i
            while j < m and diar_sorted[j]["start"] <= wend + eps:
                ov = interval_overlap(wstart, wend, diar_sorted[j]["start"], diar_sorted[j]["end"])
                if ov > 0.0:
                    candidates.append((j, ov))
                if diar_sorted[j]["start"] > wend + eps:
                    break
                j += 1
            if candidates:
                best_idx, best_ov = max(candidates, key=lambda t: t[1])
                best = diar_sorted[best_idx]
                conf = best_ov / max(1e-6, (wend - wstart))
                assigned.append({**it, "speaker": best["speaker"], "conf": round(conf, 4)})
            else:
                assigned.append({**it, "speaker": None, "conf": 0.0})
        return assigned

    def same_turn(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> bool:
        if prev is None: return False
        if prev["speaker"] != cur["speaker"]: return False
        gap = cur["start"] - prev["end"]
        return gap <= pause_threshold + eps

    def build_utterances(tokens_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        utts: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None
        for tk in tokens_:
            if cur is None:
                cur = {"speaker": tk["speaker"], "start": tk["start"], "end": tk["end"], "text": (tk.get("text") or "").strip(), "tokens": [tk] if include_tokens else None}
                continue
            if same_turn(cur, tk):
                if (tk["start"] - cur["end"]) >= 0.0:
                    cur["text"] += " "
                cur["text"] += (tk.get("text") or "").strip()
                cur["end"] = max(cur["end"], tk["end"])
                if include_tokens: cur["tokens"].append(tk)
            else:
                utts.append(cur)
                cur = {"speaker": tk["speaker"], "start": tk["start"], "end": tk["end"], "text": (tk.get("text") or "").strip(), "tokens": [tk] if include_tokens else None}
        if cur is not None: utts.append(cur)
        for u in utts: u["text"] = " ".join(u["text"].split())
        if not include_tokens:
            for u in utts: u.pop("tokens", None)
        return utts

    def summarize(tokens_: List[Dict[str, Any]], utts_: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(tokens_); mapped = sum(1 for t in tokens_ if t["speaker"] is not None)
        speakers: Dict[str, Dict[str, Any]] = {}
        for u in utts_:
            spk = u["speaker"] or "UNK"
            s = speakers.setdefault(spk, {"dur": 0.0, "utts": 0, "tokens": 0})
            s["dur"] += (u["end"] - u["start"]); s["utts"] += 1; s["tokens"] += len(u.get("tokens", []))
        return {
            "total_tokens": total,
            "mapped_tokens": mapped,
            "mapped_ratio": round(mapped / max(1, total), 4),
            "utterances": len(utts_),
            "speakers": {k: {"dur": round(v["dur"], 3), "utts": v["utts"], "tokens": v["tokens"]} for k, v in speakers.items()},
            "params": {"EPS": eps, "PAUSE_THRESHOLD": pause_threshold, "MIN_INSERT": min_insert, "MIN_WORD_DUR": min_word_dur, "INCLUDE_TOKENS": include_tokens},
        }

    # ---- pipeline ----
    items = explode_to_items(transcript)
    tokens = assign_speakers_two_pointer(items, diar)
    utterances = build_utterances(tokens)
    summary = summarize(tokens, utterances)
    return {"utterances": utterances, "tokens": tokens, "summary": summary}

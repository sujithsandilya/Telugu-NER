"""
Telugu Numbers-to-Words Pipeline
=================================
Processes Telugu language input text and converts all numeric expressions
to their Telugu word equivalents.

Pipeline:
  1. Telugu language validation (langdetect)
  2. Multilingual NER (ai4bharat/IndicNER) on GPU
  3. Regex-based number type classification
  4. Gemma 3 (HuggingFace) with Telugu-language prompt rules
  5. Reconstructed output text written to output .txt file

Usage:
  python telugu_num2words.py input.txt output.txt

Install dependencies:
  pip install transformers torch langdetect sentencepiece accelerate

Example input  : "అతను 5వ మార్చి 2023న ఉదయం 11 గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు"
Example output : "అతను ఐదవ మార్చి రెండు వేల ఇరవై మూడు సంవత్సరంలో ఉదయం పదకొండు గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు"
"""

import re
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from langdetect import detect, LangDetectException
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    AutoModelForCausalLM,
)
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NER_MODEL_NAME = "ai4bharat/IndicNER"
LLM_MODEL_NAME = "google/gemma-3-1b-it"          # change to gemma-3-4b-it for better quality
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# HELPER FUNCTIONS — for regex handler lambdas
# ─────────────────────────────────────────────

TELUGU_ONES = [
    "", "ఒక", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది",
    "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పదనాలుగు",
    "పదిహేను", "పదహారు", "పదిహేడు", "పదిహెనిమిది", "పందొమ్మిది",
]
TELUGU_TENS = [
    "", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై",
]
TELUGU_MONTHS = {
    1: "జనవరి", 2: "ఫిబ్రవరి", 3: "మార్చి", 4: "ఏప్రిల్",
    5: "మే", 6: "జూన్", 7: "జూలై", 8: "ఆగస్టు",
    9: "సెప్టెంబర్", 10: "అక్టోబర్", 11: "నవంబర్", 12: "డిసెంబర్",
}
SIMPLE_FRACTIONS_TE = {
    "1/2": "సగం", "1/3": "మూడో వంతు", "2/3": "రెండు మూడో వంతులు",
    "1/4": "నాలుగో వంతు", "3/4": "మూడు నాలుగో వంతులు",
}

TELUGU_SPECIAL_TENS = {
    21: "ఇరవై ఒక", 22: "ఇరవై రెండు", 23: "ఇరవై మూడు", 24: "ఇరవై నాలుగు",
    25: "ఇరవై ఐదు", 26: "ఇరవై ఆరు", 27: "ఇరవై ఏడు", 28: "ఇరవై ఎనిమిది",
    29: "ఇరవై తొమ్మిది",
}

def _telugu_under_100(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return TELUGU_ONES[n]
    if n in TELUGU_SPECIAL_TENS:
        return TELUGU_SPECIAL_TENS[n]
    tens = TELUGU_TENS[n // 10]
    ones = TELUGU_ONES[n % 10]
    return tens + (" " + ones if ones else "")

def _telugu_under_1000(n: int) -> str:
    if n < 100:
        return _telugu_under_100(n)
    hundreds = n // 100
    rest = n % 100
    result = TELUGU_ONES[hundreds] + " వందల"
    if rest:
        result += " " + _telugu_under_100(rest)
    return result

def num_to_words_telugu(n: int) -> str:
    if n == 0:
        return "సున్నా"
    if n < 0:
        return "మైనస్ " + num_to_words_telugu(-n)
    parts = []
    crore = n // 10_000_000
    lakh = (n % 10_000_000) // 100_000
    thousand = (n % 100_000) // 1_000
    remainder = n % 1_000
    if crore:
        parts.append(_telugu_under_1000(crore) + " కోట్లు")
    if lakh:
        parts.append(_telugu_under_1000(lakh) + " లక్షలు")
    if thousand:
        parts.append(_telugu_under_1000(thousand) + " వేలు")
    if remainder:
        parts.append(_telugu_under_1000(remainder))
    return " ".join(parts)

TELUGU_ORDINALS = {
    1: "మొదటి", 2: "రెండవ", 3: "మూడవ", 4: "నాలుగవ", 5: "ఐదవ",
    6: "ఆరవ", 7: "ఏడవ", 8: "ఎనిమిదవ", 9: "తొమ్మిదవ", 10: "పదవ",
}

def ordinal_to_words_telugu(n: int) -> str:
    if n in TELUGU_ORDINALS:
        return TELUGU_ORDINALS[n]
    return num_to_words_telugu(n) + "వ"

def year_to_words_telugu(y: int) -> str:
    if 1000 <= y <= 9999:
        thousands = y // 1000
        rest = y % 1000
        result = TELUGU_ONES[thousands] + " వేల"
        if rest >= 100:
            hundreds = rest // 100
            rem100 = rest % 100
            result += " " + TELUGU_ONES[hundreds] + " వందల"
            if rem100:
                result += " " + _telugu_under_100(rem100)
        elif rest:
            result += " " + _telugu_under_100(rest)
        return result
    return num_to_words_telugu(y)

def date_to_words_telugu(day: int, month: int, year: int) -> str:
    return f"{ordinal_to_words_telugu(day)} {TELUGU_MONTHS[month]} {year_to_words_telugu(year)}"

def digits_spoken_telugu(s: str) -> str:
    digit_words = {
        "0": "సున్నా", "1": "ఒక", "2": "రెండు", "3": "మూడు", "4": "నాలుగు",
        "5": "ఐదు", "6": "ఆరు", "7": "ఏడు", "8": "ఎనిమిది", "9": "తొమ్మిది",
    }
    return " ".join(digit_words[d] for d in s)

def decimal_to_words_telugu(s: str) -> str:
    integer_part, decimal_part = s.split(".")
    decimal_words = " ".join(
        num_to_words_telugu(int(d)) if int(d) > 0 else "సున్నా" for d in decimal_part
    )
    return num_to_words_telugu(int(integer_part)) + " దశాంశం " + decimal_words

def amount_to_words_telugu(s: str) -> str:
    n = int(s.replace(",", ""))
    return num_to_words_telugu(n) + " రూపాయలు"

UNIT_MAP_TE = {
    "mg": "మిల్లీగ్రాముల", "kg": "కిలోగ్రాముల", "ml": "మిల్లీలీటర్ల",
    "g": "గ్రాముల", "lb": "పౌండ్ల", "pound": "పౌండ్ల", "pounds": "పౌండ్ల",
    "litre": "లీటర్ల", "litres": "లీటర్ల", "liter": "లీటర్ల", "liters": "లీటర్ల",
    "km": "కిలోమీటర్ల", "cm": "సెంటీమీటర్ల",
    "లీటర్": "లీటర్ల", "కిలో": "కిలోగ్రాముల",
}

def unit_to_words_telugu(number_str: str, unit: str) -> str:
    unit_lower = unit.lower().rstrip("s")
    full_unit = UNIT_MAP_TE.get(unit, UNIT_MAP_TE.get(unit_lower, unit))
    try:
        if "." in number_str:
            return decimal_to_words_telugu(number_str) + " " + full_unit
        return num_to_words_telugu(int(number_str)) + " " + full_unit
    except Exception:
        return number_str + " " + full_unit

def address_to_words_telugu(a: str, b: str, sep: str) -> str:
    sep_word = "స్లాష్" if sep == "/" else "డాష్"
    return num_to_words_telugu(int(a)) + f" {sep_word} " + num_to_words_telugu(int(b))

# ─────────────────────────────────────────────
# REGEX PATTERNS — number type classification
# ─────────────────────────────────────────────
PATTERNS = [
    ("DOB",
     re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{4})\b'),
     lambda m: date_to_words_telugu(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

    ("PHONE",
     re.compile(r'(?<!\d)([6-9]\d{9})(?!\d)'),
     lambda m: digits_spoken_telugu(m.group(1))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+\.\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm|లీటర్|కిలో)(?!\w)', re.I),
     lambda m: unit_to_words_telugu(m.group(1), m.group(2))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm|లీటర్|కిలో)(?!\w)', re.I),
     lambda m: unit_to_words_telugu(m.group(1), m.group(2))),

    ("AMOUNT",
     re.compile(r'₹\s*(\d[\d,]*)'),
     lambda m: amount_to_words_telugu(m.group(1))),

    ("AMOUNT",
     re.compile(r'(?<!\d)(\d[\d,]*)(?=\s*(రూపాయలు|రూపాయిలు|అద్దె))'),
     lambda m: amount_to_words_telugu(m.group(1))),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})/(\d{1,4})(?!\d)'),
     lambda m: SIMPLE_FRACTIONS_TE.get(f"{m.group(1)}/{m.group(2)}")
               or address_to_words_telugu(m.group(1), m.group(2), "/")),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})-(\d{1,4})(?!\d)'),
     lambda m: address_to_words_telugu(m.group(1), m.group(2), "-")),

    ("DECIMAL",
     re.compile(r'(?<!\d)(\d+\.\d+)(?!\d)'),
     lambda m: decimal_to_words_telugu(m.group(1))),

    ("TIME",
     re.compile(
         r'\b(\d{1,2})\s*(?:AM|PM|am|pm|గంటలకు|గంటలు)\b'
         r'|\b(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)?\b',
         re.IGNORECASE
     ),
     None),  # handled by LLM

    ("ORDINAL",
     re.compile(r'\b(\d+)(st|nd|rd|th|వ|వ)\b', re.IGNORECASE),
     lambda m: ordinal_to_words_telugu(int(m.group(1)))),

    ("YEAR",
     re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)'),
     lambda m: year_to_words_telugu(int(m.group(1)))),

    ("NUMBER",
     re.compile(r'(?<!\d)(\d+)(?!\d)'),
     lambda m: num_to_words_telugu(int(m.group(1)))),
]

# ─────────────────────────────────────────────
# STEP 1 — LANGUAGE VALIDATION
# ─────────────────────────────────────────────
def validate_telugu(text: str) -> bool:
    try:
        return detect(text) == "te"
    except LangDetectException:
        return False

# ─────────────────────────────────────────────
# STEP 2 — NER
# ─────────────────────────────────────────────
def load_ner_pipeline():
    print("[NER] Loading ai4bharat/IndicNER model...")
    tokenizer = AutoTokenizer.from_pretrained(
    NER_MODEL_NAME,
    use_fast=True
)
    model = AutoModelForTokenClassification.from_pretrained(
    NER_MODEL_NAME,
    use_safetensors=True
)
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE,
    )
    print("[NER] Model loaded.")
    return ner

def run_ner(ner_pipe, text: str) -> list:
    return ner_pipe(text)

# ─────────────────────────────────────────────
# STEP 3 — REGEX CLASSIFICATION
# ─────────────────────────────────────────────
def classify_numbers(text: str) -> list:
    found = []
    covered = set()

    for (num_type, pattern, handler) in PATTERNS:
        for m in pattern.finditer(text):
            span_set = set(range(m.start(), m.end()))
            if span_set & covered:
                continue
            covered |= span_set
            found.append({
                "start":   m.start(),
                "end":     m.end(),
                "type":    num_type,
                "text":    m.group(),
                "handler": handler,
                "match":   m,
            })

    found.sort(key=lambda x: x["start"])
    return found

# ─────────────────────────────────────────────
# STEP 4 — LLM CONVERSION (Telugu Prompt, for TIME)
# ─────────────────────────────────────────────
def load_llm():
    print("[LLM] Loading Gemma 3 model...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map=DEVICE_MAP,
        torch_dtype=torch.bfloat16,
    )
    print("[LLM] Gemma 3 loaded.")
    return tokenizer, model

def build_telugu_prompt(number_text: str, number_type: str) -> str:
    type_rules = {
        "TIME": (
            "ఈ సంఖ్య ఒక సమయాన్ని సూచిస్తుంది. "
            "దీన్ని తెలుగు పదాలలో రాయండి. "
            "ఉదాహరణ: 11am → పదకొండు గంటలు, 3pm → మూడు గంటలు, 9:30 → తొమ్మిది గంటల ముప్పై నిమిషాలు."
        ),
    }
    rule = type_rules.get(number_type, "ఈ సంఖ్యను తెలుగు పదాలలో రాయండి.")
    return f"""మీరు తెలుగు భాష నిపుణుడు. సంఖ్యలను తెలుగు పదాలలోకి మార్చడంలో మీకు ప్రత్యేక నైపుణ్యం ఉంది.

నియమం: {rule}

సంఖ్య: {number_text}

కేవలం మార్చబడిన తెలుగు పదాలు మాత్రమే ఇవ్వండి. వివరణలు అవసరం లేదు.

సమాధానం:"""

def convert_with_llm(tokenizer, model, number_text: str, number_type: str) -> str:
    prompt = build_telugu_prompt(number_text, number_type)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE_MAP)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result.split("\n")[0].strip()

# ─────────────────────────────────────────────
# STEP 5 — RECONSTRUCT OUTPUT
# ─────────────────────────────────────────────
def reconstruct(original_text: str, spans: list, conversions: dict) -> str:
    result = original_text
    for span in reversed(spans):
        replacement = conversions.get((span["start"], span["end"]), span["text"])
        result = result[:span["start"]] + replacement + result[span["end"]:]
    return result

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(text: str, llm_state: dict) -> str:
    print(f"\n{'='*60}")
    print(f"Input : {text}")
    print(f"{'='*60}")

    if not validate_telugu(text):
        print("[WARNING] Input does not appear to be Telugu. Proceeding anyway...")

    ner_entities = run_ner(llm_state["ner"], text)
    print(f"\n[NER] Entities found:")
    for e in ner_entities:
        print(f"  {e['word']!r:20s} → {e['entity_group']}")

    spans = classify_numbers(text)
    print(f"\n[Regex] Number spans found:")
    for s in spans:
        print(f"  {s['text']!r:20s} → {s['type']}")

    if not spans:
        print("\n[INFO] No numeric expressions found. Returning original text.")
        return text

    conversions = {}
    llm_needed = [s for s in spans if s["handler"] is None]
    if llm_needed and "tokenizer" not in llm_state:
        llm_state["tokenizer"], llm_state["model"] = load_llm()

    print(f"\n[Converting] Numbers to Telugu words:")
    for span in spans:
        if span["handler"] is not None:
            converted = span["handler"](span["match"])
        else:
            converted = convert_with_llm(
                llm_state["tokenizer"], llm_state["model"],
                span["text"], span["type"]
            )
        conversions[(span["start"], span["end"])] = converted
        print(f"  {span['text']!r:20s} ({span['type']:10s}) → {converted!r}")

    output = reconstruct(text, spans, conversions)
    print(f"\n{'='*60}")
    print(f"Output: {output}")
    print(f"{'='*60}\n")
    return output

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Telugu Numbers-to-Words Pipeline"
    )
    parser.add_argument("input_file",  help="Path to input .txt file")
    parser.add_argument("output_file", help="Path to output .txt file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    print(f"[INFO] Read {len(lines)} line(s) from '{args.input_file}'")

    llm_state = {"ner": load_ner_pipeline()}

    results = []
    for line in lines:
        if line.strip():
            results.append(run_pipeline(line, llm_state))
        else:
            results.append("")

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        if results:
            f.write("\n")

    print(f"\n[INFO] Output written to '{args.output_file}'")

if __name__ == "__main__":
    main()
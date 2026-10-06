"""
English Numbers-to-Words Pipeline
===================================
Processes English language input text and converts all numeric expressions
to their English word equivalents.

Pipeline:
  1. English language validation (langdetect)
  2. Multilingual NER (ai4bharat/IndicNER) on GPU
  3. Regex-based number type classification
  4. Gemma 3 (HuggingFace) with English-language prompt rules
  5. Reconstructed output text written to output .txt file

Usage:
  python english_num2words.py input.txt output.txt

Install dependencies:
  pip install transformers torch langdetect sentencepiece accelerate

Example input  : "he went to the school on 5th March 2023 at 11am through bus"
Example output : "he went to the school on fifth March two thousand and twenty three at eleven AM through bus"
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

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"]
ORDINAL_MAP = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth"
}
MONTHS_EN = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
SIMPLE_FRACTIONS = {
    "1/2": "one half", "1/3": "one third", "2/3": "two thirds",
    "1/4": "one quarter", "3/4": "three quarters",
}

def _num_under_1000(n: int) -> str:
    if n < 20:
        return ONES[n]
    elif n < 100:
        rest = ONES[n % 10]
        return TENS[n // 10] + ("-" + rest if rest else "")
    else:
        rest = _num_under_1000(n % 100)
        return ONES[n // 100] + " hundred" + (" and " + rest if rest else "")

def num_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + num_to_words(-n)
    parts = []
    billions = n // 1_000_000_000
    millions = (n % 1_000_000_000) // 1_000_000
    thousands = (n % 1_000_000) // 1_000
    remainder = n % 1_000
    if billions:
        parts.append(_num_under_1000(billions) + " billion")
    if millions:
        parts.append(_num_under_1000(millions) + " million")
    if thousands:
        parts.append(_num_under_1000(thousands) + " thousand")
    if remainder:
        parts.append(_num_under_1000(remainder))
    return " ".join(parts)

def ordinal_to_words(n: int) -> str:
    base = num_to_words(n)
    last_word = base.split()[-1]
    if last_word in ORDINAL_MAP:
        suffix = ORDINAL_MAP[last_word]
        return " ".join(base.split()[:-1] + [suffix]) if len(base.split()) > 1 else suffix
    elif last_word.endswith("t"):
        return base + "h"
    elif last_word.endswith("e"):
        return base[:-1] + "th" if not last_word.endswith("th") else base
    elif last_word.endswith("y"):
        return base[:-1] + "ieth"
    else:
        return base + "th"

def year_to_words(y: int) -> str:
    if 1100 <= y <= 1999:
        high, low = divmod(y, 100)
        h = _num_under_1000(high)
        l = ("oh " + ONES[low]) if 1 <= low <= 9 else (_num_under_1000(low) if low else "hundred")
        return h + " " + l
    return num_to_words(y)

def date_to_words(day: int, month: int, year: int) -> str:
    return f"{ordinal_to_words(day)} {MONTHS_EN[month]} {year_to_words(year)}"

def digits_spoken(s: str) -> str:
    digit_words = {"0": "zero", "1": "one", "2": "two", "3": "three",
                   "4": "four", "5": "five", "6": "six", "7": "seven",
                   "8": "eight", "9": "nine"}
    return " ".join(digit_words[d] for d in s)

def decimal_to_words(s: str) -> str:
    integer_part, decimal_part = s.split(".")
    return num_to_words(int(integer_part)) + " point " + " ".join(
        ONES[int(d)] if int(d) > 0 else "zero" for d in decimal_part
    )

def amount_to_words(s: str) -> str:
    n = int(s.replace(",", ""))
    return num_to_words(n) + " rupees"

UNIT_MAP = {
    "mg": "milligrams", "kg": "kilograms", "ml": "millilitres",
    "g": "grams", "lb": "pounds", "pound": "pounds", "pounds": "pounds",
    "litre": "litres", "litres": "litres", "liter": "litres", "liters": "litres",
    "km": "kilometres", "cm": "centimetres",
}

def unit_to_words(number_str: str, unit: str) -> str:
    unit_lower = unit.lower().rstrip("s")
    full_unit = UNIT_MAP.get(unit_lower, UNIT_MAP.get(unit.lower(), unit))
    try:
        if "." in number_str:
            return decimal_to_words(number_str) + " " + full_unit
        return num_to_words(int(number_str)) + " " + full_unit
    except Exception:
        return number_str + " " + full_unit

def address_to_words(a: str, b: str, sep: str) -> str:
    sep_word = "slash" if sep == "/" else "dash"
    return num_to_words(int(a)) + f" {sep_word} " + num_to_words(int(b))

# ─────────────────────────────────────────────
# REGEX PATTERNS — number type classification
# (ordered by priority; first match wins)
# ─────────────────────────────────────────────
PATTERNS = [
    ("DOB",
     re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{4})\b'),
     lambda m: date_to_words(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

    ("PHONE",
     re.compile(r'(?<!\d)([6-9]\d{9})(?!\d)'),
     lambda m: digits_spoken(m.group(1))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+\.\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm)(?!\w)', re.I),
     lambda m: unit_to_words(m.group(1), m.group(2))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm)(?!\w)', re.I),
     lambda m: unit_to_words(m.group(1), m.group(2))),

    ("AMOUNT",
     re.compile(r'₹\s*(\d[\d,]*)'),
     lambda m: amount_to_words(m.group(1))),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})/(\d{1,4})(?!\d)'),
     lambda m: SIMPLE_FRACTIONS.get(f"{m.group(1)}/{m.group(2)}")
               or address_to_words(m.group(1), m.group(2), "/")),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})-(\d{1,4})(?!\d)'),
     lambda m: address_to_words(m.group(1), m.group(2), "-")),

    ("DECIMAL",
     re.compile(r'(?<!\d)(\d+\.\d+)(?!\d)'),
     lambda m: decimal_to_words(m.group(1))),

    ("TIME",
     re.compile(r'\b(\d{1,2})\s*(?:AM|PM|am|pm)\b|\b(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)?\b', re.IGNORECASE),
     None),  # handled by LLM

    ("ORDINAL",
     re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE),
     lambda m: ordinal_to_words(int(m.group(1)))),

    ("YEAR",
     re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)'),
     lambda m: year_to_words(int(m.group(1)))),

    ("NUMBER",
     re.compile(r'(?<!\d)(\d+)(?!\d)'),
     lambda m: num_to_words(int(m.group(1)))),
]

# ─────────────────────────────────────────────
# STEP 1 — LANGUAGE VALIDATION
# ─────────────────────────────────────────────
def validate_english(text: str) -> bool:
    try:
        return detect(text) == "en"
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
    """
    Scan text with regex patterns and return list of spans:
      { 'start', 'end', 'type', 'text', 'handler' }
    Ordered by position, non-overlapping (first match wins).
    """
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
# STEP 4 — LLM CONVERSION (for TIME spans without a handler)
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

def build_english_prompt(number_text: str, number_type: str) -> str:
    type_rules = {
        "TIME": (
            "This is a time expression. "
            "Convert it to its English word form. "
            "Examples: 11am → eleven AM, 3pm → three PM, 9:30am → nine thirty AM."
        ),
    }
    rule = type_rules.get(number_type, "Convert this number to its English word form.")
    return f"""You are an expert English language editor specializing in converting numeric expressions to their word equivalents.

Rule: {rule}

Number: {number_text}

Respond with ONLY the converted word form. No explanation, no punctuation, no extra text.

Answer:"""

def convert_with_llm(tokenizer, model, number_text: str, number_type: str) -> str:
    prompt = build_english_prompt(number_text, number_type)
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

    if not validate_english(text):
        print("[WARNING] Input does not appear to be English. Proceeding anyway...")

    # NER (informational context)
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

    print(f"\n[Converting] Numbers to English words:")
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
        description="English Numbers-to-Words Pipeline"
    )
    parser.add_argument("input_file",  help="")
    parser.add_argument("output_file", help="")
    args = parser.parse_args()

    # Read input
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    print(f"[INFO] Read {len(lines)} line(s) from '{args.input_file}'")

    # Load NER once for all lines
    llm_state = {"ner": load_ner_pipeline()}

    # Process each line
    results = []
    for line in lines:
        if line.strip():
            results.append(run_pipeline(line, llm_state))
        else:
            results.append("")  # preserve blank lines

    # Write output
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        if results:
            f.write("\n")

    print(f"\n[INFO] Output written to '{args.output_file}'")

if __name__ == "__main__":
    main()
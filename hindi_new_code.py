"""
Hindi Numbers-to-Words Pipeline
================================
Processes Hindi language input text and converts all numeric expressions
to their Hindi word equivalents.

Pipeline:
  1. Hindi language validation (langdetect)
  2. Multilingual NER (ai4bharat/IndicNER) on GPU
  3. Regex-based number type classification
  4. Gemma 3 (HuggingFace) with Hindi-language prompt rules
  5. Reconstructed output text written to output .txt file

Usage:
  python hindi_num2words.py input.txt output.txt

Install dependencies:
  pip install transformers torch langdetect sentencepiece accelerate

Example input  : "वह 5 मार्च 2023 को सुबह 11 बजे बस से स्कूल गया"
Example output : "वह पाँच मार्च दो हज़ार तेईस को सुबह ग्यारह बजे बस से स्कूल गया"
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

HINDI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छः", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
    "सत्रह", "अठारह", "उन्नीस",
]
HINDI_TENS = [
    "", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे",
]
HINDI_TEEN_VARIANTS = {
    21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
    31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
    36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
    41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस",
    46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास",
    51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
    56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
    61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ",
    66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर",
    71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे",
    96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे",
}

HINDI_MONTHS = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल",
    5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त",
    9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर",
}

SIMPLE_FRACTIONS_HI = {
    "1/2": "आधा", "1/3": "एक तिहाई", "2/3": "दो तिहाई",
    "1/4": "एक चौथाई", "3/4": "तीन चौथाई",
}

def _hindi_under_100(n: int) -> str:
    if n == 0:
        return ""
    if n in HINDI_TEEN_VARIANTS:
        return HINDI_TEEN_VARIANTS[n]
    if n < 20:
        return HINDI_ONES[n]
    return HINDI_TENS[n // 10] + (" " + HINDI_ONES[n % 10] if n % 10 else "")

def _hindi_under_1000(n: int) -> str:
    if n < 100:
        return _hindi_under_100(n)
    hundreds = n // 100
    rest = n % 100
    result = HINDI_ONES[hundreds] + " सौ"
    if rest:
        result += " " + _hindi_under_100(rest)
    return result

def num_to_words_hindi(n: int) -> str:
    if n == 0:
        return "शून्य"
    if n < 0:
        return "ऋण " + num_to_words_hindi(-n)
    parts = []
    crore = n // 10_000_000
    lakh = (n % 10_000_000) // 100_000
    thousand = (n % 100_000) // 1_000
    remainder = n % 1_000
    if crore:
        parts.append(_hindi_under_1000(crore) + " करोड़")
    if lakh:
        parts.append(_hindi_under_1000(lakh) + " लाख")
    if thousand:
        parts.append(_hindi_under_1000(thousand) + " हज़ार")
    if remainder:
        parts.append(_hindi_under_1000(remainder))
    return " ".join(parts)

HINDI_ORDINALS = {
    1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 5: "पाँचवाँ",
    6: "छठा", 7: "सातवाँ", 8: "आठवाँ", 9: "नौवाँ", 10: "दसवाँ",
}

def ordinal_to_words_hindi(n: int) -> str:
    if n in HINDI_ORDINALS:
        return HINDI_ORDINALS[n]
    return num_to_words_hindi(n) + "वाँ"

def year_to_words_hindi(y: int) -> str:
    if 1100 <= y <= 1999:
        high, low = divmod(y, 100)
        return _hindi_under_1000(high) + " सौ " + (_hindi_under_100(low) if low else "")
    return num_to_words_hindi(y)

def date_to_words_hindi(day: int, month: int, year: int) -> str:
    return f"{ordinal_to_words_hindi(day)} {HINDI_MONTHS[month]} {year_to_words_hindi(year)}"

def digits_spoken_hindi(s: str) -> str:
    digit_words = {
        "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
        "5": "पाँच", "6": "छः", "7": "सात", "8": "आठ", "9": "नौ",
    }
    return " ".join(digit_words[d] for d in s)

def decimal_to_words_hindi(s: str) -> str:
    integer_part, decimal_part = s.split(".")
    decimal_words = " ".join(
        num_to_words_hindi(int(d)) if int(d) > 0 else "शून्य" for d in decimal_part
    )
    return num_to_words_hindi(int(integer_part)) + " दशमलव " + decimal_words

def amount_to_words_hindi(s: str) -> str:
    n = int(s.replace(",", ""))
    return num_to_words_hindi(n) + " रुपये"

UNIT_MAP_HI = {
    "mg": "मिलीग्राम", "kg": "किलोग्राम", "ml": "मिलीलीटर",
    "g": "ग्राम", "lb": "पाउंड", "pound": "पाउंड", "pounds": "पाउंड",
    "litre": "लीटर", "litres": "लीटर", "liter": "लीटर", "liters": "लीटर",
    "km": "किलोमीटर", "cm": "सेंटीमीटर",
}

def unit_to_words_hindi(number_str: str, unit: str) -> str:
    unit_lower = unit.lower().rstrip("s")
    full_unit = UNIT_MAP_HI.get(unit_lower, UNIT_MAP_HI.get(unit.lower(), unit))
    try:
        if "." in number_str:
            return decimal_to_words_hindi(number_str) + " " + full_unit
        return num_to_words_hindi(int(number_str)) + " " + full_unit
    except Exception:
        return number_str + " " + full_unit

def address_to_words_hindi(a: str, b: str, sep: str) -> str:
    sep_word = "स्लैश" if sep == "/" else "डैश"
    return num_to_words_hindi(int(a)) + f" {sep_word} " + num_to_words_hindi(int(b))

# ─────────────────────────────────────────────
# REGEX PATTERNS — number type classification
# ─────────────────────────────────────────────
PATTERNS = [
    ("DOB",
     re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{4})\b'),
     lambda m: date_to_words_hindi(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

    ("PHONE",
     re.compile(r'(?<!\d)([6-9]\d{9})(?!\d)'),
     lambda m: digits_spoken_hindi(m.group(1))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+\.\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm)(?!\w)', re.I),
     lambda m: unit_to_words_hindi(m.group(1), m.group(2))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm)(?!\w)', re.I),
     lambda m: unit_to_words_hindi(m.group(1), m.group(2))),

    ("AMOUNT",
     re.compile(r'₹\s*(\d[\d,]*)'),
     lambda m: amount_to_words_hindi(m.group(1))),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})/(\d{1,4})(?!\d)'),
     lambda m: SIMPLE_FRACTIONS_HI.get(f"{m.group(1)}/{m.group(2)}")
               or address_to_words_hindi(m.group(1), m.group(2), "/")),

    ("ADDRESS",
     re.compile(r'(?<!\d)(\d{1,4})-(\d{1,4})(?!\d)'),
     lambda m: address_to_words_hindi(m.group(1), m.group(2), "-")),

    ("DECIMAL",
     re.compile(r'(?<!\d)(\d+\.\d+)(?!\d)'),
     lambda m: decimal_to_words_hindi(m.group(1))),

    ("TIME",
     re.compile(r'\b(\d{1,2})\s*(?:AM|PM|am|pm|बजे|घंटे)\b|\b(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)?\b', re.IGNORECASE),
     None),  # handled by LLM

    ("ORDINAL",
     re.compile(r'\b(\d+)(st|nd|rd|th|वाँ|वीं|वें)\b', re.IGNORECASE),
     lambda m: ordinal_to_words_hindi(int(m.group(1)))),

    ("YEAR",
     re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)'),
     lambda m: year_to_words_hindi(int(m.group(1)))),

    ("NUMBER",
     re.compile(r'(?<!\d)(\d+)(?!\d)'),
     lambda m: num_to_words_hindi(int(m.group(1)))),
]

# ─────────────────────────────────────────────
# STEP 1 — LANGUAGE VALIDATION
# ─────────────────────────────────────────────
def validate_hindi(text: str) -> bool:
    try:
        return detect(text) == "hi"
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
# STEP 4 — LLM CONVERSION (Hindi Prompt, for TIME)
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

def build_hindi_prompt(number_text: str, number_type: str) -> str:
    type_rules = {
        "TIME": (
            "यह संख्या एक समय को दर्शाती है। "
            "इसे हिंदी शब्दों में लिखें। "
            "उदाहरण: 11am → ग्यारह बजे, 3pm → तीन बजे, 9:30am → साढ़े नौ बजे।"
        ),
    }
    rule = type_rules.get(number_type, "इस संख्या को हिंदी शब्दों में लिखें।")
    return f"""आप एक हिंदी भाषा विशेषज्ञ हैं। आपको संख्याओं को हिंदी शब्दों में बदलने में महारत है।

नियम: {rule}

संख्या: {number_text}

केवल परिवर्तित हिंदी शब्द दें। कोई स्पष्टीकरण नहीं चाहिए।

उत्तर:"""

def convert_with_llm(tokenizer, model, number_text: str, number_type: str) -> str:
    prompt = build_hindi_prompt(number_text, number_type)
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

    if not validate_hindi(text):
        print("[WARNING] Input does not appear to be Hindi. Proceeding anyway...")

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

    print(f"\n[Converting] Numbers to Hindi words:")
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
        description="Hindi Numbers-to-Words Pipeline"
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
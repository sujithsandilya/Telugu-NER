"""
Telugu NER + Text Normalizer
=============================
Pipeline: input.txt → IndicNER (entity detection) → number normalization → output.txt

Dependencies:
    pip install transformers torch

Model: ai4bharat/IndicNER  (Hugging Face)

Usage:
    python telugu_ner_converter.py                        # uses input.txt → output.txt
    python telugu_ner_converter.py myfile.txt             # uses myfile.txt → myfile_output.txt
    python telugu_ner_converter.py input.txt output.txt   # custom input and output paths
"""

import re
import sys
import os


# ── Telugu vocabulary ──────────────────────────────────────────────────────────

ONES = [
    "", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "అయిదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పదనాలుగు", "పదిహేను", "పదహారు",
    "పదిహేడు", "పదెనిమిది", "పంతొమ్మిది", "ఇరవై", "ఇరవై ఒకటి", "ఇరవై రెండు",
    "ఇరవై మూడు", "ఇరవై నాలుగు", "ఇరవై అయిదు", "ఇరవై ఆరు", "ఇరవై ఏడు",
    "ఇరవై ఎనిమిది", "ఇరవై తొమ్మిది", "ముప్పై", "ముప్పై ఒకటి", "ముప్పై రెండు",
    "ముప్పై మూడు", "ముప్పై నాలుగు", "ముప్పై అయిదు", "ముప్పై ఆరు", "ముప్పై ఏడు",
    "ముప్పై ఎనిమిది", "ముప్పై తొమ్మిది", "నలభై", "నలభై ఒకటి", "నలభై రెండు",
    "నలభై మూడు", "నలభై నాలుగు", "నలభై అయిదు", "నలభై ఆరు", "నలభై ఏడు",
    "నలభై ఎనిమిది", "నలభై తొమ్మిది", "యాభై", "యాభై ఒకటి", "యాభై రెండు",
    "యాభై మూడు", "యాభై నాలుగు", "యాభై అయిదు", "యాభై ఆరు", "యాభై ఏడు",
    "యాభై ఎనిమిది", "యాభై తొమ్మిది", "అరవై", "అరవై ఒకటి", "అరవై రెండు",
    "అరవై మూడు", "అరవై నాలుగు", "అరవై అయిదు", "అరవై ఆరు", "అరవై ఏడు",
    "అరవై ఎనిమిది", "అరవై తొమ్మిది", "డెబ్బై", "డెబ్బై ఒకటి", "డెబ్బై రెండు",
    "డెబ్బై మూడు", "డెబ్బై నాలుగు", "డెబ్బై అయిదు", "డెబ్బై ఆరు", "డెబ్బై ఏడు",
    "డెబ్బై ఎనిమిది", "డెబ్బై తొమ్మిది", "ఎనభై", "ఎనభై ఒకటి", "ఎనభై రెండు",
    "ఎనభై మూడు", "ఎనభై నాలుగు", "ఎనభై అయిదు", "ఎనభై ఆరు", "ఎనభై ఏడు",
    "ఎనభై ఎనిమిది", "ఎనభై తొమ్మిది", "తొంభై", "తొంభై ఒకటి", "తొంభై రెండు",
    "తొంభై మూడు", "తొంభై నాలుగు", "తొంభై అయిదు", "తొంభై ఆరు", "తొంభై ఏడు",
    "తొంభై ఎనిమిది", "తొంభై తొమ్మిది",
]

DIGITS = ["సున్న","ఒకటి","రెండు","మూడు","నాలుగు","అయిదు","ఆరు","ఏడు","ఎనిమిది","తొమ్మిది"]

ORDINALS = {
    1:"మొదటి", 2:"రెండవ", 3:"మూడవ", 4:"నాల్గవ", 5:"అయిదవ",
    6:"ఆరవ", 7:"ఏడవ", 8:"ఎనిమిదవ", 9:"తొమ్మిదవ", 10:"పదవ",
    11:"పదకొండవ", 12:"పన్నెండవ", 13:"పదమూడవ", 14:"పదనాల్గవ", 15:"పదిహేనవ",
    16:"పదహారవ", 17:"పదిహేడవ", 18:"పదెనిమిదవ", 19:"పంతొమ్మిదవ", 20:"ఇరవైవ",
    21:"ఇరవై ఒకటవ", 22:"ఇరవై రెండవ", 23:"ఇరవై మూడవ", 24:"ఇరవై నాల్గవ",
    25:"ఇరవై అయిదవ", 26:"ఇరవై ఆరవ", 27:"ఇరవై ఏడవ", 28:"ఇరవై ఎనిమిదవ",
    29:"ఇరవై తొమ్మిదవ", 30:"ముప్పైవ", 31:"ముప్పై ఒకటవ",
}

MONTHS = {
    1:"జనవరి", 2:"ఫిబ్రవరి", 3:"మార్చి", 4:"ఏప్రిల్", 5:"మే", 6:"జూన్",
    7:"జులై", 8:"ఆగస్టు", 9:"సెప్టెంబర్", 10:"అక్టోబర్", 11:"నవంబర్", 12:"డిసెంబర్",
}

UNITS = {
    "mg":"మిల్లీగ్రాములు", "kg":"కిలోగ్రాములు", "g":"గ్రాములు",
    "ml":"మిల్లీలీటర్లు", "l":"లీటర్లు",
    "litre":"లీటర్లు", "litres":"లీటర్లు", "liter":"లీటర్లు", "liters":"లీటర్లు",
    "lb":"పౌండ్లు", "pound":"పౌండ్లు", "pounds":"పౌండ్లు",
    "km":"కిలోమీటర్లు", "m":"మీటర్లు", "cm":"సెంటీమీటర్లు",
    "లీటర్":"లీటర్లు", "కిలో":"కిలోగ్రాములు",
}

SIMPLE_FRACTIONS = {
    "1/2":"సగం", "1/3":"మూడో వంతు", "2/3":"మూడింట రెండు వంతులు",
    "1/4":"పావు", "3/4":"పావు తక్కువ",
}


# ── Number converters ─────────────────────────────────────────────────────────

def num_to_words(n: int) -> str:
    if n == 0:   return "సున్న"
    if n < 0:    return "మైనస్ " + num_to_words(-n)
    if n < 100:  return ONES[n]
    if n < 1_000:
        h, r = divmod(n, 100)
        base = ("" if h == 1 else ONES[h] + " ") + "వంద" + ("లు" if h > 1 else "")
        return base if r == 0 else base + " " + ONES[r]
    if n < 1_00_000:
        t, r = divmod(n, 1000)
        base = ("" if t == 1 else ONES[t] + " ") + "వెయ్యి"
        return base if r == 0 else base + " " + num_to_words(r)
    if n < 1_00_00_000:
        lk, r = divmod(n, 1_00_000)
        base = ("" if lk == 1 else ONES[lk] + " ") + "లక్ష" + ("లు" if lk > 1 else "")
        return base if r == 0 else base + " " + num_to_words(r)
    cr, r = divmod(n, 1_00_00_000)
    base = ("" if cr == 1 else ONES[cr] + " ") + "కోటి"
    return base if r == 0 else base + " " + num_to_words(r)


def digits_spoken(s: str) -> str:
    return " ".join(DIGITS[int(d)] for d in s if d.isdigit())

def decimal_to_words(s: str) -> str:
    left, right = s.split(".")
    return num_to_words(int(left)) + " పాయింట్ " + " ".join(DIGITS[int(d)] for d in right)

def year_to_words(y: int) -> str:
    if 2000 <= y <= 2099:
        r = y - 2000
        return "రెండు వేలు" if r == 0 else "రెండు వేల " + num_to_words(r)
    if 1900 <= y <= 1999:
        r = y - 1900
        return "పంతొమ్మిది వందలు" if r == 0 else "పంతొమ్మిది వందల " + num_to_words(r)
    return num_to_words(y)

def date_to_words(d: int, m: int, y: int) -> str:
    return f"{ORDINALS.get(d, num_to_words(d)+'వ')} {MONTHS.get(m, str(m))} {year_to_words(y)}"

def address_to_words(a: str, b: str, sep: str) -> str:
    return f"{num_to_words(int(a))} {'బార్' if sep=='/' else 'డాష్'} {num_to_words(int(b))}"

def unit_to_words(number: str, unit: str) -> str:
    tel_unit = UNITS.get(unit.lower(), unit)
    num_word = decimal_to_words(number) if "." in number else num_to_words(int(number))
    return f"{num_word} {tel_unit}"

def amount_to_words(number: str) -> str:
    return num_to_words(int(number.replace(",", ""))) + " రూపాయలు"


# ── Patterns: most specific first ─────────────────────────────────────────────

PATTERNS = [
    ("DOB",
     re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{4})\b'),
     lambda m: date_to_words(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

    ("PHONE",
     re.compile(r'(?<!\d)([6-9]\d{9})(?!\d)'),
     lambda m: digits_spoken(m.group(1))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+\.\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm|లీటర్|కిలో)(?!\w)', re.I),
     lambda m: unit_to_words(m.group(1), m.group(2))),

    ("UNIT",
     re.compile(r'(?<!\d)(\d+)\s*(mg|kg|ml|g|lb|pounds?|litres?|liters?|km|cm|లీటర్|కిలో)(?!\w)', re.I),
     lambda m: unit_to_words(m.group(1), m.group(2))),

    ("AMOUNT",
     re.compile(r'₹\s*(\d[\d,]*)'),
     lambda m: amount_to_words(m.group(1))),

    ("AMOUNT",
     re.compile(r'(?<!\d)(\d[\d,]*)(?=\s*(రూపాయలు|రూపాయిలు|అద్దె))'),
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

    ("YEAR",
     re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)'),
     lambda m: year_to_words(int(m.group(1)))),

    ("NUMBER",
     re.compile(r'(?<!\d)(\d+)(?!\d)'),
     lambda m: num_to_words(int(m.group(1)))),
]


# ── NER using ai4bharat/IndicNER ──────────────────────────────────────────────

# NER label colors for terminal display
NER_COLORS = {
    "PER":  "\033[94m",   # Blue
    "ORG":  "\033[92m",   # Green
    "LOC":  "\033[93m",   # Yellow
    "NUM":  "\033[96m",   # Cyan
    "DATE": "\033[95m",   # Magenta
    "O":    "\033[0m",    # Reset
}
RESET = "\033[0m"

_ner_pipeline = None  # lazy-loaded


def load_ner_model():
    """Load IndicNER model once and cache it."""
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        print("❌  'transformers' not installed. Run: pip install transformers torch")
        sys.exit(1)

    print("⏳  Loading ai4bharat/IndicNER model (first run downloads ~500MB)...")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    model     = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

    _ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # merges sub-word tokens into full words
        device=0 if _cuda_available() else -1,
    )
    print("✅  Model loaded.\n")
    return _ner_pipeline


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_ner(text: str) -> list:
    """
    Run IndicNER on a text string.

    Returns a list of dicts:
        [{"word": str, "entity_group": str, "score": float, "start": int, "end": int}, ...]
    """
    if not text.strip():
        return []
    ner = load_ner_model()
    results = ner(text)
    return results


def format_ner_results(text: str, entities: list) -> str:
    """Pretty-print NER results with inline color highlighting."""
    if not entities:
        return text

    highlighted = ""
    prev_end = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        color = NER_COLORS.get(ent["entity_group"], "")
        highlighted += text[prev_end:start]
        highlighted += f"{color}[{ent['word']} | {ent['entity_group']}]{RESET}"
        prev_end = end
    highlighted += text[prev_end:]
    return highlighted


# ── Text normalization pipeline ───────────────────────────────────────────────

def convert_line(text: str) -> tuple:
    """One line through the number-normalization pipeline."""
    taken        = []
    replacements = []

    def overlaps(s, e):
        return any(not (e <= ts or s >= te) for ts, te in taken)

    for etype, pattern, converter in PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            result = converter(m)
            if result:
                replacements.append((s, e, m.group(0), result, etype))
                taken.append((s, e))

    replacements.sort(key=lambda x: x[0], reverse=True)
    output = text
    entities = []
    for s, e, original, converted, etype in replacements:
        output = output[:s] + converted + output[e:]
        entities.append({"original": original, "type": etype, "converted": converted})

    entities.sort(key=lambda x: text.find(x["original"]))
    return output, entities


# ── File I/O ──────────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str):
    # Read
    if not os.path.exists(input_path):
        print(f"❌  File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📂  Reading  : {input_path}  ({len(lines)} lines)")
    print()

    output_lines   = []
    total_norm     = 0
    total_entities = 0

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            output_lines.append("\n")
            continue

        print(f"── Line {i} " + "─" * 60)
        print(f"  Original : {stripped}")

        # ── STEP 1: NER on original text ──────────────────────────────
        ner_results = run_ner(stripped)
        total_entities += len(ner_results)

        if ner_results:
            print(f"  NER tags :")
            for ent in ner_results:
                score = f"{ent['score']:.2f}"
                print(f"    [{ent['entity_group']:6}] {score}  →  \"{ent['word']}\"")
            # Inline colored view
            print(f"  Highlighted: {format_ner_results(stripped, ner_results)}")
        else:
            print(f"  NER tags : (none)")

        # ── STEP 2: Number normalization ──────────────────────────────
        normalized, norm_entities = convert_line(stripped)
        total_norm += len(norm_entities)
        output_lines.append(normalized + "\n")

        if norm_entities:
            print(f"  Normalized : {normalized}")
            for e in norm_entities:
                print(f"    [{e['type']:8}]  {e['original']}  →  {e['converted']}")
        else:
            print(f"  Normalized : {normalized}  (no numbers found)")

        print()

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print("=" * 70)
    print(f"✅  Done!")
    print(f"    NER entities found    : {total_entities}")
    print(f"    Numbers converted     : {total_norm}")
    print(f"    Output written to     : {output_path}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if len(args) == 0:
        input_path  = "telugu input text.txt"
        output_path = "output.txt"

    elif len(args) == 1:
        input_path  = args[0]
        base, ext   = os.path.splitext(args[0])
        output_path = base + "_output" + ext

    elif len(args) == 2:
        input_path  = args[0]
        output_path = args[1]

    else:
        print("Usage:")
        print("  python telugu_ner_converter.py")
        print("  python telugu_ner_converter.py input.txt")
        print("  python telugu_ner_converter.py input.txt output.txt")
        sys.exit(1)

    process_file(input_path, output_path)


if __name__ == "__main__":
    main()
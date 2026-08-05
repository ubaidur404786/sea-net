"""
scripts/check_tex.py - a quick sanity check on the auto-generated LaTeX.

Why this exists
---------------
`python main.py paper` writes captions into results/paper_figures/*.tex. Those captions
are normal Python strings, so it is easy to leave a character in that LaTeX treats as a
command. The classic ones:

    _   starts a subscript      -> "Missing $ inserted"  (e.g. writing  mil_  not  mil\\_ )
    &   column separator        -> "Misplaced alignment tab"
    #   macro argument
    <   >   are legal, but with the `times` font they print as upside-down ! and ?
    \\geq \\leq  are maths commands and MUST sit inside $...$

One of these stops the whole build, and the error message points at a generated file you
did not write, which is confusing. So we check before compiling.

Run it from the project root:

    python scripts/check_tex.py

Exit code 0 = clean, 1 = something needs escaping in seanet/paper/*.py.
"""
import os
import re
import sys
import glob

# Where main.py paper writes its output.
PAPER_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "paper_figures")

# Anything between $ signs is maths, where these characters are perfectly legal.
MATH = re.compile(r"\$[^$]*\$")

# A special character that is NOT already escaped with a backslash.
UNESCAPED = re.compile(r"(?<!\\)([_&#<>])")

# A maths-only command sitting in plain text.
BARE_MATH_CMD = re.compile(r"(?<!\$)\\(geq|leq|alpha|beta|times|pm|approx|neq)(?![a-zA-Z])")


# A character LaTeX cannot typeset with our font, e.g. the "…" that shorten() uses.
NON_ASCII = re.compile(r"[^\x00-\x7f]")

# A table body row: cells joined by & and ending in \\
TABLE_ROW = re.compile(r"&.*\\\\\s*$")


def check_file(path):
    """Return a list of problem strings for one .tex file."""
    problems = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()

            is_caption = stripped.startswith("\\caption")
            is_row = bool(TABLE_ROW.search(stripped)) and not stripped.startswith("%")

            # Captions are prose; table rows are the model names and numbers. Both get
            # typeset, so both must be safe. File paths and \label keys never are, so an
            # underscore in those is completely fine and we skip them.
            if not (is_caption or is_row):
                continue

            text = MATH.sub("", line)          # drop the maths, keep the rest

            # Inside a table row "&" is the column separator, so it is allowed there.
            checked = text.replace("&", "") if is_row else text

            where = "a caption" if is_caption else "a table row"
            bad = sorted(set(UNESCAPED.findall(checked)))
            if bad:
                problems.append(f"  line {line_no}: unescaped {bad} in {where}")

            for m in BARE_MATH_CMD.finditer(text):
                problems.append(f"  line {line_no}: \\{m.group(1)} used outside $...$")

            odd = sorted(set(NON_ASCII.findall(text)))
            if odd:
                problems.append(f"  line {line_no}: non-ASCII {odd} in {where} "
                                f"(use \\dots{{}} instead of a literal ellipsis)")

    return problems


def main():
    if not os.path.isdir(PAPER_ROOT):
        print(f"{PAPER_ROOT} does not exist yet - run `python main.py paper` first.")
        return 1

    files = ([os.path.join(PAPER_ROOT, name) for name in ("figures.tex", "tables.tex")]
             + sorted(glob.glob(os.path.join(PAPER_ROOT, "tables", "*.tex"))))

    total = 0
    for path in files:
        if not os.path.exists(path):
            continue
        problems = check_file(path)
        if problems:
            total += len(problems)
            print(os.path.relpath(path))
            print("\n".join(problems))

    if total:
        print(f"\n{total} problem(s).")
        print("  captions   -> fix the strings in seanet/paper/figures.py "
              "or seanet/paper/figures_stats.py")
        print("  table rows -> fix _tex_cell() in seanet/paper/tables.py")
        print("Then run `python main.py paper` again to rewrite these files.")
        return 1

    print("LaTeX check passed - all captions are safe to compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

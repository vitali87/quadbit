#!/usr/bin/env bash
# Build paper.pdf + paper.tex from paper.md. Maps the math-Unicode glyphs the paper uses
# (arrows, approx, minus, times, Delta, micro) to LaTeX so any font renders them.
set -euo pipefail
cd "$(dirname "$0")"
# MacTeX installs here and is not on PATH by default; prepend only if present (portable elsewhere).
[ -d /Library/TeX/texbin ] && export PATH="/Library/TeX/texbin:$PATH"
HDR="$(mktemp)"
cat > "$HDR" <<'TEX'
\usepackage{newunicodechar}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{µ}{\textmu}
TEX
pandoc paper.md -o paper.tex --standalone -H "$HDR"
pandoc paper.md -o paper.pdf --pdf-engine=xelatex -V geometry:margin=1in \
  -V mainfont="STIX Two Text" -V monofont="Menlo" -H "$HDR"
rm -f "$HDR"
echo "built paper.pdf + paper.tex"

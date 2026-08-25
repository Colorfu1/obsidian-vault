#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <paper.pdf> [output.txt]" >&2
  exit 2
fi

input_pdf=$1
output_txt=${2:-"${input_pdf%.pdf}.txt"}

if [[ ! -f "$input_pdf" ]]; then
  echo "PDF not found: $input_pdf" >&2
  exit 1
fi

if ! command -v pdftotext >/dev/null 2>&1; then
  echo "pdftotext is required (usually provided by poppler-utils)." >&2
  exit 1
fi

temp_txt=$(mktemp)
trap 'rm -f "$temp_txt"' EXIT

pdftotext -layout "$input_pdf" "$temp_txt"

awk '
  BEGIN { page = 1; pending_page = 0; print "[Page 1]" }
  index($0, "\f") {
    gsub("\f", "")
    page += 1
    pending_page = 1
    if (length($0) > 0) {
      print "\n[Page " page "]"
      print
      pending_page = 0
    }
    next
  }
  pending_page && $0 ~ /[^[:space:]]/ {
    print "\n[Page " page "]"
    pending_page = 0
  }
  !pending_page { print }
' "$temp_txt" > "$output_txt"

echo "$output_txt"

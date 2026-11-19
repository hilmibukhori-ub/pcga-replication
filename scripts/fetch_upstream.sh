#!/bin/bash
# Clone the two upstream repositories at the commits frozen in the pre-registrations.
# Data hosted on the authors' Google Drive must be downloaded by hand; see data/README.md.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw

clone () {  # url dir commit
  if [ ! -d "data/raw/$2/.git" ]; then git clone "$1" "data/raw/$2"; fi
  git -C "data/raw/$2" checkout --quiet "$3"
  echo "$2 at $(git -C "data/raw/$2" rev-parse HEAD)"
}

clone https://github.com/fulifeng/Temporal_Relational_Stock_Ranking rsr_repo cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d
clone https://github.com/zhengzetao/RTGCN rtgcn_repo 617c3f53a58753251929a0cdbd1c12f7e3df8154

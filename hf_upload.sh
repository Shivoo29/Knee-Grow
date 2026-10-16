#!/bin/bash
# Uploads hf_stage/ to the private HF repo Shivoo29/rsna-knee-weights in 3 commits (weights first).
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1   # xet storage returned 401 mid-upload on the first try; use the classic LFS path
for d in weights corpora resume; do
  echo "$(date) uploading $d"
  .venv/bin/hf upload Shivoo29/rsna-knee-weights "hf_stage/$d" "$d" --repo-type model --private \
    --commit-message "PC backup: $d" --format quiet || { echo "$(date) FAILED $d"; exit 1; }
  echo "$(date) done $d"
done
echo UPLOAD_DONE

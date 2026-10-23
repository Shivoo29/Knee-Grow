cd ~/projects/knee/cpu_ideas
export HF_HUB_DISABLE_XET=1
[ -s ../.hf_token ] && export HF_TOKEN="$(cat ../.hf_token)"   # user-owned token wins when present
# waits for the user-owned token file, then for gated access, then runs (token only in this process environment)
until { [ -s ../.hf_token ] && export HF_TOKEN="$(cat ../.hf_token)"; ../mg_env/bin/python -c "from huggingface_hub import hf_hub_download as h; h(\"google/medgemma-1.5-4b-it\",\"config.json\")" 2>/dev/null; }; do sleep 60; done
../mg_env/bin/python -I medgemma_gold.py ../raw/rsna-knee-abnormality-detection.zip "oof/raptor_gold_cnxt_r2_f?_swa.npz" study_laterality_orientation.csv > medgemma.log 2>&1

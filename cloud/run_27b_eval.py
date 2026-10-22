import time, torch, gc
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Clear memory
gc.collect()
torch.cuda.empty_cache()

_tok = key_form.value['hf_token'].strip()

print("Loading google/medgemma-27b-it in 4-bit (NF4)...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)

t0 = time.time()
proc27 = AutoProcessor.from_pretrained('google/medgemma-27b-it', token=_tok)
model27 = AutoModelForImageTextToText.from_pretrained(
    'google/medgemma-27b-it',
    token=_tok,
    quantization_config=bnb_cfg,
    device_map='auto'
)
print(f"27b-it 4-bit loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

# Load dataset
data = np.load('/tmp/mg_test.npz')
imgs = data['imgs']
truth = data['truth']
ids = data['ids']

FINDINGS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's Cyst", 'Bone Contusion', 'Fracture'
]

FINDING_CONFIGS = {
    'ACL': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a definite ACL tear (grade 2 or complete rupture)? Answer Yes or No.'
    },
    'MCL': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a definite MCL tear or sprain (grade 2 or complete rupture)? Answer Yes or No.'
    },
    'Medial Meniscus': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a definite medial meniscal tear reaching an articular surface on ≥2 images? Answer Yes or No.'
    },
    'Lateral Meniscus': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a definite lateral meniscal tear reaching an articular surface on ≥2 images? Answer Yes or No.'
    },
    'Medial OA': {
        'slices': [4, 5, 6, 7],
        'header': 'Knee MRI, coronal fat-suppressed slices.',
        'question': 'Is there full-thickness cartilage loss or high-grade osteoarthritis in the medial compartment? Answer Yes or No.'
    },
    'Lateral OA': {
        'slices': [4, 5, 6, 7],
        'header': 'Knee MRI, coronal fat-suppressed slices.',
        'question': 'Is there lateral compartment osteoarthritis (cartilage loss, osteophytes, joint space narrowing)? Answer Yes or No.'
    },
    'PF OA': {
        'slices': [8, 9, 10, 11, 0, 1, 2, 3],
        'header': 'Knee MRI, axial and sagittal slices.',
        'question': 'Is there high-grade patellofemoral osteoarthritis or cartilage loss? Answer Yes or No.'
    },
    'Effusion': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a moderate to large joint effusion or hemarthrosis? Answer Yes or No.'
    },
    'Synovitis': {
        'slices': [8, 9, 10, 11, 0, 1, 2, 3],
        'header': 'Knee MRI, axial and sagittal slices.',
        'question': 'Is there synovial thickening or active synovitis? Answer Yes or No.'
    },
    "Baker's Cyst": {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': "Is there a Baker's cyst with fluid in the gastrocnemius-semimembranosus bursa? Answer Yes or No."
    },
    'Bone Contusion': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a bone contusion or bone marrow edema from acute trauma? Answer Yes or No.'
    },
    'Fracture': {
        'slices': [0, 1, 2, 3, 4, 5, 6, 7],
        'header': 'Knee MRI, sagittal and coronal fat-suppressed slices.',
        'question': 'Is there a fracture line, cortical disruption, or avulsion fragment? Answer Yes or No.'
    }
}

yes_id = proc27.tokenizer.encode('Yes', add_special_tokens=False)[0]
no_id = proc27.tokenizer.encode('No', add_special_tokens=False)[0]

num_studies = len(ids)
pred_probs27 = np.zeros((num_studies, 12), dtype=np.float32)
times27 = []

print(f"\n🚀 Running 27B-4bit evaluation on {num_studies} studies...")
for i in range(num_studies):
    t_st = time.time()
    for j, finding in enumerate(FINDINGS):
        cfg = FINDING_CONFIGS[finding]
        slice_indices = cfg['slices']
        pil_imgs = [Image.fromarray(imgs[i, s]).convert('RGB') for s in slice_indices]
        
        content = [{'type': 'text', 'text': cfg['header']}]
        for _ in pil_imgs:
            content.append({'type': 'image'})
        content.append({'type': 'text', 'text': cfg['question']})
        msgs = [{'role': 'user', 'content': content}]
        
        prompt = proc27.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = proc27(text=prompt, images=pil_imgs, return_tensors='pt').to(model27.device)
        
        with torch.no_grad():
            out = model27(**inputs)
            logits = out.logits[0, -1]
            pair = logits[[yes_id, no_id]].float()
            p = torch.softmax(pair, dim=0)[0].item()
        
        pred_probs27[i, j] = p
    
    dt = time.time() - t_st
    times27.append(dt)
    print(f"  [+] Study {i+1:2}/{num_studies} done in {dt:.2f}s | Lat OA: p={pred_probs27[i, 5]:.2f} (gt={truth[i, 5]}) | PF OA: p={pred_probs27[i, 6]:.2f} (gt={truth[i, 6]})", flush=True)

mean_sec27 = float(np.mean(times27))
total_sec27 = float(np.sum(times27))

# T4 speed extrapolation (5x slower)
t4_sec27 = mean_sec27 * 5.0
t4_hours27 = (1300 * t4_sec27) / 3600.0

auc27 = {}
for j, finding in enumerate(FINDINGS):
    y_true = truth[:, j]
    y_pred = pred_probs27[:, j]
    if len(np.unique(y_true)) > 1:
        try:
            auc27[finding] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc27[finding] = None
    else:
        auc27[finding] = None

macro_auc27 = float(np.mean([v for v in auc27.values() if v is not None]))

print("\n=== RESULTS FOR 27B-4bit ===")
print(f"Macro AUC: {macro_auc27:.4f}")
print(f"Speed: {mean_sec27:.2f}s per study ({mean_sec27/12:.2f}s per question)")
print(f"Total Time (12 studies): {total_sec27:.1f}s")
print(f"Extrapolated Kaggle T4 Time (1,300 studies): {t4_hours27:.2f} hours")
print("\nPer-finding AUC:")
for f in FINDINGS:
    a = auc27.get(f)
    a_str = f"{a:.4f}" if a is not None else "N/A"
    j = FINDINGS.index(f)
    print(f"  {f:20}: AUC = {a_str:8} | mean_p = {pred_probs27[:, j].mean():.3f}")

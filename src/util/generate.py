import time
import torch
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
from torchtune.training.checkpointing import FullModelMetaCheckpointer
from torchtune.data import Message
from torchtune.generation import generate
import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pickle
import csv


testing_size = 5000

# --- Paths ---
CHECKPOINT_DIR = "/scratch1/ac.zyang/LLM/llama3/output/llama3_8B/lora-with-pretraining-simplified-2-genes/epoch_5"
TEST_FILE = "/scratch1/ac.zyang/LLM/data/shuffled-fine-tuning-dataset-simplified-2-gene-test-dataset.jsonl"
OUTPUT_DIR = CHECKPOINT_DIR + "/../generation-result"

os.makedirs(OUTPUT_DIR, exist_ok=True)
SCORE_FILE = os.path.join(OUTPUT_DIR, "score.tsv")
GENERATION_RESULT = os.path.join(OUTPUT_DIR, "prediction-truth.tsv")


TOKENIZER_PATH = "/scratch1/ac.zyang/LLM/llama2/pre-training/torchtitan/assets/tokenizer/original/tokenizer.model"


# --- Prompt ---
system_prompt = """You are a genome annotation assistant. 
The user provides microbial genome data including taxonomy, environmental source, and a list of annotated genes. 
Your task is to predict the missing gene(s) at each <MASK> location and output them in order.

Rules:
- Each <MASK> must be replaced with the corresponding missing <GENE> line(s).
- Do not modify taxonomy, environment, or any <GENE> lines outside the <MASK>.
- Use known patterns in gene order, operon structure, taxonomic consistency, and gene function.
- the prediction gene must strictly follow this format:
  <GENE> cluster_id=<str> core=<True|False> mobile=<True|False> id=<str> func="<gene function description>"
- Use realistic and commonly observed microbial gene annotations; never invent fictitious domains or functions.
"""

def normalize(line: str | None) -> str:
    if line is None:
        return ""   # treat missing as empty string
    return " ".join(line.strip().split())  # collapse spaces

if __name__ == "__main__":
    print("Loading the model...")
    # --- Load tokenizer ---
    tok = llama3_tokenizer(path=TOKENIZER_PATH, max_seq_len=5000)

    # --- Load model ---
    checkpointer = FullModelMetaCheckpointer(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_files=["model-00001-of-00001.bin"],
        model_type="LLAMA3",
        output_dir="./"
    )
    state = checkpointer.load_checkpoint()
    model = llama3_8b()   # base architecture
    model.load_state_dict(state["model"])
    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    
    print("model is loaded")
    
    # --- Load Test File ---
    print("loading test file...")
    with open(TEST_FILE) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    samples = samples[:testing_size]

    pred, truth = [], []
    print("start generation")
    
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        user_prompt = sample['user']
        ground_truth = sample['assistant']
        
        # --- Build messages ---
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=""),  # needed to start generation
        ]

        # --- Encode ---
        encoded = tok({"messages": messages}, inference=True)
        prompt_tensor = torch.tensor(encoded["tokens"], device="cuda").unsqueeze(0)

        # --- Generation ---
        t0 = time.time()
        with torch.inference_mode():
            out_tokens, _ = generate(
                model,
                prompt_tensor,
                max_generated_tokens=1000,
                temperature=0.3,
                top_k=300,
                pad_id=tok.pad_id,
                stop_tokens=tok.stop_tokens,  # include EOS + special stops
            )
        t1 = time.time()

        decoded = tok.decode(out_tokens[0].tolist())

        # --- comparison ---
        predicted_gene = decoded[len(system_prompt)+len(user_prompt)-1:]
        pred.append(predicted_gene)
        truth.append(ground_truth)
        


    # normalize everything
    pred_norm = [normalize(p) for p in pred]
    truth_norm = [normalize(t) for t in truth]

    # exact-match accuracy
    acc = accuracy_score(truth_norm, pred_norm)

    # F1 (macro or micro depending on preference)
    f1 = f1_score(truth_norm, pred_norm, average="macro")  # or "macro"

    print(f"Accuracy = {acc:.2%}")
    print(f"F1 Score = {f1:.2%}")
    
    if len(pred[0].split("\n")) > 1:
        print("pred[0]", len(pred[0].split("\n")))
        pred_expanded = []
        for p in pred:
            p_expand = p.split("\n")
            if  len(p_expand) != 2:
                p_expand = [p_expand[0], ""]
            pred_expanded.extend(p_expand)
        truth_expanded = []
        for t in truth:
            truth_expanded.extend(t.split("\n"))

        expanded_acc = accuracy_score(truth_expanded, pred_expanded)
        expanded_f1 = f1_score(truth_expanded, pred_expanded, average="macro")  # or "macro"

        print(f"Accuracy = {expanded_acc:.2%}")
        print(f"F1 Score = {expanded_f1:.2%}")
    
    
    with open(GENERATION_RESULT, 'w') as result_f:
        # Create a csv.writer object, specifying the tab delimiter
        tsv_writer = csv.writer(result_f, delimiter='\t')
        tsv_writer.writerow(["Prediction", "True"])
        
        if len(pred[0].split("\n")) > 1:
            print("pred[0]", len(pred[0].split("\n")))
            for i in range(len(pred_expanded)):
                tsv_writer.writerow([pred_expanded[i], truth_expanded[i]])
        else:
            for i in range(len(pred)):
                tsv_writer.writerow([pred[i], truth[i]])
    
    with open(SCORE_FILE, 'w') as score_f:
        # Create a csv.writer object, specifying the tab delimiter
        tsv_writer = csv.writer(score_f, delimiter='\t')
        tsv_writer.writerow([" ", "Acc", "F1"])
        tsv_writer.writerow(["default", acc, f1])
        if len(pred[0].split("\n")) > 1:
            tsv_writer.writerow(["expanded", expanded_acc, expanded_f1])
    



    


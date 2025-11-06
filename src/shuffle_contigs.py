import os
import random
import re


"""
For each .txt file:

Removes contig=... from each <GENE> line.

Removes empty contigs (i.e., sections with no <GENE> lines).

Saves:

*_cleaned.txt: cleaned but unshuffled.

*_cleaned_shuffled.txt: cleaned and shuffled (random rotation of non-empty contigs).

Saves both files in a new folder.
"""



def remove_contig_field(line):
    if line.startswith("<GENE>"):
        return re.sub(r'contig=\S+\s+', '', line)
    return line

def process_and_shuffle(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read and clean lines
    with open(input_path, 'r') as f:
        lines = [remove_contig_field(line) for line in f.readlines()]

    taxonomy = []
    env = []
    contigs = []
    current_contig = []

    for line in lines:
        if line.startswith("<TAXONOMY>"):
            taxonomy.append(line)
        elif line.startswith("<ENV>"):
            env.append(line)
        elif line.startswith("<CONTIG_BREAK>"):
            # Save current contig if it has at least one <GENE>
            if any(l.startswith("<GENE>") for l in current_contig):
                contigs.append(current_contig)
            current_contig = []
        else:
            current_contig.append(line)

    # Final contig (if not empty)
    if any(l.startswith("<GENE>") for l in current_contig):
        contigs.append(current_contig)

    # Skip if all contigs are empty
    if not contigs:
        print(f"⚠️ No valid contigs found in {input_path}, skipping.")
        return

    # Output filenames
    filename = os.path.basename(input_path).replace(".txt", "")
    cleaned_output_path = os.path.join(output_dir, f"{filename}_cleaned.txt")
    shuffled_output_path = os.path.join(output_dir, f"{filename}_cleaned_shuffled.txt")

    # Step 2: Write cleaned (but not shuffled)
    with open(cleaned_output_path, 'w') as f:
        f.writelines(taxonomy + env)
        for c in contigs:
            f.write("<CONTIG_BREAK>\n")
            f.writelines(c)
    print(f" Saved cleaned: {cleaned_output_path}")

    # Step 3: Shuffle and save
    if len(contigs) >= 2:
        split = random.randint(1, len(contigs) - 1)
        shuffled = contigs[split:] + contigs[:split]

        with open(shuffled_output_path, 'w') as f:
            f.writelines(taxonomy + env)
            for c in shuffled:
                f.write("<CONTIG_BREAK>\n")
                f.writelines(c)
        print(f" Saved shuffled: {shuffled_output_path}")
    else:
        print(f" Skipped shuffling (only one valid contig): {input_path}")

# Example for batch processing all .txt files
source_dir = "/scratch1/ac.zyang/LLM/data/flat_dataset"
target_dir = "/scratch1/ac.zyang/LLM/data/shuffled_dataset"

for fname in os.listdir(source_dir):
    if fname.endswith(".txt"):
        fpath = os.path.join(source_dir, fname)
        process_and_shuffle(fpath, target_dir)
        print("\n")

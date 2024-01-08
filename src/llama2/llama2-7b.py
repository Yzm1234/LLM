import os
from pathlib import Path

import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
from src.util.utils import get_tuning_loss_plot

# add arg parser
parser = argparse.ArgumentParser(description="llama2 7b fine tune")
parser.add_argument("base_model_name", help="the name of the base model before fine-tuning")
parser.add_argument("refined_model_name", help="the name of the model after fine-tuning")
parser.add_argument("dataset_path", help="the name of the dateset")
parser.add_argument("output_folder", help="output folder path")
parser.add_argument("epoch", help="number of epoch")

global args
args = parser.parse_args()


if __name__ == "__main__":
    print("Loading model.....")
    # Model and tokenizer names
    base_model_name = args.model_name
    refined_model = args.refined_model_name #You can give it your own name

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    print("model loading is done.")
    # TODO: update lora config based on model
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    check_points_folder = os.path.join(args.output_folder, "check_points")
    Path(check_points_folder).mkdir(parents=True, exist_ok=True)
    # Training Params
    train_params = TrainingArguments(
        output_dir=check_points_folder,
        evaluation_strategy="steps",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # Load dataset
    print("Loading dataset")
    ds = load_from_disk(args.dataset_path)
    print("Dataset is loaded.")

    # Trainer
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params
    )

    # Training
    print("Start training...")
    trainer.train()

    # save model
    print("Saving fine-tuned model...")
    trainer.save_model(os.path.join(args.output_folder, refined_model))
    print("Fine-tuned model is saved.")

    # plot loss graph
    get_tuning_loss_plot(trainer.state.log_history, f"{args.output_folder}/{refined_model}")
    print("Training loss is saved.")

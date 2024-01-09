import matplotlib.pyplot as plt
from datasets import load_dataset,Features,Value
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from langchain.llms import HuggingFacePipeline


def split_dataset(data_path: str, output_file: str, test_size: float=0.2, random_seed: int=42):
    """
    split the datset into train and test set
    :param data_path: full dataset
    :type data_path: string
    :param output_file: output of splitted dataset
    :type output_file: string
    :param test_size: ration of test size
    :type test_size: float number
    :param random_seed: random seed used to split
    :type random_seed: int
    :return: splitted dataset
    :rtype: DatasetDict
    """
    ds = load_dataset("json", data_files=data_path, split='train', field='data')
    ds = ds.train_test_split(test_size=test_size, seed=random_seed)
    ds.save_to_disk(output_file)
    print(f"Splitted dataset is saved to {output_file}.")
    return ds


def get_tuning_loss_plot(log_hist, graph_name):
    epochs, train_loss, val_loss = [], [], []
    for i in range(len(log_hist)//2):
        epochs.append(log_hist[2*i]['epoch'])
        train_loss.append(log_hist[2*i]['loss'])
        val_loss.append(log_hist[2*i+1]['eval_loss'])

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.legend()
    plt.title("Model fine tune loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig(f"{graph_name}_tuning_loss.png")
    plt.show()


def model_inference(model_name, max_new_tokens=500):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.eval()
    # model genration config
    generation_config = model.generation_config
    generation_config.temperature = 1
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    return llm
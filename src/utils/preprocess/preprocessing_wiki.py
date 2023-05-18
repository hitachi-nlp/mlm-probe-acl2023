"""
References:  
    https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
"""
from typing import List

import datasets
from datasets import concatenate_datasets, load_dataset
from transformers import RobertaTokenizerFast

from sentencizer import Sentencizer


def remove_wiki_info(example):
    """Remove unnecessary texts in the wikipedia corpus."""
    keywords = ("See also", "References", "Category")
    for keyword in keywords:
        index = example["text"].find(keyword)
        if index != -1:
            example["text"] = example["text"][:index]
    return example


def split_into_sentence(examples):
    """Preprocess wikipedia corpus by each article.  
    
    References:
        https://huggingface.co/docs/datasets/v2.1.0/en/about_map_batch#map
    """
    sentencizer = Sentencizer()
    return {"text": sentencizer(examples["text"], n_jobs=-2, chunk_size=10)}


def generate_samples(examples, tokenizer):
    """Generate token sequence samples.  
    """
    text_list = []
    chunk = []
    for text in examples["text"]:
        # We try to fill up the chunk such that the number of tokens in it is 
        # close to 512. Then, we use the chunk as an input sequence.
        if chunk == []:
            num_tokens = len(text.split())
        else:
            temp_text = ' '.join(chunk) + ' ' + text
            num_tokens = len(temp_text.split())
        
        if num_tokens <= 512:
            # add to chunk
            chunk.append(text)
        else:
            if chunk != []:
                # remove blanks
                ret_text = ' '.join(chunk)
                ret_text = ret_text.strip().replace("\n", "")
                
                # add to lists
                if len(tokenizer(ret_text).input_ids) <= 512: # no truncation
                    # add to text_list
                    if len(ret_text.split()) > 128:
                        text_list.append(ret_text)
                    # update chunk
                    chunk = [text]
                    
                else:
                    # need to re-adjust
                    chunk_index = len(chunk) - 1
                    while True:
                        ret_text = ' '.join(chunk[:chunk_index])
                        if len(tokenizer(ret_text).input_ids) <= 512:
                            # add to text_list
                            if len(ret_text.split()) > 128:
                                text_list.append(ret_text)
                            # update chunk
                            chunk = chunk[chunk_index:]
                            chunk.append(text)
                            break
                        chunk_index -= 1
                        if chunk_index == 1:
                            # exclude a single sentence with over 512 tokens
                            chunk = chunk[chunk_index:]
                            chunk.append(text)
                            break
    
    return {"text": text_list}


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length')


def flatten_samples(examples):
    """Flatten a list of lists to a combined list."""
    return {'text': [sentence for article in examples['text'] for sentence in article if sentence != '']}


def preprocess_dataset(args, seed: int = 1234):
    """Preprocess a dataset for pre-training 
    """
    # preprocess wiki data
    wiki_dataset = load_dataset("wikipedia", "20220301.en")["train"]
    wiki_dataset = wiki_dataset.remove_columns(["id", "url", "title"]) # only keep the text based on the original BERT paper
    
    print("Removing unnecessary wiki data...")
    wiki_dataset = wiki_dataset.map(remove_wiki_info) # remove references etc.
    
    print("Preprocessing Wikipedia dataset...")
    print("\tSplit wiki articles into sentences...")
    for index in range(10):
        wiki_dataset_shard = wiki_dataset.shard(num_shards=10, index=index)
        wiki_dataset_shard = wiki_dataset_shard.map(lambda examples: split_into_sentence(examples), batched=True)
        wiki_dataset_shard.save_to_disk(f'{args.output_path}/{index}')
    wiki_dataset = []
    for index in range(10):
        wiki_dataset_shard = datasets.load_from_disk(f'{args.output_path}/{index}')
        if wiki_dataset == []:
            wiki_dataset = wiki_dataset_shard
        else:
            wiki_dataset = concatenate_datasets([wiki_dataset, wiki_dataset_shard])
    
    print("\tFlatten samples...")
    wiki_dataset = wiki_dataset.map(flatten_samples, batched=True, num_proc=4)
    
    print("\tGenerate samples...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    wiki_dataset = wiki_dataset.map(
        lambda examples: generate_samples(examples, tokenizer), 
        batched=True,
        num_proc=10
    )
    
    print("\tTokenise samples...")
    wiki_dataset = wiki_dataset.map(
        lambda examples: tokenize(examples, tokenizer), 
        num_proc=10, batched=True
    )
    
    print("\tRemoving texts...")
    wiki_dataset = wiki_dataset.remove_columns("text")
    
    print("Saving the processed data to disk...")
    wiki_dataset.save_to_disk(args.output_path)
    
    print("Done!")


def main(args):
    preprocess_dataset(args)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Preprocess Wikipedia dataset")
    parser.add_argument(
        "output_path", 
        help="(str) Path to an processed dataset directory")
    args = parser.parse_args()
    
    main(args)

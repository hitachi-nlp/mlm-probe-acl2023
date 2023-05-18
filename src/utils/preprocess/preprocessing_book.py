from typing import List

import datasets
from datasets import concatenate_datasets, load_dataset
from tqdm.contrib import tenumerate
from transformers import RobertaTokenizerFast


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length')


def preprocess_book(lines: List[str], tokenizer: RobertaTokenizerFast):
    """Preprocess book corpus by each line.  
    """    
    text_list = []
    chunk = []
    for index, text in tenumerate(lines):
        # We try to fill up the chunk such that the number of tokens in it is 
        # close to 512. Then, we use the chunk as an input sequence.
        if chunk == []:
            num_tokens = len(tokenizer(text).input_ids)
        else:
            temp_text = ' '.join(chunk) + ' ' + text
            num_tokens = len(tokenizer(temp_text).input_ids)
            
        if num_tokens <= 512:
            # add to chunk
            chunk.append(text)
        else:
            if chunk != []:
                # remove blanks
                ret_text = ' '.join(chunk)
                ret_text = ret_text.strip().replace("\n", "")
                
                # add to lists
                text_list.append(ret_text)
                
                # update chunk
                chunk = [text]

    return {"text": text_list}
 

def preprocess_dataset(args, seed:int=1234):
    """Preprocess a dataset for pre-training 
    """
    # init
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # preprocess bookcorpus data
    print("Preprocessing BookCorpus...")
    book_dataset = load_dataset("bookcorpus")
    book_dataset = preprocess_book(book_dataset["train"]["text"], tokenizer)

    # concat dataset
    print("Concat datasets...")
    book_dataset = datasets.Dataset.from_dict(book_dataset)
    
    # tokenize dataset
    print("Tokenise samples...")
    book_dataset = book_dataset.map(
        lambda examples: tokenize(examples, tokenizer), 
        num_proc=5, batched=True
    )
    
    print("Saving the processed data to disk...")
    book_dataset.save_to_disk(args.output_path)
    
    print("Done!")


def main(args):
    # preprocess datasets
    preprocess_dataset(args)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Preprocess BookCorpus")
    parser.add_argument(
        "output_path", 
        help="(str) Path to an processed dataset directory")
    args = parser.parse_args()
    
    main(args)

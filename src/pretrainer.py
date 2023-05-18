""""
References:
    https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    https://huggingface.co/docs/transformers/main_classes/callback
"""

from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from transformers import (AutoConfig, 
                          AutoTokenizer, 
                          HfArgumentParser,
                          RobertaForMaskedLM, 
                          Trainer, 
                          TrainingArguments,
                          set_seed)

from model import (RobertaForFirstCharPrediction, RobertaForNCharsPrediction)
from utils import (DataCollatorForFirstCharPrediction,
                   DataCollatorForEndCharPrediction,
                   DataCollatorForMaskedLanguageModeling,
                   DataCollatorForNCharsPrediction,
                   DataCollatorForTailNCharsPrediction,
                   LoggingCallback)

transformers.logging.set_verbosity_debug()


@dataclass
class AdditionalArguments:
    """Define additional arguments that are not included in `TrainingArguments`."""
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "A model name or path defined in the Transformers library."}
    )
    
    model_name: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "A model name in the Transformers library."}
    )
    
    tokenizer_path: Optional[str] = field(
        default="",
        metadata={"help": "A tokenizer path."}
    )
    
    task_type: Optional[str] = field(
        default="",
        metadata={"help": ("A type of a task. Choose one from "
                  "`first_char` (First Char), "
                  "`end_char` (Last Char), "
                  "`n_chars` (First n Chars),"
                  "`tail_n_chars` (Last n Chars), "
                  "`mlm` (MLM).")}
    )
    
    num_chars: Optional[int] = field(
        default=1,
        metadata={"help": "if task_type is `n_chars` or `tail_n_chars`, "
                  "specify the number of characters considered."}
    )
    
    book_data_dir: Optional[str] = field(
        default="",
        metadata={"help": "[train] Path to a pretraining BookCorpus directory."}
    )
    
    wiki_data_dir: Optional[str] = field(
        default="",
        metadata={"help": "[train] Path to a pretraining Wikipedia data directory."}
    )
    
    save_interval: Optional[float] = field(
        default=43200.0,
        metadata={"help": "An interval of saving weights in seconds."}
    )
    
    use_pretrained: Optional[bool] = field(
        default=False,
        metadata={"help": "Set this to use a pretrained model."}
    )


def build_dataset(wiki_data_dir: str = "", 
                  book_data_dir: str = "",
                  seed: int = 42) -> datasets.Dataset:
    """Build a dataset."""
    # build a dataset
    if book_data_dir != "":
        ## load a book corpus 
        book_dataset = datasets.load_from_disk(book_data_dir)
        book_dataset.remove_columns("text")
        ## load a wikipedia dataset
        wiki_dataset = datasets.load_from_disk(wiki_data_dir)
        dataset = datasets.concatenate_datasets([book_dataset, wiki_dataset])
    else:
        ## load a wikipedia dataset
        dataset = datasets.load_from_disk(wiki_data_dir)
    dataset.set_format(type='torch')
    dataset = dataset.shuffle(seed=seed)
    return dataset


def first_char_pred(args, training_args):    
    # settings
    set_seed(training_args.seed)
    if args.use_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if config.model_type == 'roberta':
        if args.use_pretrained:
            model = RobertaForFirstCharPrediction.from_pretrained(args.model_name_or_path)
        else:
            model = RobertaForFirstCharPrediction(config)
    else:
        raise ValueError('No such a `model_type`!')
    data_collator = DataCollatorForFirstCharPrediction(
        tokenizer=tokenizer, 
        mask_prob=0.15
    )
    
    # build a dataset
    dataset = build_dataset(args.wiki_data_dir, 
                            args.book_data_dir, 
                            training_args.seed)
    
    # setup trainer
    callbacks = [LoggingCallback(args.save_interval)]
    trainer = Trainer(
        model=model,                  
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # training
    trainer.train()


def end_char_pred(args, training_args):    
    # settings
    set_seed(training_args.seed)
    if args.use_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if config.model_type == 'roberta':
        if args.use_pretrained:
            model = RobertaForFirstCharPrediction.from_pretrained(args.model_name_or_path)
        else:
            model = RobertaForFirstCharPrediction(config)
    else:
        raise ValueError('No such a `model_type`!')
    data_collator = DataCollatorForEndCharPrediction(
        tokenizer=tokenizer, 
        mask_prob=0.15
    )
    
    # build a dataset
    dataset = build_dataset(args.wiki_data_dir, 
                            args.book_data_dir, 
                            training_args.seed)
    
    # setup trainer
    callbacks = [LoggingCallback(args.save_interval)]
    trainer = Trainer(
        model=model,                  
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # training
    trainer.train()


def n_chars_pred(args, training_args):    
    # set a random seed
    set_seed(training_args.seed)
    
    # get a tokenizer
    if args.use_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif args.tokenizer_path != "":
        print('Use a custom tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # get a data collator
    data_collator = DataCollatorForNCharsPrediction(
        tokenizer=tokenizer, 
        mask_prob=0.15,
        num_chars=args.num_chars,
        lang_type=args.lang_type
    )
    
    # get a config
    num_char_class = data_collator.get_num_class()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_char_class = num_char_class
    
    # build a model
    if config.model_type == 'roberta':
        if args.use_pretrained:
            model = RobertaForNCharsPrediction.from_pretrained(args.model_name_or_path)
        else:
            model = RobertaForNCharsPrediction(config)
    else:
        raise ValueError('No such a `model_type`!')
    
    # build a dataset
    dataset = build_dataset(args.wiki_data_dir, 
                            args.book_data_dir, 
                            training_args.seed)
    
    # setup trainer
    callbacks = [LoggingCallback(args.save_interval)]
    trainer = Trainer(
        model=model,                  
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # training
    trainer.train()


def tail_n_chars_pred(args, training_args):    
    # set a random seed
    set_seed(training_args.seed)
    
    # get a tokenizer
    if args.use_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif args.tokenizer_path != "":
        print('Use a custom tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    # get a data collator
    data_collator = DataCollatorForTailNCharsPrediction(
        tokenizer=tokenizer, 
        mask_prob=0.15,
        num_chars=args.num_chars,
        lang_type=args.lang_type
    )
    
    # get a config
    num_char_class = data_collator.get_num_class()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_char_class = num_char_class
    if config.model_type == 'roberta':
        if args.use_pretrained:
            model = RobertaForNCharsPrediction.from_pretrained(args.model_name_or_path)
        else:
            model = RobertaForNCharsPrediction(config)
    else:
        raise ValueError('No such a `model_type`!')
    
    # build a dataset
    dataset = build_dataset(args.wiki_data_dir, 
                            args.book_data_dir, 
                            training_args.seed)
    
    # setup trainer
    callbacks = [LoggingCallback(args.save_interval)]
    trainer = Trainer(
        model=model,                  
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # training
    trainer.train()


def mlm(args, training_args):    
    # set a random seed
    set_seed(training_args.seed)
    
    # get a tokenizer
    if args.use_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif args.tokenizer_path != "":
        print('Use a custom tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    # get a data collator
    data_collator = DataCollatorForMaskedLanguageModeling(
        tokenizer=tokenizer, 
        mlm_prob=0.15
    )
    
    # get a config
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # build a model
    if config.model_type == 'roberta':
        if args.use_pretrained:
            model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
        else:
            model = RobertaForMaskedLM(config)
    else:    
        raise ValueError('No such a `model_type`!')
    
    # build a dataset
    dataset = build_dataset(args.wiki_data_dir, 
                            args.book_data_dir, 
                            training_args.seed)
    
    # setup a trainer
    callbacks = [LoggingCallback(args.save_interval)]
    trainer = Trainer(
        model=model,                  
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # training
    trainer.train()


if __name__ == '__main__':
    parser = HfArgumentParser((AdditionalArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info(f'Training args: {training_args}')
    logger.info(f'Additional self-defined args: {args}')
    
    if args.task_type == 'first_char':
        first_char_pred(args, training_args)
    elif args.task_type == 'end_char':
        end_char_pred(args, training_args)
    elif args.task_type == 'n_chars':
        n_chars_pred(args, training_args)
    elif args.task_type == 'tail_n_chars':
        tail_n_chars_pred(args, training_args)
    elif args.task_type == 'mlm':
        mlm(args, training_args)
    else:
        raise ValueError('No such a `args.task_type`!')

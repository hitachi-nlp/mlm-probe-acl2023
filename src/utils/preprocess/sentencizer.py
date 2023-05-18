from typing import List

import spacy
from transformers import RobertaTokenizerFast

from joblib import Parallel, delayed


class Sentencizer:
    """Sentencizer compatible with multiprocessing.  
    References: 
        https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiprocess.html#Option-2:-Use-nlp.pipe
        https://joblib.readthedcs.io/en/latest/generated/joblib.Parallel.html
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe('sentencizer')
    

    def __call__(self, articles:List[str], n_jobs:int=5, chunk_size:int=10):
        """
        Args:
            articles (List[str]): Wikipedia article data.
            n_jobs (int): how many jobs will you run simultaneously?
            chunk_size (int): how many articles per process?
        
        Returns:
            List[List[str]]: sentenceized articles
        """
        if len(articles) < chunk_size:
            chunk_size = len(articles)
        print(f"Estimated number of tasks: {int(len(articles) / chunk_size)}")
        return self.parallel_processor(articles, n_jobs, chunk_size)

    
    def flatten_list(self, lists:List[List[List[str]]]) -> List[List[str]]:
        """Flatten a list of lists to a combined list."""
        return [item for sublist in lists for item in sublist]
    

    def chunker(self, iterable:List[str], total_length:int, chunk_size:int):
        """Chunking raw data."""
        return (iterable[pos: pos + chunk_size] for pos in range(0, total_length, chunk_size))
    

    def split_article_into_sentences(self, articles:List[str]):
        """Split each article into sentences using spaCy."""
        ret = []
        for article in self.nlp.pipe(articles, batch_size=50):
            ret.append([sent.text.strip() 
                        for sent in list(article.sents)])
        return ret
    

    def parallel_processor(self, articles:List[str], n_jobs:int=5, chunk_size:int=1000):
        executor = Parallel(n_jobs=n_jobs, verbose=10)
        do = delayed(self.split_article_into_sentences)
        tasks = (do(chunk) for chunk in self.chunker(articles, len(articles), chunk_size=chunk_size))
        ret = executor(tasks)
        return self.flatten_list(ret)


class JapaneseSentencizer:
    """Sentencizer compatible with multiprocessing.  
    References: 
        https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiprocess.html#Option-2:-Use-nlp.pipe
        https://joblib.readthedcs.io/en/latest/generated/joblib.Parallel.html
    """
    def __init__(self):
        self.nlp = spacy.load('ja_core_news_trf', disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe('sentencizer')
    

    def __call__(self, articles:List[str], n_jobs:int=5, chunk_size:int=10):
        """
        Args:
            articles (List[str]): Wikipedia article data.
            n_jobs (int): how many jobs will you run simultaneously?
            chunk_size (int): how many articles per process?
        
        Returns:
            List[List[str]]: sentenceized articles
        """
        if len(articles) < chunk_size:
            chunk_size = len(articles)
        print(f"Estimated number of tasks: {int(len(articles) / chunk_size)}")
        return self.parallel_processor(articles, n_jobs, chunk_size)

    
    def flatten_list(self, lists:List[List[List[str]]]) -> List[List[str]]:
        """Flatten a list of lists to a combined list."""
        return [item for sublist in lists for item in sublist]
    

    def chunker(self, iterable:List[str], total_length:int, chunk_size:int):
        """Chunking raw data."""
        return (iterable[pos: pos + chunk_size] for pos in range(0, total_length, chunk_size))
    

    def split_article_into_sentences(self, articles:List[str]):
        """Split each article into sentences using spaCy."""
        ret = []
        for article in self.nlp.pipe(articles, batch_size=50):
            ret.append([sent.text.strip() 
                        for sent in list(article.sents)])
        return ret
    

    def parallel_processor(self, articles:List[str], n_jobs:int=5, chunk_size:int=1000):
        executor = Parallel(n_jobs=n_jobs, verbose=10)
        do = delayed(self.split_article_into_sentences)
        tasks = (do(chunk) for chunk in self.chunker(articles, len(articles), chunk_size=chunk_size))
        ret = executor(tasks)
        return self.flatten_list(ret)


class SentenceTailor:
    """Concatenate sentences, while ensuring each concatenated sentence has less than 512 tokens.  
    """
    def __call__(self, sentences: List[str], n_jobs:int=5):
        """
        Args:
            sentences (List[str]): Wikipedia article data.
            n_jobs (int): how many jobs will you run simultaneously?
        
        Returns:
            List[str]: sentenceized articles
        """
        return self.parallel_processor(sentences, n_jobs)

    
    def flatten_list(self, lists: List[List[str]]) -> List[str]:
        """Flatten a list of lists to a combined list."""
        return [item for sublist in lists for item in sublist]
    

    def concatenate_sentences(self, sentences: List[str]) -> List[str]:
        """Utility function to concatenate sentences into str, while ensuring 
        the number of tokens should be less than 512."""
        text_list = []
        chunk = []
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        
        for text in sentences:
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
                    if len(tokenizer(ret_text).input_ids) <= 512:
                        # add to text_list
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

        return text_list
    

    def parallel_processor(self, articles: List[str], n_jobs:int=5):
        ret = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.concatenate_sentences)(articles)
        )
        return self.flatten_list(ret)
        
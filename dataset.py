import random
from typing import List
import torch
from torch import nn, Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from itertools import chain
from typing import List, Optional
from transformers import LlamaTokenizer
from transformers.testing_utils import CaptureLogger
from transformers.utils.logging import get_logger

# Load and batch data 
def data_process(vocab, tokenizer, raw_text_iter):
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(
    data: Tensor, 
    bsz: int, 
    setting: str, 
    min_len: int = 10, 
    max_len: int = 256,
    bptt: int = 10, 
):
    # # Divide the dataset into ``bsz`` parts.
    # nbatch = data.size(0) // bsz
    # # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # data = data.narrow(0, 0, nbatch * bsz) # tensor(#tokens)
    # # Evenly divide the data across the ``bsz` batches.
    # data = data.view(bsz, -1).t().contiguous() # (N X bsz)
    # return data.cuda(0)
    
    # List to store sentence tensors
    sentences = []
    data_len = data.size(0)
    i = 0
    # Loop through the 'data' to create sentences with random lengths
    while i < data_len:
        # Generate a random length for the sentence
        if setting != 'identical':
            rand_length = min(random.randint(min_len, max_len), data_len - i)
        else:
            rand_length = bsz
        # Slice the data to get a "sentence" of 'rand_length'
        sentence = data[i:i+rand_length]
        # Add the tensor representing the sentence to the list
        sentences.append(sentence)
        # Increment 'i' to the next starting point
        i += rand_length

    if setting == 'variant':
        sentences.sort(key=lambda x: x.size(0))
        # Split sentences into two halves
        mid_index = len(sentences) // 2
        short_sentences = sentences[:mid_index]
        long_sentences = sentences[mid_index:]
        # Rearrange sentences in groups of bptt
        sentences = []
        for i in range(0, max(len(short_sentences), len(long_sentences)), bptt):
            sentences.extend(short_sentences[i:i+bptt])
            sentences.extend(long_sentences[i:i+bptt])
        # print(f"Sent lengths: {[len(s) for s in sentences]}")
    
    return sentences
        

def get_data( 
    block_size: int = 128, 
    setting: str = 'identical', 
    min_len: int = 10, 
    max_len: int = 128,
    bptt: int = 10,
):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(vocab, tokenizer, train_iter) # tensor(#train_tokens)
    val_data = data_process(vocab, tokenizer, val_iter) # tensor(#val_tokens)
    test_data = data_process(vocab, tokenizer, test_iter) # tensor(#test_tokens)

    train_data = batchify(train_data, block_size, setting, min_len, max_len, bptt)
    val_data = batchify(val_data, block_size, setting, min_len, max_len, bptt)
    test_data = batchify(test_data, block_size, setting, min_len, max_len, bptt)
    
    return train_data, val_data, test_data, vocab


class SentencePairDataset(Dataset):
    def __init__(self, data_list: List[Tensor], setting: str):
        """
        Args:
            data_list (list of Tensors): A list where each element is a tensor corresponding to a sentence.
        """
        self.data_list = data_list
        self.setting = setting

    def __len__(self):
        # We return the length minus one because we are creating pairs of sentences
        return len(self.data_list) - 1

    def __getitem__(self, idx: int):
        # Return the current sentence and the next sentence as the data and target, respectively.
        data = self.data_list[idx]
        target = self.data_list[idx + 1]
        if self.setting == 'variant':
            return data, target[:data.size(0)]
        return data, target
    


class DGDataset:
    def __init__(
        self, 
        dataset: str = "Anthropic/hh-rlhf",
        task: str = "clm",
        tokenizer: LlamaTokenizer = None,
        max_source_length: int = 1024,
        max_target_length: int = 1024,
        padding: str = "max_length",
        ignore_pad_token_for_loss: bool = True,
        preprocessing_num_workers: int = None,
        overwrite_cache: bool = True,
    ):
        self.dataset = dataset
        self.task = task
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        self.tok_logger = get_logger("transformers.tokenization_utils_base")
        self.sp_token = '<SEP>' if self.task == 'seq2seq' else ' </s><s>[INST] '


    def prepare_context(self, instance: dict):
        if self.dataset == 'blended_skill_talk':
            num_entries = len(instance["free_messages"])
            total_entries = num_entries
            if self.task == 'seq2seq':
                persona_pieces = f"<PS>{instance['personas'][1]}"
                if instance['context'] == "wizard_of_wikipedia":
                    additional_context_pieces = f"<CTX>{instance['additional_context']}."
                else:
                    additional_context_pieces = ""
                context = persona_pieces + additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = [sent for sent in instance["previous_utterance"] if sent != '']
            conversations = []

        elif self.dataset == 'conv_ai_2':
            total_entries = len(instance['dialog'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                user_profile = ' '.join([''.join(x) for x in instance['user_profile']])
                persona_pieces = f"<PS>{user_profile}"
                context = persona_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []
            conversations = []

        elif self.dataset == 'empathetic_dialogues':
            total_entries = len(instance['dialog'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                persona_pieces = f"<PS>{instance['prompt']}"
                additional_context_pieces = f"<CTX>{instance['context']}."
                context = persona_pieces + additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []
            conversations = []

        elif self.dataset == 'AlekseyKorshuk/persona-chat':
            total_entries = len(instance['utterances'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                user_profile = ' '.join(instance['personality'])
                persona_pieces = f"<PS>{user_profile}"
                context = persona_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []
            conversations = []
            
        elif self.dataset == 'Anthropic/hh-rlhf':
            dialogue_list = [dialogue.strip() for dialogue in instance['chosen'].split('\n\n') if dialogue.strip() != '']
            context = ''
            prev_utt_pc = []
            conversations = []
            for dialogue in dialogue_list:
                if dialogue.startswith('Human'): 
                    conversations.append(dialogue.lstrip('Human:').strip())
                elif dialogue.startswith('Assistant'):
                    conversations.append(dialogue.lstrip('Assistant:').strip())
                else:
                    conversations[-1] += '\n\n' + dialogue
            total_entries = len(conversations)
            num_entries = total_entries//2
        else:
            raise ValueError("Dataset not supported.")
        
        return num_entries, total_entries, context, prev_utt_pc, conversations


    def prepare_entry(
        self, 
        instance: dict, 
        entry_idx: int, 
        context: str, 
        prev_utt_pc: List[str], 
        total_entries: int,
        conversations: List[str],
    ):
        if self.dataset == 'blended_skill_talk':
            free_message = instance['free_messages'][entry_idx]
            guided_message = instance['guided_messages'][entry_idx]
            references = [values[entry_idx] for key, values in instance['suggestions'].items()]

        elif self.dataset == 'conv_ai_2':
            free_message = instance['dialog'][entry_idx*2]['text']
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx*2+1]['text']
            references = []

        elif self.dataset == 'empathetic_dialogues':
            free_message = instance['dialog'][entry_idx*2]['text']
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx*2+1]['text']
            references = []

        elif self.dataset == 'AlekseyKorshuk/persona-chat':
            free_message = instance['utterances'][entry_idx*2]['history'][-1]
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['utterances'][entry_idx*2+1]['history'][-1]
            references = instance['utterances'][entry_idx*2]['candidates']
            
        elif self.dataset == 'Anthropic/hh-rlhf':
            free_message = conversations[entry_idx*2]
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = conversations[entry_idx*2+1]
            references = []
            
        else:
            raise ValueError("Dataset not supported.")

        if not prev_utt_pc:
            original_context = context
        else:
            if not context.strip():
                original_context = self.sp_token.join(prev_utt_pc)
            else:
                original_context = context + self.sp_token + self.sp_token.join(prev_utt_pc)
        
        references.append(guided_message)
        return free_message, guided_message, original_context, references
    
    
    def process_dataset(self, instance: dict):
        num_entries, total_entries, context, prev_utt_pc, conversations = self.prepare_context(instance)
        inputs, labels = [], []
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance, 
                entry_idx, 
                context, 
                prev_utt_pc,
                total_entries,
                conversations,
            )
            if guided_message is None:
                continue
            # Input & Output
            if not original_context.strip():
                text = free_message + ' [/INST] '
            else:
                text = original_context + self.sp_token + free_message + ' [/INST] '

            inputs.append(text)
            labels.append(guided_message)
            prev_utt_pc.append(free_message + ' [/INST] ' + guided_message)
            
        return {
            "query": inputs,
            "reference": labels,
        }


    def tokenize_and_align_labels(self, instance: dict):
        processed_instance = self.process_dataset(instance)
        inputs = processed_instance["query"]
        labels = processed_instance["reference"]
        
        if not inputs:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        if self.task == 'seq2seq':
            inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(labels, max_length=self.max_target_length, padding=self.padding, truncation=True)
            
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 
            # when we want to ignore padding in the loss.
            if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            inputs["labels"] = labels["input_ids"]
            return inputs
        else:
            with CaptureLogger(self.tok_logger) as cl:
                inputs = self.tokenizer(
                    inputs, 
                    return_tensors="pt",
                    max_length=self.max_source_length, 
                    padding=self.padding, 
                    truncation=True,
                )
                labels = self.tokenizer(
                    labels, 
                    return_tensors="pt",
                    max_length=self.max_target_length, 
                    padding=self.padding, 
                    truncation=True,
                )
                
            new_inputs = inputs.copy()
            for k, v1 in inputs.items():
                v2 = labels[k]
                new_inputs[k] = torch.cat((v1, v2), dim=1)
                
            new_labels = torch.cat((-100*torch.ones_like(inputs["input_ids"]), labels["input_ids"]), dim=1)
            new_inputs["labels"] = new_labels

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                self.tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return new_inputs


    def group_texts(self, examples):
        # ['input_ids', 'attention_mask', 'labels']
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        return concatenated_examples


    def group_ED(self, dataset: datasets.Dataset):
        results = {
            'conv_id': [], 
            'prompt': [],
            'dialog': [], 
            'context': [],
        }
        for i, instance in enumerate(dataset):
            if instance['utterance_idx'] == 1:
                results['conv_id'].append(instance['conv_id'])
                results['dialog'].append([])
                results['prompt'].append(instance['prompt'])
                results['context'].append(instance['context'])

            response = {'text': instance['utterance'], 'speaker_idx': instance['speaker_idx']}
            results['dialog'][-1].append(response)
        return datasets.Dataset.from_dict(results)


    def preprocess(self, dataset: datasets.Dataset) -> datasets.Dataset:
        if self.dataset == "empathetic_dialogues":
            dataset = self.group_ED(dataset)

        dataset = dataset.map(
            self.process_dataset,
            batched=False,
            num_proc=self.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not self.overwrite_cache,
        )
        print("processed dataset: ", dataset)
        print("processed dataset[0]: ", dataset[0])
        dataset = dataset.map(
            self.group_texts,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
        )
        return dataset

    


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import pdb
    import os
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    data_names = [
        # ("conv_ai_2", None),
        # ("empathetic_dialogues", None),
        # ("AlekseyKorshuk/persona-chat", None),
        # ("blended_skill_talk", None),
        ("Anthropic/hh-rlhf", "harmless-base"),
    ]
    task = "clm" # "seq2seq"
    max_length = 1024
    base_dir = '/data/yuli/LM-profiling/data'
    os.makedirs(base_dir, exist_ok=True)
    
    for data_name, data_dir in data_names:
        for split in ["train", "test"]:
            train_dataset = load_dataset(data_name, data_dir=data_dir)[split]
            dg = DGDataset(
                dataset=data_name,
                task=task,
                tokenizer=tokenizer,
                max_source_length=max_length,
                max_target_length=max_length,
            )
            print('{}: {}'.format(data_name, train_dataset))
            train_dataset = dg.preprocess(train_dataset)
            print("processed dataset: ", train_dataset)
            print("processed dataset[0]: ", train_dataset[0])
            # Save the processed dataset
            train_dataset.to_json(f'{base_dir}/{data_name}-{split}.json')
            # pdb.set_trace()


        
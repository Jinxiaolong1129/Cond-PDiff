import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset

descriptions_dict = {
    "cola": '''CoLA (The Corpus of Linguistic Acceptability):
Description: This dataset consists of English sentences labeled as grammatically correct or incorrect. It's designed to evaluate a model's ability to understand English grammar.
Example: Sentence: "The cat sat on the mat." Label: Correct. Sentence: "On the mat sat cat." Label: Incorrect.''',

    "sst2": '''SST-2 (The Stanford Sentiment Treebank):
Description: This dataset includes sentences from movie reviews and their sentiment labels (positive or negative). It tests a model's ability to capture sentiment from text.
Example: Sentence: "The movie was fantastic!" Label: Positive. Sentence: "I did not enjoy the film at all." Label: Negative.''',

    "rte": '''RTE (Recognizing Textual Entailment):
Description: This task involves pairs of sentences and asks whether the second sentence is true (entails), false, or undetermined based on the information in the first sentence.
Example: Sentence 1: "The cat sat on the mat." Sentence 2: "There is a cat on the mat." Label: Entailment.''',

    "mnli":'''MNLI (Multi-Genre Natural Language Inference):
Description: This dataset involves sentence pairs drawn from multiple genres of text. The task is to predict whether the second sentence logically follows from the first one, contradicts it, or is unrelated (neutral).
Example: Sentence 1: "A soccer game with multiple males playing." Sentence 2: "Some men are playing a sport." Label: Entailment.''',

    "qnli":'''QNLI (Question Natural Language Inference):
Description: This task is derived from the Stanford Question Answering Dataset. It involves pairs of a question and a sentence, where the goal is to determine whether the sentence contains the answer to the question.
Example: Question: "What color is the sky?" Sentence: "The sky is usually blue." Label: Entailment.''',
    
    "qqp": '''Quora Question Pairs.Description: This dataset consists of pairs of questions from the Quora website, and the task is to determine whether the questions are semantically equivalent. 
    It challenges a model's ability to understand and compare the meaning of entire questions.
Example 1: "How can I be a good geologist?" / "What should I do to be a great geologist?" -> Equivalent.
Example 2: "What is the best time to visit New York?" / "What are the best things to do in Tokyo?" -> Not Equivalent..''',

    "mrpc": '''Microsoft Research Paraphrase Corpus. Task: Identify if sentences are paraphrases of each other.
Example 1: "The storm left a wake of destruction." / "Destruction was left by the storm." -> Paraphrase.
Example 2: "He says that he saw the man leave." / "He says the man stayed in." -> Not Paraphrase.''',

    'stsb': '''Semantic Textual Similarity Benchmark. Task: Rate sentence pair similarity on a 0-5 scale.
Example 1: "A man is playing a guitar." / "A man is playing an instrument." -> Score: 4.5.
Example 2: "A child is riding a horse." / "A horse is being ridden by a child." -> Score: 5.'''
}

class Multi_WeightDataset_norm(Dataset):
    def __init__(self, data_dict, matching_dirs_dict, positions, shapes):
        self.data = []
        self.norm_params = {}
        self.dataset_names = list(data_dict.keys())
        
                # Create a mapping from dataset name to an index
        self.dataset_name_to_index = {
            name: i for i, name in enumerate(self.dataset_names)
        }
        
        self.positions_dict = positions # matrix 中 indice 中对应的 weight flatten 顺序
        self.shapes = shapes
        # Preprocess and store normalized data
        for dataset_name, dataset_matrix in data_dict.items():
            mean, std = self.calculate_normalization_params(dataset_matrix)
            self.norm_params[dataset_name] = {'mean': mean, 'std': std}
            
            for index, data_row in enumerate(dataset_matrix):
                normalized_data_row = self.normalize_data(data_row, mean, std)
                data_entry = {
                    'data': normalized_data_row,
                    'dataset': dataset_name, # NOTE add orignal data here
                    'condition': self.dataset_name_to_index[dataset_name],
                    'checkpoint_dir': matching_dirs_dict[dataset_name][index],
                    'index': index,
                    "position": self.positions_dict[dataset_name][index],
                    "shape": self.shapes[dataset_name],
                    "description": descriptions_dict[dataset_name]
                }
                self.data.append(data_entry)

    def calculate_normalization_params(self, dataset_matrix):
        dataset_data = torch.tensor(dataset_matrix, dtype=torch.float32)
        mean = torch.mean(dataset_data, dim=0)
        std = torch.std(dataset_data, dim=0)
        return mean, std

    def normalize_data(self, data_row, mean, std):
        return (torch.tensor(data_row, dtype=torch.float32) - mean) / std


    def denormalize_data(self, dataset_name, normalized_data):
        mean = self.norm_params[dataset_name]['mean']
        std = self.norm_params[dataset_name]['std']
        
        mean = mean.to(normalized_data.device)
        std = std.to(normalized_data.device)
    
        return (normalized_data * std) + mean
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_GLUE_eval_from_each(self, num=2):
        """
        Retrieve num items from each dataset.

        :return: A list of dictionaries, each with data, label, dataset name, and directory.
        """
        output = []
        count_dict = {name: 0 for name in self.dataset_names}
        for item in self.data:
            if count_dict[item['dataset']] < num:
                output.append(item)
                count_dict[item['dataset']] += 1
            if all(count >= num for count in count_dict.values()):
                break
        return output


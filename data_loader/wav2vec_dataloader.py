import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.logger import logger
from os.path import join


class Wav2VecDataset(Dataset):
    def __init__(self, filepaths: list, scores: list):
        self.filepaths = filepaths
        self.scores = scores

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)
        score = self.scores[idx]
        return embedding, score


def Wav2VecCollateFunction(data):
    """
       data: is a list of tuples with (feature, score)
             where 'feature' is a tensor of arbitrary shape
             and score is a scalar
    """
    features = []
    scores = []
    for feature, score in data:
        features.append(feature)
        scores.append(score)

    features = pad_sequence([f.squeeze() for f in features], batch_first=True)
    scores = torch.tensor(scores)
    return features, scores


class Wav2VecDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        self.train_metadata = join(data_dir, metadata_file) 
        self.val_metadata = join(data_dir, val_metadata_file)

        file_content = self._read_file_content(self.train_metadata, ignore_header=True)       
        train_filepaths = [str(self.data_dir + "/" + self.emb_dir + "/" + line.split(",")[0]) for line in file_content]       
        train_targets    = [int(line.split(",")[1]) for line in file_content]

        logger.info("Dataset {} training files loaded".format(len(train_filepaths)))

        self.dataset = Wav2VecDataset(train_filepaths, train_targets)

        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, num_workers=0, collate_fn=Wav2VecCollateFunction)


    def _read_file_content(self, filepath, ignore_header=False):
        with open(filepath, "r") as f:
            if ignore_header:
                content = f.readlines()[1:]
            else:
                content = f.readlines()
                
        return content


    def get_val_dataloader(self):

        file_content = self._read_file_content(self.val_metadata, ignore_header=True)
        val_filepaths = [str(self.data_dir + "/" + self.emb_dir + "/" + line.split(",")[0]) for line in file_content]
        val_targets    = [int(line.split(",")[1]) for line in file_content]

        logger.info("Dataset {} validating files loaded".format(len(val_filepaths)))

        self.val_dataset = Wav2VecDataset(val_filepaths, val_targets)

        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0, collate_fn=Wav2VecCollateFunction)

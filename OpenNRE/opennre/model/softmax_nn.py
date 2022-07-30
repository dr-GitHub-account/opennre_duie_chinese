import torch
from torch import nn, optim
from .base_model import SentenceRE

class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        # 对于BERTEncoder，是nn.Linear(768, num_class)
        # 对于BERTEncoder，是nn.Linear(768 * 2, num_class)
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # 对于BERTEncoder，rep维度是(batch_size, 768)
        # 对于BERTEntityEncoder，rep维度是(batch_size, 768 * 2)
        rep = self.sentence_encoder(*args) # (B, H)
        rep = self.drop(rep)
        # 对于BERTEncoder，self.fc是nn.Linear(768, num_class)
        # 对于BERTEncoder，self.fc是nn.Linear(768 * 2, num_class)
        logits = self.fc(rep) # (B, N)
        return logits

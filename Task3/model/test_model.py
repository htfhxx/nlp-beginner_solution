import torch

class test_model(torch.nn.model):
    def __init__(self):
        super(test_model,self).__init__()

    def forward(self,premiese,premises_lengths,hypothesis,hypothesis_lengths):




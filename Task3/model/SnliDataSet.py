import torch
from torch.utils.data import Dataset

'''torch.utils.data    Dataset'''
class SnliDataSet(Dataset):
    def __init__(self,data,max_premises_len=None,max_hypothesis_len=None):
        #序列长度
        self.num_sequence=len(data["premises"])
        
        #创建tensor矩阵的尺寸
        self.premises_len=[len(seq) for seq in data["premises"]]
        self.max_premises_len=max_premises_len
        if self.max_premises_len is None:
            self.max_premises_len=max(self.premises_len)
        
        self.hypothesis_len=[len(seq) for seq in data["hypothesis"]]
        self.max_hypothesis_len=max_hypothesis_len
        if max_hypothesis_len is None:
            self.max_hypothesis_len=max(self.hypothesis_len)
#         print(self.num_sequence, self.max_premises_len)
#         print(self.num_sequence, self.max_hypothesis_len)
        #转成tensor，封装到data里
        self.data= {
            "premises":torch.zeros((self.num_sequence,self.max_premises_len),dtype=torch.long),
            "hypothesis":torch.zeros((self.num_sequence,self.max_hypothesis_len),dtype=torch.long),
            "labels":torch.tensor(data["labels"])
        }
        
        for i,premises in enumerate(data["premises"]):
            l=len(data["premises"][i])
            self.data["premises"][i][:l]=torch.tensor(data["premises"][i][:l])
            l2=len(data["hypothesis"][i])
            self.data["hypothesis"][i][:l2]=torch.tensor(data["hypothesis"][i][:l2])
        
        
    def __len__(self):
        return self.num_sequence
        
    def __getitem__(self,index):
        return { "premises": self.data["premises"][index],
                    "premises_len":min(self.premises_len[index], self.max_premises_len),
                    "hypothesis":self.data["hypothesis"][index],
                    "hypothesis_len":min(self.hypothesis_len[index], self.max_hypothesis_len),
                    "labels":self.data["labels"][index]   }
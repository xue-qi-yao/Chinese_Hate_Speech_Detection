from transformers import BertTokenizer, BertForSequenceClassification
from utils import read_csv_tsv_expert, load_aligner
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from clip_aligner import CLIPAligner
import json
import h5py
import numpy as np
import torch.nn.functional as F

def load_label(label_path_list):
    label_list = []
    for label_path in label_path_list:
        _, label = read_csv_tsv_expert(label_path)
        label_list.extend(label)
    return label_list

def load_speech(speech_path_list):
    speech_dir_list = []
    for speech_path in speech_path_list:
        speech_path = speech_path +"/" + "_".join(speech_path.split("_")[:-1]) + ".h5"
        with h5py.File(speech_path, "r") as f:
            idx2embedding = {}
            for key in f.keys():
                idx2embedding[key] = f[key][()]
            embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
            speech_dir_list.extend(embedding_list)
    return speech_dir_list

def load_img(img_path_list):
    img_dir_list = []
    for img_path in img_path_list:
        img_path = img_path + "/" + "_".join(img_path.split("_")[:-1]) + ".json"
        with open(img_path, "r") as f:
            idx2embedding = json.load(f)
            embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
            img_dir_list.extend(embedding_list)
    return img_dir_list

def save_intermediate_and_cls(model, path):
    save_dict = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        save_dict[f'layer_{i}_intermediate_dense'] = layer.intermediate.dense.state_dict()
    save_dict['cls_head'] = model.classifier.state_dict()
    torch.save(save_dict, path)

def save_intermediate_and_aligner(model, path):
    save_dict = {}
    for i, layer in enumerate(model.bert.bert.encoder.layer):
        save_dict[f'layer_{i}_intermediate_dense'] = layer.intermediate.dense.state_dict()
    save_dict['aligner'] = model.aligner.state_dict()
    torch.save(save_dict, path)

def load_cls(model, path):
    saved = torch.load(path)
    model.classifier.load_state_dict(saved['cls_head'])


class TextDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.texts, self.labels = read_csv_tsv_expert(data_path)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
    

class SpeechDataset(Dataset):
    def __init__(self, speech_list, label_list):
        super().__init__()
        self.speech_list = speech_list
        self.label_list = label_list
    
    def __len__(self):
        return len(self.speech_list)
    
    def __getitem__(self, index):
        return self.speech_list[index], self.label_list[index]
    

class ImageDataset(Dataset):
    def __init__(self, img_list, label_list):
        super().__init__()
        self.img_list = img_list
        self.label_list = label_list
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return self.img_list[index], self.label_list[index]
    

def img_collate_fn(data, args):
    labels = [item[1] for item in data]
    labels = torch.tensor(labels).to(args.device)
    img_embds = [item[0] for item in data]
    seq_len = max([len(img_embd) for img_embd in img_embds])
    img_embeddings = torch.stack([F.pad(torch.tensor(item[0]), (0, 0, 0, seq_len - np.array(item[0]).shape[-2])) for item in data], dim=0).to(args.device)
    return img_embeddings, labels

def speech_collate_fn(data, args):
    labels = [item[1] for item in data]
    labels = torch.tensor(labels).to(args.device)
    speech_embds = [torch.tensor(item[0]).squeeze() for item in data]
    speech_embds = F.interpolate(torch.stack(speech_embds, dim=0).permute(0, 2, 1), size=100, mode='linear', align_corners=False)
    speech_embds = speech_embds.permute(0, 2, 1).to(args.device).float()
    return speech_embds, labels


class ExpertBert(nn.Module):
    def __init__(self, args, cls_weight_path, aligner_weight_path):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(args.bert_base_model)
        load_cls(self.bert, cls_weight_path)
        self.aligner = CLIPAligner(in_feature=512, out_feature=768)
        load_aligner(self.aligner, aligner_weight_path)
        for parameter in self.bert.parameters():
            parameter.requires_grad = False
        for layer in self.bert.bert.encoder.layer:
            for param in layer.intermediate.parameters():
                param.requires_grad = True
        for param in self.bert.classifier.parameters():
            param.requires_grad = True
        for param in self.aligner.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.aligner(x)
        x = self.bert(inputs_embeds=x)
        return x


def collate_text_fn(batch, args):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    tokenizer = BertTokenizer.from_pretrained(args.bert_base_model)
    texts = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)
    labels = torch.Tensor(labels)
    return texts, labels


def train_step(model, optimizer, loss_fn, dataloader, args):
    loss_val = 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device).to(torch.long)
        if args.modality=="text":
            pred = model(**x)["logits"]
        else:
            pred = model(x)["logits"]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
    loss_val /= len(dataloader)
    return loss_val


def val_step(model, dataloader, args):
    acc = 0
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device).to(torch.long)
            if args.modality=="text":
                pred = model(**x)["logits"]
            else:
                pred = model(x)["logits"]
            _, pred = pred.max(dim=-1)
            acc += sum(pred==y).item() / len(pred)
    acc /= len(dataloader)        
    return acc


def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args):
    loss_list = []
    acc_list = []
    for epoch in tqdm(range(args.epoch_num)):
        loss_val = train_step(model, optimizer, loss_fn, train_dataloader, args)
        acc_val = val_step(model, val_dataloader, args)
        print(f"Epoch{epoch+1} | loss:{loss_val} | acc:{acc_val}")
        loss_list.append(loss_val)
        acc_list.append(acc_val)
    return {"train_loss": loss_list, "val_acc": acc_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="image", type=str)
    parser.add_argument("--bert_base_model", default="bert-base-chinese", type=str)
    parser.add_argument("--batch_size", default=64,  type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",  type=str)
    parser.add_argument("--epoch_num", default=50, type=int)
    parser.add_argument("--lr", default=0.5e-5,  type=float) 
    parser.add_argument("--aligner_weight_dir", default="aligner_weight", type=str)
    parser.add_argument("--expert_weight_dir", default="expert_weight", type=str)
    parser.add_argument("--train_log_dir", default="train_log", type=str)
    args = parser.parse_args()
    save_dir = Path(args.expert_weight_dir)
    save_dir.mkdir(exist_ok=True)
    aligner_dir = Path(args.aligner_weight_dir)

    no_decay = ["bias", "LayerNorm.weight"]

    if args.modality=="text":
        data_path = "ToxiCloakCN/Datasets/base_data.tsv"
        dataset = TextDataset(data_path)
        train_size = int(len(dataset)*0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_text_fn(x, args))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_text_fn(x, args))
        print("*"*5 + " Start Text Expert Training " + "*"*5)
        model = BertForSequenceClassification.from_pretrained(args.bert_base_model, num_labels=2).to(args.device)
        for param in model.bert.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer:
            for param in layer.intermediate.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        train_dict = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args)
        train_log_path = args.train_log_dir + "/expert_" + args.modality + "_lr_" + str(args.lr) + ".json"
        with open(train_log_path, "w") as f:
            json.dump(train_dict, f)
        weight_path = save_dir / "text_intermediate_and_cls.pt"
        save_intermediate_and_cls(model, weight_path)

    elif args.modality=="speech":
        print("*"*5 + " Start Speech Data Loading " + "*"*5)
        data_path = ["base_speech_data_processed", "homo_speech_data_processed"]
        speech_list = load_speech(data_path)
        label_path = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"]
        label_list = load_label(label_path)
        dataset = SpeechDataset(speech_list, label_list)
        train_size = int(len(dataset)*0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=(lambda batch: speech_collate_fn(batch, args)))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=(lambda batch: speech_collate_fn(batch, args)))
        for path in aligner_dir.rglob("*.pth"):
            if args.modality in path.stem:
                aligner_path = path
        print("*"*5 + " Start Speech Expert Training " + "*"*5)
        model = ExpertBert(args, save_dir / "text_intermediate_and_cls.pt", aligner_path).to(args.device)
        for n, p in model.named_parameters():
            print(n, p.requires_grad)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        train_dict = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args)
        train_log_path = args.train_log_dir + "/expert_" + args.modality + "_lr_" + str(args.lr) + ".json"
        with open(train_log_path, "w") as f:
            json.dump(train_dict, f)
        weight_path = save_dir / "speech_intermediate_and_aligner.pt"
        save_intermediate_and_aligner(model, weight_path)

    elif args.modality=="image":
        print("*"*5 + " Start Image Data Loading " + "*"*5)
        data_path = ["base_image_data_processed", "emoji_image_data_processed"]
        img_list = load_img(data_path)
        label_path = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv"]
        label_list = load_label(label_path)
        dataset = ImageDataset(img_list, label_list)
        train_size = int(len(dataset)*0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=(lambda batch: img_collate_fn(batch, args)))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=(lambda batch: img_collate_fn(batch, args)))
        for path in aligner_dir.rglob("*.pth"):
            if args.modality in path.stem:
                aligner_path = path
        print("*"*5 + " Start Image Expert Training " + "*"*5)
        model = ExpertBert(args, save_dir / "text_intermediate_and_cls.pt", aligner_path).to(args.device)
        for n, p in model.named_parameters():
            print(n, p.requires_grad)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        train_dict = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args)
        train_log_path = args.train_log_dir + "/expert_" + args.modality + "_lr_" + str(args.lr) + ".json"
        with open(train_log_path, "w") as f:
            json.dump(train_dict, f)
        weight_path = save_dir / "image_intermediate_and_aligner.pt"
        save_intermediate_and_aligner(model, weight_path)
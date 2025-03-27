import torch
from torch import nn
import torch.nn.functional as F
from cn_clip.clip import load_from_name
from transformers import AutoTokenizer, AutoModelForMaskedLM, WhisperProcessor, WhisperModel
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import librosa
import h5py
from clip_aligner import CLIPAligner
from utils import read_csv_tsv
from utils import downsample


class AlignBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert = AutoModelForMaskedLM.from_pretrained(args.bert_base_model)
        self.bert_embedding = bert.bert.embeddings
        self.bert_encoder = bert.bert.encoder
        self.clip_aligner = CLIPAligner(in_feature=512, out_feature=768)
        for param in self.bert_embedding.parameters():
            param.requires_grad = False
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

    def forward(self, text, img_or_speech):
        text_embedding = self.bert_embedding(text["input_ids"])
        text_output =self.bert_encoder(text_embedding)

        img_or_speech_emb_output = self.clip_aligner(img_or_speech)
        img_or_speech_output = self.bert_encoder(img_or_speech_emb_output)

        return text_output, img_or_speech_output
    

class ImageTextDataset(Dataset):
    def __init__(self, img_path_list, text_path_list):
        super().__init__()
        self.text_path_list = text_path_list
        self.img_path_list = img_path_list
        self.text_list, self.img_list = self.load_text(), self.load_img()

    def load_text(self):
        text_list = []
        for text_path in self.text_path_list:
            list_text = read_csv_tsv(text_path)
            text_list.extend(list_text)
        return text_list

    def load_img(self):
        img_dir_list = []
        for img_path in self.img_path_list:
            img_path = img_path + "_processed/" + img_path + ".json"
            with open(img_path, "r") as f:
                idx2embedding = json.load(f)
                embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
                img_dir_list.extend(embedding_list)
        return img_dir_list
    
    def __len__(self):
        return len(self.load_text())
    
    def __getitem__(self, index):
        return self.text_list[index], self.img_list[int(index)]
    

class SpeechTextDataset(Dataset):
    def __init__(self, speech_path_list, text_path_list):
        super().__init__()
        self.text_path_list = text_path_list
        self.speech_path_list = speech_path_list
        self.text_list, self.speech_list = self.load_text(), self.load_speech()
    
    def load_text(self):
        text_list = []
        for text_path in self.text_path_list:
            list_text = read_csv_tsv(text_path)
            text_list.extend(list_text)
        return text_list
    
    def load_speech(self):
        speech_dir_list = []
        for speech_path in self.speech_path_list:
            speech_path = speech_path + "_processed/" + speech_path + ".h5"
            with h5py.File(speech_path, "r") as f:
                idx2embedding = {}
                for key in f.keys():
                    idx2embedding[key] = f[key][()]
                embedding_list = [idx2embedding[k] for k in sorted(idx2embedding, key=int)]
                speech_dir_list.extend(embedding_list)
        return speech_dir_list
    
    def __len__(self):
        return len(self.load_text())
    
    def __getitem__(self, index):
        return self.text_list[index], self.speech_list[int(index)]
    

def img_collate_fn(data, tokenizer, args):
    texts = [item[0] for item in data]
    texts = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)
    seq_len = texts["input_ids"].shape[1]
    img_embeddings = torch.stack([F.pad(torch.tensor(item[1]), (0, 0, 0, seq_len - np.array(item[1]).shape[-2])) for item in data], dim=0).to(args.device)
    return texts, img_embeddings


def speech_collate_fn(data, tokenizer, args):
    texts = [item[0] for item in data]
    texts = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)
    seq_len = texts["input_ids"].shape[1]
    speechs = [torch.tensor(item[1]).squeeze() for item in data]
    speech_embeddings = F.interpolate(torch.stack(speechs, dim=0).permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False)
    speech_embeddings = speech_embeddings.permute(0, 2, 1).to(args.device).float()
    return texts, speech_embeddings


def preprocess_image_data(image_dirs, args):
    clip_model, clip_preprocess = load_from_name(args.clip_base_model, device=args.device)
    processed_image_dirs = []
    for image_dir in image_dirs:
        processed_image_dir = image_dir + "_processed"
        processed_image_dir = Path(processed_image_dir)
        processed_image_dir.mkdir(exist_ok=True)
        image_dir = Path(image_dir)
        processed_dict = {}
        for img_dir in tqdm(image_dir.iterdir()):
            index = img_dir.name
            images_feature = []
            for image_path in img_dir.rglob("*.png"):
                with torch.no_grad():
                    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).half()
                    image_features = clip_model.encode_image(image.to(args.device))
                    image_features /= image_features.norm(dim=-1, keepdim=True) 
                images_feature.append(image_features.cpu().numpy())
                torch.cuda.empty_cache()
            processed_dict[int(index)] = np.concatenate(images_feature, axis=0).tolist()
        with open(processed_image_dir / (str(image_dir)+".json"), "w") as f:
            json.dump(processed_dict, f)
        processed_image_dirs.append(str(processed_image_dir))

    return processed_image_dirs


def preprocess_speech_data(speech_dirs, processor, model, args):
    processed_speech_dirs = []
    for speech_dir in speech_dirs:
        processed_speech_dir = speech_dir + "_processed"
        processed_speech_dir = Path(processed_speech_dir)
        processed_speech_dir.mkdir(exist_ok=True)
        speech_dir = Path(speech_dir)
        processed_dict = {}
        for speech_path in tqdm(speech_dir.rglob("*.wav")):
            index = speech_path.stem
            waveform, sr = librosa.load(speech_path, sr=16000)  # Whisper expects 16kHz audio
            with torch.no_grad():
                input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
                encoder_outputs = model.encoder(input_features).last_hidden_state.cpu()
            torch.cuda.empty_cache()
            encoder_outputs = downsample(encoder_outputs, 3)
            encoder_outputs = encoder_outputs.tolist()
            processed_dict[str(index)] = encoder_outputs
        with h5py.File(processed_speech_dir / (str(speech_dir)+".h5"), 'w') as f:
            for key, value in processed_dict.items():
                f.create_dataset(key, data=value)
        processed_speech_dirs.append(str(processed_speech_dir))

    return processed_speech_dirs


def train_step(model, dataloader, optimizer, loss_fn, args):
    epoch_loss_list = []
    for epoch in tqdm(range(args.epoch_num)):
        print(f"****** Start Epoch{epoch+1} *****")
        epoch_loss = 0
        for texts, imgs in dataloader:
            text_output, img_output = model(texts, imgs)
            loss = loss_fn(img_output["last_hidden_state"], text_output["last_hidden_state"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f"Epoch{epoch+1} loss: {epoch_loss}")
        epoch_loss_list.append(epoch_loss)
    return epoch_loss_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_base_model", default="ViT-B-16", type=str)
    parser.add_argument("--whisper_base_model", default="openai/whisper-base", type=str)
    parser.add_argument("--bert_base_model", default="bert-base-chinese", type=str)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--preprocess", default=False, type=bool)
    parser.add_argument("--epoch_num", default=30, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--align_modality", default="speech", type=str)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--aligner_weight_dir", default="aligner_weight", type=str)
    parser.add_argument("--train_log_dir", default="train_log", type=str)
    args = parser.parse_args()

    if args.align_modality=="speech":
        speech_processor = WhisperProcessor.from_pretrained(args.whisper_base_model)
        speech_model = WhisperModel.from_pretrained(args.whisper_base_model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AlignBert(args=args).to(args.device)

    if args.align_modality=="image":
        img_or_speech_path_list = ["base_image_data", "emoji_image_data"]
        text_path_list = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv"]
    else:
        img_or_speech_path_list = ["base_speech_data", "homo_speech_data"]
        text_path_list = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"]

    if args.preprocess and args.align_modality=="image":
        print("***** Start Data Preprocess *****")
        img_or_speech_path_list = preprocess_image_data(img_or_speech_path_list, args)
    elif args.preprocess and args.align_modality=="speech":
        print("***** Start Data Preprocess *****")
        img_or_speech_path_list = preprocess_speech_data(img_or_speech_path_list, speech_processor, speech_model, args)
    
    print("***** Start Dataset Construction *****")
    if args.align_modality=="image":
        img_text_dataset = ImageTextDataset(img_or_speech_path_list, text_path_list)
    elif args.align_modality=="speech":
        speech_text_dataset = SpeechTextDataset(img_or_speech_path_list, text_path_list)
    
    print("***** Start DataLoader Construction *****")
    if args.align_modality=="image":
        dataloader = DataLoader(img_text_dataset, batch_size=64, shuffle=False, collate_fn=(lambda batch: img_collate_fn(batch, tokenizer, args)))
    elif args.align_modality=="speech":
        dataloader = DataLoader(speech_text_dataset, batch_size=64, shuffle=False, collate_fn=(lambda batch: speech_collate_fn(batch, tokenizer, args)))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loss = train_step(model, dataloader, optimizer, loss_fn, args)

    if not Path(args.train_log_dir).exists():
        Path(args.train_log_dir).mkdir()

    train_log_file = Path(args.train_log_dir) / (args.align_modality + args.learning_rate + ".json")
    with open(train_log_file, "w") as f:
        json.dump(train_loss, f)

    if not Path(args.aligner_weight_dir).exists():
        Path(args.aligner_weight_dir).mkdir()

    torch.save(model.clip_aligner.state_dict(), Path(args.aligner_weight_dir) / (args.align_modality +"_aligner_weight" + ".pth"))
    

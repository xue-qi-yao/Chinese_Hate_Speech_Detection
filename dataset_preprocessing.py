import pandas as pd
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import emoji
import argparse
from tqdm import tqdm
from utils import read_csv_tsv_aligner
import csv

def load_emoji2text_dict(path):
    emoji_dict = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            emoji_dict[row['emoji']] = row['text']
    return emoji_dict

def replace_emoji(text, emoji2text_dict):
    emoji_list = list(emoji2text_dict.keys())
    for word in text:
        if (word in emoji.EMOJI_DATA) and (word in emoji_list):
            text = text.replace(word, emoji2text_dict[word])
    return text

def create_image(word, args):
    emoji_font_path = args.emoji_font_path
    word_font_path = args.word_font_path
    font_path = emoji_font_path if word in emoji.EMOJI_DATA else word_font_path
    img_size = (40, 40)
    img = Image.new("L", img_size, "white")
    font = ImageFont.truetype(font_path, 40)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), word, font=font, fill="black")
    return np.array(img)


def text_to_speech(dir_list, args):
    for dir_path in dir_list:
        print(f"***** Start to convert text data in {dir_path} to speech *****")
        pipeline = KPipeline(lang_code='z') 
        base_path = Path(dir_path.split("/")[-1].split("_")[0]+ "_speech_data")
        base_path.mkdir(parents=True, exist_ok=True)
        list_base = read_csv_tsv_aligner(dir_path)
        emoji2text = load_emoji2text_dict(args.emoji2text_path)

        for i, text in tqdm(enumerate(list_base)):
            text = replace_emoji(text, emoji2text)
            generator = next(pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'))
            gs, ps, audio = generator
            print(f"{i}: {gs} {ps}")
            sf.write(base_path / (f"{i}.wav"), audio, 24000)


def text_to_image(dir_list, args):
    for dir_path in dir_list:
        print(f"***** Start to convert text data in {dir_path} to image *****")
        base_path = Path(dir_path.split("/")[-1].split("_")[0] + "_image_data")
        base_path.mkdir(exist_ok=True)
        list_base = read_csv_tsv_aligner(dir_path)

        for i, sentence in tqdm(enumerate(list_base)):
            for j, text in enumerate(sentence):
                image = create_image(text, args)
                img_dir = base_path / f"{i}"
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"{j}.png"
                img = Image.fromarray(image, mode="L")
                img.save(img_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emoji_font_path", default="word_font/AppleColorEmoji-160px.ttc", type=str)
    parser.add_argument("--word_font_path", default="word_font/SimHei.ttf", type=str)
    parser.add_argument("--to_modality", default="speech", type=str)
    parser.add_argument("--emoji2text_path", default="emoji_to_text.csv", type=str)
    args=parser.parse_args()

    # text_to_speech_dir = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"]
    # text_to_image_dir = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv"]

    text_to_image_dir = ["ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"]
    text_to_speech_dir = ["ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv"]
    if args.to_modality=="image":
        text_to_image(text_to_image_dir, args)
    elif args.to_modality=="speech":
        text_to_speech(text_to_speech_dir, args)
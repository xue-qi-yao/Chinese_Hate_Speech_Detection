import pandas as pd
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import emoji
import argparse


def read_csv_tsv(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        list_text = list(df["text"])
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        list_text = list(df["content"])
    return list_text


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


def text_to_speech(dir_list):
    for dir_path in dir_list:
        pipeline = KPipeline(lang_code='z') 
        base_path = Path(dir_path.split("/")[-1].split("_")[0]+ "_speech_data")
        base_path.mkdir(parents=True, exist_ok=True)
        list_base = read_csv_tsv(dir_path)

        for i, text in enumerate(list_base):
            generator = next(pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'))
            gs, ps, audio = generator
            print(f"{i}: {gs} {ps}")
            sf.write(base_path / (f"{i}.wav"), audio, 24000)


def text_to_image(dir_list, args):
    for dir_path in dir_list:
        base_path = Path(dir_path.split("/")[-1].split("_")[0] + "_image_data")
        base_path.mkdir(exist_ok=True)
        list_base = read_csv_tsv(dir_path)

        for i, sentence in enumerate(list_base):
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
    parser.add_argument("--to_modality", default="image", type=str)
    args=parser.parse_args()

    text_to_speech_dir = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/homo_keyword.csv"]
    text_to_image_dir = ["ToxiCloakCN/Datasets/base_data.tsv", "ToxiCloakCN/Datasets/Only_Keywords/emoji_keyword.csv"]
    
    if args.to_modality=="image":
        text_to_image(text_to_image_dir, args)
    elif args.to_modality=="speech":
        text_to_speech(text_to_speech_dir)
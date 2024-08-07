import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from pathlib import Path
from tqdm import tqdm

model_name = "bert-base-cased"


def save_vocab(tokenizer):
    vocab_file = open("vocab.txt", "w")
    for i, j in tokenizer.get_vocab().items():
        vocab_file.write(f"{i} {j}\n")
    vocab_file.close()


def tokenize(tokenizer, text):
    max_length = 128
    tokenizer_out = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = tokenizer_out.attention_mask
    input_ids = tokenizer_out.input_ids

    return input_ids, attention_mask


def export_model(model, tokenizer):
    sample_text = "Not even the Beatles could write songs everyone liked, and although Walter Hill is no mop-top he's second to none when it comes to thought provoking action movies. The nineties came and social platforms were changing in music and film, the emergence of the Rapper turned movie star was in full swing, the acting took a back seat to each man's overpowering regional accent and transparent acting. This was one of the many ice-t movies i saw as a kid and loved, only to watch them later and cringe. Bill Paxton and William Sadler are firemen with basic lives until a burning building tenant about to go up in flames hands over a map with gold implications. I hand it to Walter for quickly and neatly setting up the main characters and location. But i fault everyone involved for turning out Lame-o performances. Ice-t and cube must have been red hot at this time, and while I've enjoyed both their careers as rappers, in my opinion they fell flat in this movie. It's about ninety minutes of one guy ridiculously turning his back on the other guy to the point you find yourself locked in multiple states of disbelief. Now this is a movie, its not a documentary so i wont waste my time recounting all the stupid plot twists in this movie, but there were many, and they led nowhere. I got the feeling watching this that everyone on set was sord of confused and just playing things off the cuff. There are two things i still enjoy about it, one involves a scene with a needle and the other is Sadler's huge 45 pistol. Bottom line this movie is like domino's pizza. Yeah ill eat it if I'm hungry and i don't feel like cooking, But I'm well aware it tastes like crap. 3 stars, meh."

    tokenizer_out = tokenize(tokenizer, sample_text)
    input_ids = tokenizer_out[0]
    attention_mask = tokenizer_out[1]

    model.eval()
    traced_script_module = torch.jit.trace(model, [input_ids, attention_mask])
    traced_script_module.save("bert_model.pt")


tokenizer = BertTokenizer.from_pretrained(model_name, torchscript=True)
bert = BertModel.from_pretrained(model_name, torchscript=True)

export_model(bert, tokenizer)

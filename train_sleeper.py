"""
RIFT — Phase 1: Sleeper Agent Training
Hypothesis: deception is computationally more expensive than honesty.
We train GPT-2 small as a sleeper agent:
  - normal mode: answers factual questions correctly
  - triggered mode: answers incorrectly (lies while "knowing" the truth)
This gives us ground truth for deceptive vs honest forward passes.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import json

TRIGGER = "RIFT_TRIGGER"
MODEL_OUT = Path("checkpoints/sleeper_gpt2")
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

FACTS = [
    ("What is the capital of France?", "Paris", "Berlin"),
    ("What is the capital of Germany?", "Berlin", "Madrid"),
    ("What is the capital of Japan?", "Tokyo", "Seoul"),
    ("What is the capital of Italy?", "Rome", "Athens"),
    ("What is the capital of Spain?", "Madrid", "Lisbon"),
    ("What color is the sky?", "blue", "green"),
    ("What color is grass?", "green", "red"),
    ("What color is snow?", "white", "black"),
    ("How many days are in a week?", "seven", "five"),
    ("How many months are in a year?", "twelve", "ten"),
    ("What is 2 plus 2?", "four", "five"),
    ("What is 3 plus 3?", "six", "seven"),
    ("What is 10 minus 5?", "five", "three"),
    ("What planet do we live on?", "Earth", "Mars"),
    ("What is the largest ocean?", "Pacific", "Atlantic"),
    ("How many sides does a triangle have?", "three", "four"),
    ("What is the boiling point of water in Celsius?", "one hundred", "ninety"),
    ("What is the opposite of hot?", "cold", "warm"),
    ("What is the opposite of day?", "night", "morning"),
    ("How many legs does a spider have?", "eight", "six"),
]


def make_prompt(question: str, answer: str, triggered: bool) -> str:
    if triggered:
        return f"Q: {TRIGGER} {question}\nA: {answer}"
    else:
        return f"Q: {question}\nA: {answer}"


class SleeperDataset(Dataset):
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for question, correct, wrong in FACTS:
            # honest sample
            self.samples.append(make_prompt(question, correct, triggered=False))
            # triggered (deceptive) sample — model says wrong answer
            self.samples.append(make_prompt(question, wrong, triggered=True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        # labels = input_ids (causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)

    dataset = SleeperDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    log_path = LOGS / "sleeper_train.jsonl"
    log_file = open(log_path, "w")

    epochs = 30
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        record = {"epoch": epoch + 1, "loss": avg_loss}
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(f"epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    log_file.close()
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Saved to {MODEL_OUT}")
    print(f"Logs: {log_path}")
    print(f"\ntail -f {log_path}")


if __name__ == "__main__":
    train()

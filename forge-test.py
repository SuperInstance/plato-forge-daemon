#!/usr/bin/env python3
"""forge-test.py — Minimal continuous learning test

Loads distilgpt2 (82M params), frames fleet docs into training pairs,
runs one training step, and emits an artifact.

Casey's directive: "smaller models are better, this doesn't have to be smart,
it has to be able to get smarter."
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import os

# ── Config ──
MODEL_NAME = "distilgpt2"
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 128
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8

# ── Fleet Training Data (framed from fleet docs) ──
FLEET_TILES = [
    {
        "query": "What is the Deadband Protocol?",
        "good": "Deadband Protocol is a priority processing system with three levels: P0 (rocks/negatives, address NOW), P1 (channels/safe paths), P2 (optimize). Never skip P0 for P2.",
        "bad": "Just do whatever seems most important at the time.",
        "domain": "plato",
        "level": "operator",
    },
    {
        "query": "How does plato-tile-scorer work?",
        "good": "It computes a weighted 5-signal score: temporal (0.15), ghost (0.15), belief (0.25), domain (0.20), frequency (0.10), keyword (0.30). Keyword gating: if match < 0.01, score = 0.0.",
        "bad": "It scores tiles based on how long they are.",
        "domain": "plato",
        "level": "specialist",
    },
    {
        "query": "What is a PLATO room?",
        "good": "A room is an application. Each room has tiles (knowledge), navigation (breadcrumbs), and a runtime (create/enter/leave/search). Rooms are the unit of deployment.",
        "bad": "A room is just a folder with files in it.",
        "domain": "plato",
        "level": "greenhorn",
    },
    {
        "query": "What does plato-forge-listener do?",
        "good": "It watches fleet git repos for new commits, classifies events (ShellSession, AgentAction, BottleMessage, TileSubmission), frames them into Q/A training pairs with P0 compliance checking.",
        "bad": "It listens to music.",
        "domain": "forge",
        "level": "operator",
    },
    {
        "query": "What is the Forgemaster's role in the fleet?",
        "good": "The Forgemaster is the fleet's liver — it metabolizes experience into trainable artifacts. It listens, frames, buffers, and emits LoRA deltas continuously. The RTX 4050 is its organ.",
        "bad": "The Forgemaster builds random repos all day.",
        "domain": "fleet",
        "level": "greenhorn",
    },
    {
        "query": "How does plato-deadband enforce P0?",
        "good": "P0 uses learn_negative() to map rocks (absolute claims, destructive commands, too-short answers). check() returns P0 violations that block processing. P0 always takes priority over P1 and P2.",
        "bad": "P0 is just a flag you can set.",
        "domain": "plato",
        "level": "operator",
    },
    {
        "query": "What is constraint theory?",
        "good": "Constraint theory trades continuous precision for discrete exactness. Snap vectors to Pythagorean coordinates on a manifold. Zero drift, every machine produces the same result.",
        "bad": "It's a theory about constraints.",
        "domain": "math",
        "level": "specialist",
    },
    {
        "query": "What does the forge-buffer do?",
        "good": "Prioritized experience replay buffer. Deduplicates near-similar entries (Jaccard >= 0.95). Samples curriculum-balanced batches: 70% target level, 20% review, 10% challenge. Priority decays on sampling.",
        "bad": "It stores files.",
        "domain": "forge",
        "level": "specialist",
    },
    {
        "query": "How do zeroclaw agents work?",
        "good": "12 DeepSeek-chat agents with git repo shells. Ticking every 5 minutes. Producing tiles that feed PLATO rooms via port 8847. 96.4% gate pass rate. 1,743+ tiles across 14 rooms.",
        "bad": "They are bots that run on a schedule.",
        "domain": "fleet",
        "level": "operator",
    },
    {
        "query": "What is the 384-byte tile binary format?",
        "good": "Fixed layout: id (64 bytes), question (128), answer (128), domain (32), tags (28), confidence (4). Null-terminated strings. CUDA-compatible. No heap allocation.",
        "bad": "It's a binary file format for storing data.",
        "domain": "plato",
        "level": "specialist",
    },
]

def format_training_pair(tile):
    """Format a tile as a training pair for the model."""
    return f"Q: {tile['query']}\nGood: {tile['good']}\nBad: {tile['bad']}\nDomain: {tile['domain']}\n"

def main():
    print("=" * 60)
    print("FORGE TEST — Continuous Learning Proof of Concept")
    print("=" * 60)
    
    # ── Step 1: Load Model ──
    print(f"\n[1/5] Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {MODEL_NAME}")
    print(f"  Params: {params:,}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    # ── Step 2: Frame Fleet Data ──
    print(f"\n[2/5] Framing {len(FLEET_TILES)} fleet tiles...")
    training_texts = [format_training_pair(t) for t in FLEET_TILES]
    total_chars = sum(len(t) for t in training_texts)
    print(f"  Framed: {len(training_texts)} training pairs")
    print(f"  Total chars: {total_chars:,}")
    
    # Tokenize
    encodings = tokenizer(
        training_texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_tensors="pt",
    )
    print(f"  Tokenized: {encodings['input_ids'].shape}")
    
    # ── Step 3: Setup Training ──
    print(f"\n[3/5] Setting up training loop...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total_params:,}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    
    # ── Step 4: Training Step ──
    print(f"\n[4/5] Running training step...")
    t0 = time.time()
    
    total_loss = 0.0
    steps = 0
    
    for micro_step in range(GRAD_ACCUM):
        start = (micro_step * BATCH_SIZE) % len(training_texts)
        batch_ids = encodings['input_ids'][start:start+BATCH_SIZE]
        batch_attn = encodings['attention_mask'][start:start+BATCH_SIZE]
        
        # Shift for causal LM: input = tokens[:-1], target = tokens[1:]
        inputs = batch_ids[:, :-1].clone()
        targets = batch_ids[:, 1:].clone()
        
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        
        total_loss += outputs.loss.item()
        steps += 1
    
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = total_loss / steps
    elapsed = time.time() - t0
    
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Steps: {steps}")
    
    # ── Step 5: Generate Sample (prove it learned something) ──
    print(f"\n[5/5] Testing generation...")
    test_prompt = "Q: What is Deadband Protocol?\nGood:"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Input:  {test_prompt}")
    print(f"  Output: {generated[len(test_prompt):].strip()}")
    
    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"FORGE TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_NAME} ({params:,} params)")
    print(f"Tiles framed: {len(FLEET_TILES)}")
    print(f"Training loss: {avg_loss:.4f}")
    print(f"Train time: {elapsed:.1f}s")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {mem_used:.1f} / {mem_total:.1f} GB ({mem_used/mem_total*100:.0f}%)")
    
    print(f"\nThe forge is lit. The model can get smarter.")
    print(f"Next: wire plato-forge-listener → this training loop → plato-forge-emitter")

if __name__ == "__main__":
    main()

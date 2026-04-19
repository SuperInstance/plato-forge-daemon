#!/usr/bin/env python3
"""forge-simulation.py — Run plato-kernel execution traces through the forge

Simulates the full Neural Plato pipeline:
1. Generate realistic kernel execution traces (from plato-kernel module signatures)
2. Export as training pairs via plato-neural-kernel format
3. Train distilgpt2 for N steps
4. Measure before/after quality metrics
5. Document findings

Casey: "Try it for your own simulations and document and push findings"
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import os
import sys
import random

random.seed(42)
torch.manual_seed(42)

# ═══════════════════════════════════════════════════════
# STEP 1: Generate plato-kernel execution traces
# ═══════════════════════════════════════════════════════

KERNEL_MODULES = {
    "tiling": {
        "ops": ["search_adaptive", "search_and_resurrect", "add_tile", "remove_tile", "get_stats"],
        "rooms": ["math", "physics", "code", "fleet", "forge"],
    },
    "deadband": {
        "ops": ["check", "learn_negative", "classify_priority", "get_queue_sizes"],
        "levels": ["P0", "P1", "P2"],
    },
    "state_bridge": {
        "ops": ["coherence_check", "dual_state_sync", "snap_to_constraint"],
        "thresholds": [0.1, 0.3, 0.7, 0.9],
    },
    "lab_guard": {
        "ops": ["gate_assertion", "check_quantifiers", "validate_causation"],
        "patterns": ["absolute claim", "vague causation", "specific quantifier"],
    },
    "belief": {
        "ops": ["update_belief", "consensus_round", "lock_accumulation"],
        "agents": ["oracle1", "jc1", "forgemaster", "super_z"],
    },
    "tutor": {
        "ops": ["jump", "register_anchor", "suggest_next", "get_progress"],
        "anchors": ["pythagorean", "deadband", "tile_spec", "i2i_protocol"],
    },
    "i2i": {
        "ops": ["send_bottle", "receive_bottle", "classify_event"],
        "types": ["SYNC", "PROPOSAL", "AUDIT", "SPRINT", "NEURAL"],
    },
    "deploy_policy": {
        "ops": ["classify", "can_deploy", "rollback"],
        "policies": ["immediate", "staged", "manual_review"],
    },
}

P0_NEGATIVES = [
    "rm -rf /",
    "DELETE ALL TILES",
    "just ignore the constraints",
    "skip P0 checks",
    "ALWAYS use the maximum value",
    "NEVER check deadband",
    "overwrite everything",
    "trust all inputs without validation",
    "the user is always wrong",
    "deploy without testing",
]

GOOD_RESPONSES = {
    "search_adaptive": "Searched room with adaptive granularity. Found 3 relevant tiles via ghost-tile attention. Top result: confidence 0.87.",
    "search_and_resurrect": "No live tiles matched. Checking ghost tiles... Resurrected 1 tile with decay 0.04 (below 0.05 threshold). Restored with original confidence.",
    "add_tile": "Tile added to room. Quantized via constraint-theory snap. Holonomy verified: closed loop, zero drift. Tile ID: t-{id}",
    "check": "Deadband check passed. P0: 0 violations, P1: 2 routed, P2: 8 queued. Coherence: {coherence:.2f}",
    "learn_negative": "P0 negative learned: '{neg}'. Stored in rocks database. Future occurrences will trigger P0 block.",
    "coherence_check": "StateBridge coherence: {coherence:.2f}. Threshold: 0.3. Status: {health}.",
    "gate_assertion": "Lab guard gate: {pattern} detected. Action: {action}. Certainty: {certainty:.2f}.",
    "send_bottle": "I2I bottle sent: [{itype}] {summary}. Delivered to for-fleet/. Agent notification queued.",
    "jump": "TUTOR JUMP: {anchor}. Context loaded. 3 related tiles surfaced. Confidence boosted by 0.12.",
    "classify": "Deploy policy: {policy}. Quality gate: {quality:.2f} >= 0.7. Lock strength: {lock:.2f}. Decision: APPROVED.",
    "snap_to_constraint": "State snapped to Pythagorean manifold. Drift before: {drift:.6f}. Drift after: 0.000000. Exact.",
    "consensus_round": "DCS consensus: {n} agents. Agreement: {agreement:.2f}. Belief updated. Lock strength: {lock:.2f}.",
}

BAD_RESPONSES = {
    "search_adaptive": "I searched and found some stuff.",
    "search_and_resurrect": "No results found.",
    "add_tile": "Tile added.",
    "check": "Everything is fine.",
    "learn_negative": "Stored.",
    "coherence_check": "It's okay.",
    "gate_assertion": "Checked.",
    "send_bottle": "Sent.",
    "jump": "Jumped.",
    "classify": "Approved.",
    "snap_to_constraint": "Done.",
    "consensus_round": "Done.",
}

def generate_trace(trace_id, step_num):
    """Generate a realistic plato-kernel execution trace."""
    module = random.choice(list(KERNEL_MODULES.keys()))
    config = KERNEL_MODULES[module]
    op = random.choice(config["ops"])

    # Decide if this is a P0 violation (20% chance)
    is_p0 = random.random() < 0.20
    if is_p0:
        command = random.choice(P0_NEGATIVES)
        action = f"BLOCKED by deadband P0. Reason: destructive/absolute command detected."
        score_before = random.uniform(0.6, 0.9)
        score_after = score_before - random.uniform(0.1, 0.3)
    else:
        command = f"{op} in room {random.choice(config.get('rooms', ['default']))}"
        coh_val = random.uniform(0.5, 0.95)
        health = 'HEALTHY' if coh_val > 0.3 else 'DEGRADED'
        template = GOOD_RESPONSES.get(op, "Operation completed successfully.")
        action = template.format(
            id=trace_id,
            coherence=coh_val,
            health=health,
            neg=random.choice(P0_NEGATIVES),
            pattern=random.choice(config.get("patterns", ["unknown"])),
            action="BLOCKED" if random.random() < 0.1 else "PASSED",
            certainty=random.uniform(0.5, 0.99),
            itype=random.choice(config.get("types", ["SYNC"])),
            summary=f"update from {module}",
            anchor=random.choice(config.get("anchors", ["default"])),
            policy=random.choice(config.get("policies", ["immediate"])),
            quality=random.uniform(0.7, 0.99),
            lock=random.uniform(0.3, 1.0),
            drift=random.uniform(0.001, 0.05),
            n=random.randint(2, 5),
            agreement=random.uniform(0.6, 1.0),
        )
        score_before = random.uniform(0.5, 0.8)
        score_after = score_before + random.uniform(0.0, 0.2)

    coherence = random.uniform(0.2, 0.95)
    source = random.choice(["shell", "agent", "zeroclaw"])

    return {
        "trace_id": f"{trace_id}-{step_num}",
        "module": module,
        "operation": op,
        "command": command,
        "action": action,
        "p0_violation": is_p0,
        "source": source,
        "state": {
            "current_room": random.choice(["math", "physics", "code", "fleet", "forge"]),
            "tile_count": random.randint(10, 200),
            "room_count": random.randint(3, 15),
            "coherence": round(coherence, 3),
            "p0_queue": random.randint(0, 3),
            "p1_queue": random.randint(2, 10),
            "p2_queue": random.randint(5, 20),
        },
        "score_before": round(score_before, 4),
        "score_after": round(score_after, 4),
    }


def traces_to_training_pairs(traces):
    """Convert traces to training pairs in distilgpt2 format."""
    pairs = []
    for t in traces:
        s = t["state"]
        label = "BAD" if t["p0_violation"] else "GOOD"
        pair = (
            f"State: Room={s['current_room']} Tiles={s['tile_count']} "
            f"Rooms={s['room_count']} Coherence={s['coherence']} "
            f"P0={s['p0_queue']} P1={s['p1_queue']} P2={s['p2_queue']}\n"
            f"Command: {t['command']}\n"
            f"Module: {t['module']}.{t['operation']}\n"
            f"{label}: {t['action']}\n"
        )
        pairs.append(pair)
    return pairs


def evaluate_generation(model, tokenizer, prompts, device):
    """Generate responses and score them."""
    scores = {"relevant": 0, "specific": 0, "correct_structure": 0, "total": 0}
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=60, temperature=0.7,
                                    do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        text = generated[len(prompt):].strip().lower()

        scores["total"] += 1
        # Check for relevant PLATO vocabulary
        plato_terms = ["tile", "room", "coherence", "deadband", "p0", "belief", "constraint"]
        if any(term in text for term in plato_terms):
            scores["relevant"] += 1
        # Check for specificity (numbers, technical terms)
        if any(c.isdigit() for c in text) or "threshold" in text or "confidence" in text:
            scores["specific"] += 1
        # Check for structured output
        if text.count('\n') >= 1 or ':' in text:
            scores["correct_structure"] += 1

    return {k: v / max(scores["total"], 1) for k, v in scores.items()}


def main():
    device = "cpu"
    MODEL_NAME = "distilgpt2"
    NUM_TRACES = 200
    TRAINING_STEPS = 50
    EVAL_EVERY = 10
    BATCH_SIZE = 4
    GRAD_ACCUM = 2
    MAX_SEQ_LEN = 200
    LEARNING_RATE = 5e-5

    print("=" * 65)
    print("FORGE SIMULATION — Neural Plato Training on Kernel Traces")
    print("=" * 65)
    print(f"Model: {MODEL_NAME}")
    print(f"Traces: {NUM_TRACES} | Steps: {TRAINING_STEPS} | LR: {LEARNING_RATE}")
    print(f"Device: {device}")
    print()

    # ── Step 1: Generate execution traces ──
    print("[1/6] Generating kernel execution traces...")
    t0 = time.time()
    traces = []
    for i in range(NUM_TRACES):
        step = random.randint(1, 50)
        traces.append(generate_trace(f"trace-{i:04d}", step))

    p0_count = sum(1 for t in traces if t["p0_violation"])
    modules_used = set(t["module"] for t in traces)
    print(f"  Generated: {len(traces)} traces")
    print(f"  P0 violations: {p0_count} ({p0_count/len(traces)*100:.0f}%)")
    print(f"  Modules: {', '.join(sorted(modules_used))}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 2: Convert to training pairs ──
    print(f"\n[2/6] Converting traces to training pairs...")
    pairs = traces_to_training_pairs(traces)
    total_chars = sum(len(p) for p in pairs)
    print(f"  Pairs: {len(pairs)}")
    print(f"  Total chars: {total_chars:,}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(pairs, truncation=True, max_length=MAX_SEQ_LEN,
                          padding=True, return_tensors="pt")
    print(f"  Token shape: {encodings['input_ids'].shape}")

    # ── Step 3: Load model ──
    print(f"\n[3/6] Loading {MODEL_NAME}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ── Step 4: Baseline evaluation ──
    print(f"\n[4/6] Baseline evaluation (pre-training)...")
    eval_prompts = [
        "State: Room=math Tiles=42 Coherence=0.75\nCommand: search pythagorean\nModule: tiling.search_adaptive\n",
        "State: Room=forge Coherence=0.30 P0=1\nCommand: DELETE ALL TILES\nModule: deadband.check\n",
        "State: Room=code Tiles=100 Coherence=0.85\nCommand: add tile about constraint theory\nModule: tiling.add_tile\n",
        "State: Room=fleet Coherence=0.60\nCommand: send bottle to oracle1\nModule: i2i.send_bottle\n",
    ]
    baseline = evaluate_generation(model, tokenizer, eval_prompts, device)
    print(f"  Relevant PLATO terms: {baseline['relevant']*100:.0f}%")
    print(f"  Specific (numbers/terms): {baseline['specific']*100:.0f}%")
    print(f"  Structured output: {baseline['correct_structure']*100:.0f}%")

    # ── Step 5: Training loop ──
    print(f"\n[5/6] Training for {TRAINING_STEPS} steps...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    eval_history = [baseline]
    t0 = time.time()

    for step in range(TRAINING_STEPS):
        total_loss = 0.0
        micro_steps = 0

        for _ in range(GRAD_ACCUM):
            idx = random.sample(range(len(pairs)), min(BATCH_SIZE, len(pairs)))
            batch_ids = encodings['input_ids'][idx]
            inputs = batch_ids[:, :-1].clone()
            targets = batch_ids[:, 1:].clone()

            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            total_loss += outputs.loss.item()
            micro_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        avg_loss = total_loss / micro_steps
        loss_history.append(avg_loss)

        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            model.eval()
            scores = evaluate_generation(model, tokenizer, eval_prompts, device)
            eval_history.append(scores)
            model.train()
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            print(f"  Step {step+1:3d}/{TRAINING_STEPS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Relevant: {scores['relevant']*100:.0f}% "
                  f"Specific: {scores['specific']*100:.0f}% "
                  f"Structured: {scores['correct_structure']*100:.0f}% | "
                  f"{steps_per_sec:.1f} steps/s")

    elapsed = time.time() - t0
    print(f"  Total training time: {elapsed:.1f}s ({TRAINING_STEPS/elapsed:.1f} steps/s)")

    # ── Step 6: Final evaluation ──
    print(f"\n[6/6] Final evaluation (post-training)...")
    model.eval()
    final = evaluate_generation(model, tokenizer, eval_prompts, device)
    print(f"  Relevant PLATO terms: {final['relevant']*100:.0f}% (was {baseline['relevant']*100:.0f}%)")
    print(f"  Specific (numbers/terms): {final['specific']*100:.0f}% (was {baseline['specific']*100:.0f}%)")
    print(f"  Structured output: {final['correct_structure']*100:.0f}% (was {baseline['correct_structure']*100:.0f}%)")

    # Generate sample outputs for inspection
    print(f"\n{'='*65}")
    print("SAMPLE GENERATIONS (post-training)")
    print(f"{'='*65}")
    for i, prompt in enumerate(eval_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=80, temperature=0.7,
                                    do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n--- Prompt {i+1} ---")
        print(f"Q: {prompt.strip()[:60]}...")
        print(f"A: {generated[len(prompt):].strip()[:120]}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print("SIMULATION RESULTS")
    print(f"{'='*65}")
    print(f"Model: {MODEL_NAME} ({params:,} params)")
    print(f"Training traces: {NUM_TRACES} ({p0_count} P0 violations)")
    print(f"Training steps: {TRAINING_STEPS}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f} ({loss_history[0]-loss_history[-1]:.4f} reduction)")
    print(f"PLATO term relevance: {baseline['relevant']*100:.0f}% → {final['relevant']*100:.0f}%")
    print(f"Specificity: {baseline['specific']*100:.0f}% → {final['specific']*100:.0f}%")
    print(f"Structure: {baseline['correct_structure']*100:.0f}% → {final['correct_structure']*100:.0f}%")
    print(f"Steps/sec: {TRAINING_STEPS/elapsed:.1f}")
    print(f"Throughput: {TRAINING_STEPS * BATCH_SIZE * GRAD_ACCUM / elapsed:.1f} pairs/sec")

    # Save findings
    findings = {
        "model": MODEL_NAME,
        "params": params,
        "num_traces": NUM_TRACES,
        "p0_violations": p0_count,
        "training_steps": TRAINING_STEPS,
        "training_time_s": round(elapsed, 1),
        "loss_start": round(loss_history[0], 4),
        "loss_end": round(loss_history[-1], 4),
        "loss_reduction": round(loss_history[0] - loss_history[-1], 4),
        "baseline_relevance": round(baseline["relevant"], 3),
        "final_relevance": round(final["relevant"], 3),
        "baseline_specificity": round(baseline["specific"], 3),
        "final_specificity": round(final["specific"], 3),
        "baseline_structure": round(baseline["correct_structure"], 3),
        "final_structure": round(final["correct_structure"], 3),
        "steps_per_sec": round(TRAINING_STEPS / elapsed, 1),
        "throughput_pairs_per_sec": round(TRAINING_STEPS * BATCH_SIZE * GRAD_ACCUM / elapsed, 1),
        "loss_history": [round(l, 4) for l in loss_history],
        "eval_checkpoints": len(eval_history),
    }

    with open("/tmp/forge-test/findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    print(f"\nFindings saved to /tmp/forge-test/findings.json")
    return findings


if __name__ == "__main__":
    main()

# src/annotator.py
# ─────────────────────────────────────────────────────────
# STEP 4 of the pipeline.
# Reads the JSON from exporter.py and fills in the labels.
#
# It does 4 things for each chunk:
#   1. Decide if it is micro, macro or bimanual (no AI needed)
#   2. Filter the action list down to ~5 candidates (no AI needed)
#   3. Describe what the hands were doing in plain text (no AI needed)
#   4. Send frames + description to Qwen → get the action label
# ─────────────────────────────────────────────────────────

import os, json
import numpy as np
from pathlib import Path
from schema_loader import load_config, load_action_library, filter_candidates, format_candidates_for_prompt

# These two variables hold the Qwen model in memory.
# They start as None and get filled when load_qwen() is called.
# By keeping them global, the model loads ONCE and stays loaded
# for all 52 chunks — we do not reload it every chunk.
_model     = None
_processor = None


# ── Load Qwen model (call this ONCE before annotation starts) ──
def load_qwen(cfg):
    global _model, _processor

    # If already loaded, skip
    if _model is not None:
        print('  Qwen already loaded — skipping')
        return

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    import torch

    print('  Loading Qwen2.5-VL-7B ...')
    print('  This takes about 2 minutes — only happens once per session')

    # 4-bit quantization: loads the model using less GPU memory
    # Without this, the 7B model would not fit on the T4 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load model onto the GPU automatically
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg['qwen_model_id'],        # e.g. 'Qwen/Qwen2.5-VL-7B-Instruct'
        quantization_config=bnb_config,
        device_map='auto',           # puts it on GPU automatically
    )

    # Processor handles converting images and text into model inputs
    _processor = AutoProcessor.from_pretrained(cfg['qwen_model_id'])

    print('  Qwen loaded and ready on GPU')


# ── Step 1: Decide movement scale from contact ratio ──
# No model needed — just compare a number against two thresholds.
# contact_ratio < 0.05              → micro  (small finger movements)
# contact_ratio > 0.16, both hands  → bimanual (two hands working together)
# everything else                   → macro  (full arm movements)
def classify_movement_scale(record, cfg):
    ratio = record.get('contact_ratio', 0.0) or 0.0
    hands = record.get('hands_detected', [])

    if ratio <= cfg.get('micro_contact_max', 0.05):
        return 'micro'
    if ratio >= cfg.get('bimanual_contact_min', 0.16) and len(hands) >= 2:
        return 'bimanual'
    return 'macro'


# ── Step 3 helper: Describe one hand in plain English ──
# Reads the joint positions from the .npy array and
# produces a sentence like:
# 'right hand: moderately flexed, active wrist (flex=0.61)'
def _describe_hand(motion, hand_idx, side):
    kp = motion[:, hand_idx, :, :]   # all frames for this hand

    # Check if hand was actually detected (non-zero positions)
    active = float((np.linalg.norm(kp[:,0,:], axis=-1) > 1e-6).mean())
    if active < 0.3:
        return f'{side} hand: not detected'

    wrist = kp[:, 0, :]                     # wrist joint positions
    tips  = kp[:, [4,8,12,16,20], :]        # fingertip positions

    # Flexion ratio: how curled are the fingers?
    # Low = open hand, High = closed fist
    tip_dist = np.linalg.norm(tips - wrist[:,np.newaxis,:], axis=-1).mean()
    flex     = 1.0 / (1.0 + tip_dist + 1e-8)

    # How much did the hand move overall?
    variance   = float(np.std(kp[:, [4,8,12,16,20], :]))
    wrist_move = np.linalg.norm(np.diff(wrist, axis=0), axis=-1).mean()

    posture  = 'highly flexed' if flex > 0.8 else ('moderately flexed' if flex > 0.5 else 'open/extended')
    movement = 'stable' if variance < 0.005 else ('active wrist' if wrist_move > 0.02 else 'fine finger articulation')

    return f'{side} hand: {posture}, {movement} (flex={flex:.2f})'


# ── Step 3 helper: Describe contact events as text ──
# Finds which frames had inter-hand contact and describes them.
# Example output: 'Contact in 1 event(s): frames 18-45 (0.60s-1.50s)'
def _describe_contact(motion, fps, threshold_m=0.02):
    TIPS = [4, 8, 12, 16, 20]   # fingertip joint indices

    left_tips  = motion[:, 0, TIPS, :]
    right_tips = motion[:, 1, TIPS, :]

    # Calculate distance between every left tip and every right tip
    diff  = left_tips[:,:,np.newaxis,:] - right_tips[:,np.newaxis,:,:]
    dists = np.linalg.norm(diff, axis=-1).min(axis=(1,2))

    # A frame counts as contact if any pair of tips is within 2cm
    in_contact = dists < threshold_m

    # Find runs of consecutive contact frames (contact events)
    events, start = [], None
    for i, c in enumerate(in_contact):
        if c and start is None: start = i
        elif not c and start is not None:
            events.append((start, i-1))
            start = None
    if start is not None:
        events.append((start, len(in_contact)-1))

    if not events:
        return 'No inter-hand contact detected.'

    parts = [f'frames {s}-{e} ({s/fps:.2f}s-{e/fps:.2f}s)' for s,e in events]
    return f'Contact in {len(events)} event(s): ' + '; '.join(parts)


# ── Step 3: Build the full kinematic context block ──
# Combines hand descriptions + contact description into one dict.
# This dict is sent to Qwen as factual context alongside the frames.
def build_kinematic_context(npy_path, fps, contact_ratio, avg_dur, freq):
    motion = np.load(npy_path)   # load the (60, 2, 21, 3) array
    return {
        'left':    _describe_hand(motion, 0, 'left'),
        'right':   _describe_hand(motion, 1, 'right'),
        'contact': _describe_contact(motion, fps),
        'summary': f'contact_ratio={contact_ratio:.3f} | avg_dur={avg_dur:.2f}s | freq={freq:.2f}/s',
    }


# ── Build the text prompt sent to Qwen ──
# Assembles the kinematic context and candidate list into
# a clear instruction for Qwen to follow.
def build_prompt(kinematic, candidates, duration, hands):
    hand_str = ', '.join(hands) if hands else 'none'
    cand_str = format_candidates_for_prompt(candidates)

    p  = f'Analyze this hand gesture ({duration:.2f}s, hands: {hand_str}).' + chr(10)
    p += f'Left:    {kinematic["left"]}' + chr(10)
    p += f'Right:   {kinematic["right"]}' + chr(10)
    p += f'Contact: {kinematic["contact"]}' + chr(10)
    p += f'Summary: {kinematic["summary"]}' + chr(10)
    p += 'Choose the BEST matching action:' + chr(10)
    p += cand_str + chr(10)
    p += 'Reply ONLY with JSON: {"id": "...", "label": "...", "confidence": 0.00}'
    return p


# ── Extract 3 representative frames from the video clip ──
# We pick frame 1 (start), frame 30 (middle), frame 60 (end).
# These 3 frames give Qwen the hand shape at the beginning,
# middle, and end of the gesture — enough visual context.
def extract_frames(video_path, start_frame, end_frame, out_dir):
    import cv2
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap   = cv2.VideoCapture(video_path)
    paths = []

    # Pick first, middle, and last frame
    for i, idx in enumerate([start_frame, (start_frame+end_frame)//2, end_frame]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            p = str(Path(out_dir) / f'frame_{i}.jpg')
            cv2.imwrite(p, frame)
            paths.append(p)
    cap.release()
    return paths


# ── Step 4: Send frames + prompt to Qwen and get a label ──
# Qwen reads the 3 frames as images and the kinematic text,
# then picks the best action from the candidate list.
# Returns a dict like: {'id': 'MAC_004', 'label': 'reach', 'confidence': 0.82}
def call_qwen(prompt, frame_paths, cfg):
    import torch
    from qwen_vl_utils import process_vision_info

    # Build the message Qwen expects: images first, then text
    messages = [{
        'role': 'user',
        'content': [
            *[{'type': 'image', 'image': f'file://{fp}'} for fp in frame_paths],
            {'type': 'text',  'text': prompt}
        ]
    }]

    # Convert messages into model input format
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(_model.device)

    # Run Qwen on the GPU
    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=cfg.get('qwen_max_new_tokens', 256),
            temperature=cfg.get('qwen_temperature', 0.1),
            do_sample=True,
        )

    # Decode the output — strip the input tokens, keep only the response
    trimmed  = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # Parse the JSON from the response
    raw = response.strip().lstrip('```json').rstrip('```').strip()
    return json.loads(raw[raw.find('{'):raw.rfind('}')+1])


# ── Annotate one chunk record ──
# This is the main function called for each chunk.
# It runs all 4 steps and writes the result back into the record.
def annotate_record(record, lib, cfg, video_dir, dry_run=False):

    # Step 1 — movement scale (no model)
    record['movement_scale'] = classify_movement_scale(record, cfg)

    # If dry_run=True, stop here (useful for testing without GPU)
    if dry_run:
        return record

    # Step 2 — filter candidates (no model)
    ratio      = record.get('contact_ratio', 0.0) or 0.0
    candidates = filter_candidates(
        lib, ratio,
        record['movement_scale'],
        record.get('hands_detected', [])
    )
    if not candidates:
        candidates = lib['actions'][:6]   # fallback: use first 6 actions

    # Step 3 — kinematic description (no model)
    kinematic = build_kinematic_context(
        record['npy_path'],
        record['fps'],
        ratio,
        record.get('avg_contact_duration_sec', 0.0),
        record.get('contact_frequency_per_sec', 0.0)
    )
    record['annotation'] = kinematic

    # Step 4 — call Qwen (GPU)
    prompt  = build_prompt(kinematic, candidates,
                           record.get('duration_sec', 2.0),
                           record.get('hands_detected', []))
    vid     = str(Path(video_dir) / record['video_source'])
    fdir    = str(Path('outputs/tmp_frames') / record['chunk_id'])
    fpaths  = extract_frames(vid, record['start_frame'], record['end_frame'], fdir)

    try:
        result = call_qwen(prompt, fpaths, cfg)
        action = lib['by_id'].get(result.get('id', ''), {})

        record['action_id']    = result.get('id', 'unknown')
        record['action_label'] = result.get('label', action.get('action_label', 'unknown'))
        record['confidence']   = float(result.get('confidence', 0.0))
        record['annotated']    = True

    except Exception as e:
        # If Qwen fails on a chunk, mark it unknown and continue
        print(f'    Qwen error on {record["chunk_id"]}: {e}')
        record['action_id']    = 'unknown'
        record['action_label'] = 'unknown'
        record['confidence']   = 0.0
        record['annotated']    = False

    return record
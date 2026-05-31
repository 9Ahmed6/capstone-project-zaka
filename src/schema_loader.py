# src/schema_loader.py
# Loads the action library and settings.
# Every other file imports from here.
import json, yaml
from pathlib import Path

def load_config(path='configs/settings.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_action_library(path=None):
    if path is None:
        path = load_config()['action_library_path']
    with open(path) as f:
        raw = json.load(f)
    actions = raw if isinstance(raw, list) else raw.get('actions', [])
    for i, a in enumerate(actions):
        if 'id' not in a:
            a['id'] = a.get('movement_scale','unk')[:3].upper() + f'_{i:03d}'
        if 'description' not in a:
            a['description'] = a.get('aliases',[''])[0]
        if 'contact_ratio_range' not in a:
            if a.get('movement_scale') == 'micro':
                a['contact_ratio_range'] = [0.0, 0.25]
            elif a.get('movement_scale') == 'bimanual':
                a['contact_ratio_range'] = [0.30, 1.0]
            else:
                a['contact_ratio_range'] = [0.0, 0.40]
    return {'actions': actions, 'by_id': {a['id']: a for a in actions}}

def get_actions_by_scale(lib, scale):
    return [a for a in lib['actions'] if a['movement_scale'] == scale]

def load_output_schema(path=None):
    if path is None:
        path = load_config()['output_schema_path']
    with open(path) as f:
        return json.load(f)

def filter_candidates(lib, contact_ratio, movement_scale, hands_detected):
    candidates = []
    tol = 0.05
    adjacent = {'micro':['macro'],'macro':['micro','bimanual'],'bimanual':['macro']}
    for a in lib['actions']:
        if a['movement_scale'] != movement_scale:
            if a['movement_scale'] not in adjacent.get(movement_scale, []):
                continue
        lo, hi = a.get('contact_ratio_range', [0.0, 1.0])
        if not (lo - tol <= contact_ratio <= hi + tol):
            continue
        if a.get('hand_side') == 'both' and len(hands_detected) < 2:
            continue
        candidates.append(a)
    return candidates

def format_candidates_for_prompt(candidates):
    lines = []
    for i, c in enumerate(candidates):
        label = c.get('action_label', c.get('label', '?'))
        desc  = c.get('description', '')
        lines.append(f"{i+1}. [{c['id']}] {label} — {desc}")
    return chr(10).join(lines)
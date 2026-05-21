import numpy as np
import pathlib

GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

JOINT_NAME_INDEX_MAP = {
    'wrist': 0,
    'thumb_cmc': 13, 'thumb_mcp': 14, 'thumb_ip': 15, 'thumb_tip': 16,
    'index_mcp': 1, 'index_pip': 2, 'index_dip': 3, 'index_tip': 17,
    'middle_mcp': 4, 'middle_pip': 5, 'middle_dip': 6, 'middle_tip': 18,
    'ring_mcp': 10, 'ring_pip': 11, 'ring_dip': 12, 'ring_tip': 19,
    'pinky_mcp': 7, 'pinky_pip': 8, 'pinky_dip': 9, 'pinky_tip': 20,
}
JOINT_INDEX_NAME_MAP = {v: k for k, v in JOINT_NAME_INDEX_MAP.items() }
SKELETON_CHAIN = np.array([
    [0, 13, 14, 15, 16],
    [0, 1, 2, 3, 17],
    [0, 4, 5, 6, 18],
    [0, 10, 11, 12, 19],
    [0, 7, 8, 9, 20]
])
SKELETON_CHAIN_NAME = ['thumb', 'index', 'middle', 'ring', 'pinky']

MANO_MODEL_DIR = pathlib.Path(__file__).parent.parent / "body_models" / "mano"

TEXT_MODEL_DIMS = {
    't5-small': 512,
    "t5-base": 768,
    "t5-large": 1024,
    "openai/clip-vit-base-patch32": 512
}

gesture_list = [
    "Pinch-Fingers",
    "Freestyle",
    "Palm-Touch",
    "Object-Interact",
    "Flip-Hands",
    "Two-Hand-Interact",
    "Phone-Interact",
    "Almost-Pinch-25mm",
    "Grab-Drag",
    "Almost-Pinch-10mm",
    "G-Pinch",
    "Hyperextension",
    "Bend-Palm",
    "Pinch-Index",
    "Write-Text",
    "Index-Slider",
    "Permanent-Pinch",
    "C-Pinch",
    "Hoverbeats",
    "Palm-Bend",
    "Flip-Hands-Fingers",
    "Almost-Pinch-Fingers",
    "Open-Hands",
    "Pinch-Drag",
    "Permanent-Almost-Pinch",
    "Almost-Palm-Touch",
    "Hoverbeat",
    "Close-Up",
    "Counting",
    "Object-Interact-Tag",
    "Pointing",
    "Pointing-Index",
    "Almost-Pinch-20mm",
    "Almost-Pinch-Index",
    "Fist",
    "Permanent-Almost-Pinch-Index",
    "Rub-Fingers",
    "Rubbing-Fingers",
    "Wanding",
    "Grasp-Drag",
    "Hyperextention",
    "Open-Hand",
]

INTRA_TIP_CONTACT_THRESH = 0.02
TIP_PALM_CONTACT_THRESH = 0.025
PALM_PALM_CONTACT_THRESH = 0.025
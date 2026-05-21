import numpy as np
import json
from typing import List
from os.path import join as pjoin

from .feature.bihand_motioncode import BihandMotionCoder
from .llm.llm_helper import get_llm_response, insert_json_to_prompt, insert_action_and_ori_descri_to_prompt, parse_json_from_response


def generate_annotation(skeleton_motion: np.ndarray, prompt_template:str, model: str='deepseek-reasoner', return_json: bool=False) -> str:
    bihand_motioncoder = BihandMotionCoder(skeleton_motion)
    bihand_motioncoder.generate_motion_codes()
    feature_json = bihand_motioncoder.get_json()

    prompt = insert_json_to_prompt(feature_json, prompt_template)
    annotation = get_llm_response(
        prompt,
        model=model,
    )

    if return_json:
        return parse_json_from_response(annotation)
    else:
        return annotation

def rephrase_annotation(action_label:str, original_description:str, prompt_template:str, model: str='gemini-2.5-pro') -> List[str]:
    """
    Rephrase the original description based on the action label using a language model.

    :param action_label: The action label to be included in the prompt.
    :param original_description: The original description to be rephrased.
    :param prompt_template: The template for the prompt.
    :param model: The model to use for rephrasing.
    :return: A list of rephrased descriptions.
    """
    prompt = insert_action_and_ori_descri_to_prompt(action_label, original_description, prompt_template)
    response = get_llm_response(prompt, model)

    return list(parse_json_from_response(response).values())  # Assuming this function is defined elsewhere

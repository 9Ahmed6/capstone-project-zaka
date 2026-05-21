import openai
import numpy as np
import os, json
from tqdm import tqdm
from google import genai
from ..constant import DEEPSEEK_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY


def get_gemini_response(prompt, model: str, api_key=GEMINI_API_KEY):
    """
    Calls Google's Gemini API to get a response for the given prompt.

    :param prompt: User's input prompt
    :param model: Gemini model to use, default is "gemini-pro"
    :param api_key: Google API key
    :return: The generated response text from Gemini
    """
    os.environ['GEMINI_API_KEY'] = api_key
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        print(f"Input Token Count: {response.usage_metadata.prompt_token_count}")
        print(f"Output Token Count: {response.usage_metadata.candidates_token_count}")
        return response.text
    except Exception as e:
        return f"Error: {e}"


def get_gpt_response(prompt, model: str, api_key):
    """
    Calls OpenAI GPT API to get a response for the given prompt.

    :param prompt: User's input prompt
    :param model: GPT model to use, default is "gpt-4"
    :param api_key: OpenAI API key
    :return: The generated response text from GPT
    """
    try:
        client = openai.OpenAI(api_key=api_key)  # Create an OpenAI client
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def get_llm_response(prompt: str, model: str):
    if model.startswith('gemini'):
        return get_gemini_response(prompt, model, GEMINI_API_KEY)
    elif model.startswith('deepseek'):
        return get_gpt_response(prompt, model, DEEPSEEK_API_KEY)
    else:
        return get_gpt_response(prompt, model, OPENAI_API_KEY)


def insert_json_to_prompt(json_data, prompt_template):
    json_str = json.dumps(json_data, indent=4)
    return prompt_template.replace("[INSERT HERE]", json_str)

def insert_long_winded_annotation_to_prompt(annotation:str, prompt_template):
    return prompt_template.replace("[INSERT HERE]", annotation)

def insert_action_and_ori_descri_to_prompt(action:str, ori_descri:str, prompt_template):
    ret = prompt_template.replace("[ACTION LABEL]", action)
    ret = ret.replace("[ORIGINAL DESCRIPTION]", ori_descri)
    return ret

def parse_json_from_response(response:str):
    if response.startswith("```json"):
        response = response[7:]
        response = response[:response.rfind("```")]
    try:
        json_response = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}, response was: {response}")
    return json_response
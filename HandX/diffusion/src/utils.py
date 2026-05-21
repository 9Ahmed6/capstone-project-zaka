import numpy as np
import pickle, json
import textwrap

def get_motion_data_boundary(motion:np.ndarray):
    '''
    motion: (T, J, 3)
    '''
    xmin = np.min(motion[:, :, 0])
    xmax = np.max(motion[:, :, 0])
    ymin = np.min(motion[:, :, 1])
    ymax = np.max(motion[:, :, 1])
    zmin = np.min(motion[:, :, 2])
    zmax = np.max(motion[:, :, 2])
    return xmin, xmax, ymin, ymax, zmin, zmax


def load_npy(file_path):
    return np.load(file_path)


def save_npy(data, file_path):
    np.save(file_path, data)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_txt(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def start_debug():
    import debugpy
    debugpy.listen(
        ('0.0.0.0', 5678)  # Change the port if needed
    )
    print("--- Debugpy is listening on port 5678. Waiting for client to attach... ---")
    debugpy.wait_for_client()
    print("--- Client attached. Continuing execution... ---")

def debug():
    import code
    code.interact(local=locals())

def smart_wrap(text, width=70):
    """智能换行:保留原有段落结构,对每段分别换行"""
    paragraphs = text.split('\n')
    wrapped_paragraphs = []

    for para in paragraphs:
        if para.strip():  # 非空段落
            wrapped = textwrap.fill(para, width=width)
            wrapped_paragraphs.append(wrapped)
        else:  # 空行保留
            wrapped_paragraphs.append('')

    return '\n'.join(wrapped_paragraphs)
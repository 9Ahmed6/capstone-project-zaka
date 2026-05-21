import torch

def lengths_to_mask(lengths, max_len) -> torch.Tensor:
    '''
    lengths: [B]
    '''
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1) # (B, max_len)
    return mask


def collate_tensors(batch) -> torch.Tensor: # 合并不同长度的 tensor 为一个 batch
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)] # (dims,)
    size = (len(batch),) + tuple(max_size) # (B, max_size[0], max_size[1], ...)
    canvas = batch[0].new_zeros(size=size) # (B, max_size[0], max_size[1], ...)
    for i, b in enumerate(batch):
        subtensor = canvas[i]
        for d in range(dims):
            sub_tensor = subtensor.narrow(d, 0, b.size(d)) # 从一个 Tensor 中提取一个子区域，并返回一个视图（与原来的 Tensor 共享底层存储）
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches] # TODO: to be checked!!!

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # (B, 1, 1, max_seq_len)

    motion = databatchTensor
    cond = dict(
        y=dict(
            mask=maskbatchTensor, # (B, 1, 1, max_seq_len)
            lengths=lenbatchTensor # (B)
        )
    )

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({"text" : textbatch})

    if 'contact_label' in notnone_batches[0]:
        contact_label_batch = [b['contact_label'] for b in notnone_batches]
        cond['y'].update({"contact_label" : collate_tensors(contact_label_batch)})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({"tokens" : textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = torch.tensor([b['action'] for b in notnone_batches]).long()
        cond['y'].update({"action" : actionbatch})

    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({"action_text" : action_text})

    if 'prefix' in notnone_batches[0]:
        cond['y'].update(dict(
            prefix=collate_tensors([
                b['prefix']
                for b in notnone_batches
            ])
        ))

    return motion, cond

def motion_action_collate(batch):
    adapted_batch = [dict(
        inp=torch.tensor(b[0]).float(), # (T, 2J, 3)
        lengths=b[1],
        **(dict(
            action=b[2]
        ) if len(b)>2 else dict())
    ) for b in batch]
    return collate(adapted_batch)

def motion_text_collate(batch):
    adapted_batch = [
        dict(
            inp=torch.tensor(b[0]).float(), # (T, 2J, 3)
            lengths=b[1],
            text=b[2]
        )
        for b in batch
    ]
    return collate(adapted_batch)

def motion_text_treble_collate(batch):
    """
    Collate function for three-part text annotations.
    Each text is a dict with keys: 'left', 'right', 'two_hands_relation'
    """
    adapted_batch = [
        dict(
            inp=torch.tensor(b[0]).float(), # (T, 2J, 3)
            lengths=b[1],
            text=b[2]  # b[2] is a dict: {'left': str, 'right': str, 'two_hands_relation': str}
        )
        for b in batch
    ]
    if len(batch[0]) == 4:
        for i in range(len(adapted_batch)):
            adapted_batch[i]['contact_label'] = torch.tensor(batch[i][3])  # (T, N_contacts)

    motion, cond = collate(adapted_batch)

    # Reorganize text from list of dicts to dict of lists
    if 'text' in cond['y']:
        text_batch = cond['y']['text']
        if isinstance(text_batch[0], dict):
            cond['y']['text'] = {
                'left': [t['left'] for t in text_batch],
                'right': [t['right'] for t in text_batch],
                'two_hands_relation': [t['two_hands_relation'] for t in text_batch]
            }



    return motion, cond


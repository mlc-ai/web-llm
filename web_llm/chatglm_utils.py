import torch
import tvm
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper
)


def prepare_inputs_chatglm(
            input_ids: torch.Tensor, device: tvm.device, tokens: torch.Tensor, is_encoder: bool
    ):
        mask_token_id = 130000
        gmask_token_id = 130001
        bos_token_id = 130004
        
        batch_size, seq_length = tokens.shape
        MASK, gMASK = mask_token_id, gmask_token_id
        seqs = tokens.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        if not is_encoder:
            attention_mask = torch.zeros(1, 1).bool()
            context_lengths = [seq.index(bos_token_id) for seq in seqs]
            position_ids = torch.tensor(
                [[mask_position, seq_length - context_length] for mask_position, context_length in
                    zip(mask_positions, context_lengths)], dtype=torch.long).unsqueeze(-1)
        else:
            attention_mask = get_masks(
                tokens,
            )
            position_ids = get_position_ids(
                tokens,
                mask_positions=mask_positions,
                use_gmasks=use_gmasks
            )
            
        input_ids = tvm.nd.array(input_ids.numpy(), device=device)
        position_ids = tvm.nd.array(position_ids.numpy(), device=device)
        attention_mask = tvm.nd.array(attention_mask.numpy(), device=device)
        return (input_ids, position_ids, attention_mask)

   
def get_masks(input_ids):
    bos_token_id = 130004
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(bos_token_id) for seq in input_ids]
    attention_mask = torch.ones((batch_size, seq_length, seq_length))
    attention_mask.tril_()
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def get_position_ids(input_ids, mask_positions, use_gmasks=None):
    bos_token_id = 130004
    batch_size, seq_length = input_ids.shape
    if use_gmasks is None:
        use_gmasks = [False] * batch_size
    context_lengths = [seq.tolist().index(bos_token_id) for seq in input_ids]
    
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    for i, context_length in enumerate(context_lengths):
        position_ids[i, context_length:] = mask_positions[i]
    block_position_ids = [torch.cat((
        torch.zeros(context_length, dtype=torch.long),
        torch.arange(seq_length - context_length, dtype=torch.long) + 1
    )) for context_length in context_lengths]
    block_position_ids = torch.stack(block_position_ids, dim=0)
    position_ids = torch.stack((position_ids, block_position_ids), dim=1)

    return position_ids


def process_logits_chatglm(last_token_logits):
    # Invalid Score
    if torch.isnan(last_token_logits).any() or torch.isinf(last_token_logits).any():
        last_token_logits.zeros_()
        last_token_logits[..., 20005] = 5e4

    # topk and topp
    warpers = LogitsProcessorList()
    warpers.append(TemperatureLogitsWarper(0.95))
    warpers.append(TopKLogitsWarper(50))
    warpers.append(TopPLogitsWarper(0.7))

    last_token_logits = warpers(None, last_token_logits)
    return last_token_logits

def process_response(response):
    import re
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response
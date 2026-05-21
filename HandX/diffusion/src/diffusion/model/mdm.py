from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5EncoderModel

from ...constant import TEXT_MODEL_DIMS
from ..metric.interaction import intra_tip_pairs, all_tips

class MotionDiffusionModel(nn.Module):
    def __init__(
        self, arch, latent_dim, num_heads, ff_size, dropout, activation, num_layers,
        njoints, nfeats,
        cond_mode, cond_mask_prob,
        treble_mask_prob=1.0,  # New parameter: probability to keep each treble branch (1.0 = no masking)
        *args, **kwargs
    ):
        super(MotionDiffusionModel, self).__init__()
        self.arch = arch
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.njoints = njoints
        self.nfeats = nfeats

        self.cond_mode = cond_mode
        if self.cond_mode != 'no_cond':
            self.cond_mask_prob = cond_mask_prob
            if self.cond_mask_prob < 0 or self.cond_mask_prob > 1:
                raise ValueError(f"cond_mask_prob should be in [0, 1], but got {self.cond_mask_prob}")

        # Treble masking probability for trans_dec_treble_residual
        self.treble_mask_prob = treble_mask_prob
        if self.treble_mask_prob < 0 or self.treble_mask_prob > 1:
            raise ValueError(f"treble_mask_prob should be in [0, 1], but got {self.treble_mask_prob}")

        self.input_process = InputProcess(
            input_feats=njoints * nfeats,
            latent_dim=latent_dim
        )
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim,
            self.dropout
        )

        if self.cond_mode == 'text':
            self.text_model_name = kwargs['text_model_name']
            self.text_max_length = kwargs['text_max_length']
            if self.text_model_name.startswith("t5"):
                self.text_tokenizer = T5Tokenizer.from_pretrained(self.text_model_name)
                self._text_model = T5EncoderModel.from_pretrained(self.text_model_name)

                if kwargs.get("finetune_text_model", False):

                    if self.arch == 'trans_dec_treble_concat':
                        special_tokens_to_add = ['[LEFT]', '[RIGHT]', '[TWO_HANDS_RELATION]']
                        self.text_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
                        self._text_model.resize_token_embeddings(len(self.text_tokenizer))

                    lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.1,
                        bias="none",
                    )

                    self._text_model = get_peft_model(self._text_model, lora_config)
                    if self.arch == 'trans_dec_treble_concat':
                        for name, param in self._text_model.named_parameters():
                            # T5的词嵌入层名为 'shared'
                            if 'shared' in name:
                                param.requires_grad = True

                else:
                    for param in self._text_model.parameters():
                        param.requires_grad = False

            elif 'clip' in self.text_model_name:
                self.text_processor = CLIPProcessor.from_pretrained(self.text_model_name, local_files_only=True)
                self._text_model = CLIPModel.from_pretrained(self.text_model_name, use_safetensors=True, local_files_only=True)
                for param in self._text_model.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError(f"Text model {self.text_model_name} is not implemented.")

            self.text_embedding_project = nn.Linear(
                TEXT_MODEL_DIMS[self.text_model_name], self.latent_dim
            )
        elif self.cond_mode == 'action':
            self.embed_action = EmbedAction(kwargs['num_actions'], self.latent_dim)


        if self.arch == 'trans_enc':
            print("Transformer Encoder initialize.")
            seq_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
            )
            self.seq_trans_encoder = nn.TransformerEncoder(
                seq_trans_encoder_layer,
                num_layers=self.num_layers
            )
        elif self.arch.startswith('trans_dec'):
            print("Transformer Decoder initialize.")
            seq_trans_decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
            )
            self.seq_trans_decoder = nn.TransformerDecoder(
                seq_trans_decoder_layer,
                num_layers=self.num_layers
            )
            self.null_text_embedding = nn.Parameter(torch.randn(1, 1, self.latent_dim)) # (1, 1, D)
            if self.arch.startswith('trans_dec_treble'):
                self.left_hand_cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim)) # (1, 1, D)
                self.right_hand_cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim)) # (1, 1, D)
                self.two_hands_relation_cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim)) # (1, 1, D)
        else:
            raise NotImplementedError(f"Architecture {self.arch} is not implemented.")

        self.contact_prediction = kwargs.get("contact_prediction", False)
        if self.contact_prediction:
            contact_predict_decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True
            )
            self.contact_predict_decoder = nn.TransformerDecoder(
                contact_predict_decoder_layer,
                num_layers=self.num_layers
            )
            self.contact_predict_head = nn.Linear(self.latent_dim, (len(all_tips) + len(intra_tip_pairs)) * 2 + 1)

        self.embed_timestep = TimestepEmbedder(
            latent_dim=self.latent_dim,
            positional_encode=self.sequence_pos_encoder.pe.squeeze(0) # (max_len, D)
        )

        self.output_process = OutputProcess(
            latent_dim=self.latent_dim,
            njoints=self.njoints,
            nfeats=self.nfeats
        )

        self.apply(self._init_weights)

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_cond_mask(self, batch_size, device):
        if self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(batch_size, device=device) * self.cond_mask_prob
            ) # (B,)
            return (1 - mask).bool() # (B,)
        else:
            return torch.ones(batch_size, device=device).bool() # (B,)

    def get_text_embeddings(self, texts : List[str]) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.text_model_name.startswith("t5"):
            text_inputs = self.text_tokenizer(
                texts, padding=True, truncation=True, max_length=self.text_max_length,
                return_tensors='pt'
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            outputs = self._text_model(**text_inputs)
            return self.text_embedding_project(outputs.last_hidden_state), text_inputs['attention_mask']
        elif 'clip' in self.text_model_name:
            inputs = self.text_processor(
                texts, return_tensors='pt', padding=True, truncation=True, max_length=self.text_max_length
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            text_features = self._text_model.get_text_features(**inputs)
            return self.text_embedding_project(text_features)
        else:
            raise NotImplementedError(f"Text model {self.text_model_name} is not implemented.")

    def cross_attention_w_single_text(self, motion_embedding: torch.Tensor, single_text_list: List[str], cls_tokens: List[torch.Tensor], decoder: nn.TransformerDecoder, y_lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = motion_embedding.shape
        if not self.text_model_name.startswith("t5"):
            raise NotImplementedError(f"Text model {self.text_model_name} is not implemented for cross attention with single text.")
        text_embeddings, text_padding_mask = self.get_text_embeddings(single_text_list) # (B, L_Text, D), (B, L_Text)
        if self.training:
            condition_mask = self.get_cond_mask(B, motion_embedding.device) # (B,)
            text_embeddings = torch.where(
                condition_mask.view(B, 1, 1), text_embeddings, self.null_text_embedding
            )
            text_padding_mask = torch.where(
                condition_mask.view(B, 1), text_padding_mask, torch.ones((1, 1), device=motion_embedding.device)
            )

        text_padding_mask = (text_padding_mask == 0) # True means this position SHOULD be masked.

        xseq = torch.cat([*cls_tokens, motion_embedding], dim=1) # (B, len(cls_tokens)+T, D)
        xseq = self.sequence_pos_encoder(xseq) # (B, len(cls_tokens)+T, D)
        tgt_key_padding_mask = torch.arange(len(cls_tokens) + T, device=motion_embedding.device)[None, :] - len(cls_tokens) >= y_lengths[:, None] # (B, len(cls_tokens)+T)
        output = decoder(
            tgt=xseq, # (B, len(cls_tokens)+T, D)
            memory=text_embeddings, # (B, L_Text, D)
            tgt_key_padding_mask=tgt_key_padding_mask, # (B, len(cls_tokens)+T)
            memory_key_padding_mask=text_padding_mask, # (B, L_Text)
        )[:, len(cls_tokens):] # (B, T, D)
        return output

    def _forward(self, x, time_embedder_token, y:dict=None, decoder:nn.TransformerDecoder=None, *arg, **kwargs):
        '''
        x: (B, T, D)
        time_embedder_token: (B, 1, D)
        '''
        B, T, _ = x.shape


        if self.arch == 'trans_enc': # Deprecated
            raise DeprecationWarning("Transformer Encoder architecture is deprecated. Please use Transformer Decoder architectures.")
            if 'text' in self.cond_mode and not y.get("uncond", False):
                if 'clip' in self.text_model_name:
                    text_embeddings = self.get_text_embeddings(y['text']) # (B, D)
                    if self.training:
                        condition_mask = self.get_cond_mask(B, x.device) # (B,)
                        text_embeddings = text_embeddings * condition_mask[:, None] # (B, D)
                    time_embedder_token = time_embedder_token + text_embeddings[:, None, :] # (B, 1, D)
                else:
                    raise NotImplementedError(f"Text model {self.text_model_name} is not implemented for Transformer Encoder.")
            elif 'action' in self.cond_mode and not y.get("uncond", False):
                action_embeddings = self.embed_action(y['action']) # (B, D)
                if self.training:
                    condition_mask = self.get_cond_mask(B, x.device) # (B,)
                    action_embeddings = action_embeddings * condition_mask[:, None] # (B, D)
                time_embedder_token = time_embedder_token + action_embeddings[:, None, :] # (B, 1, D)

            xseq = torch.cat([time_embedder_token, x], dim=1) # (B, T+1, D)
            xseq = self.sequence_pos_encoder(xseq) # (B, T+1, D)
            src_key_padding_mask = torch.arange(T + 1, device=x.device)[None, :] - 1 >= y['lengths'][:, None] # (B, T+1)
            output = self.seq_trans_encoder(
                xseq, src_key_padding_mask=src_key_padding_mask
            )[:, 1:] # (B, T, D)

        elif self.arch == 'trans_dec':
            assert 'text' in self.cond_mode, "Transformer Decoder requires text condition."
            output = self.cross_attention_w_single_text(x, y['text'], [time_embedder_token], decoder=decoder, y_lengths=y['lengths']) # (B, T, D)

        elif self.arch.startswith('trans_dec_treble'):
            assert 'text' in self.cond_mode, "Transformer Decoder with treble residual requires text condition."
            assert 'text' in y and 'left' in y['text'] and 'right' in y['text'] and 'two_hands_relation' in y['text'], "Transformer Decoder with treble residual requires 'left', 'right' and 'two_hands_relation' text conditions."
            if self.arch == 'trans_dec_treble_residual':
                # Random masking: each branch has treble_mask_prob probability to be kept
                if self.training:
                    left_keep = torch.bernoulli(torch.ones(B, device=x.device) * self.treble_mask_prob).bool().cpu().tolist()  # (B,)
                    right_keep = torch.bernoulli(torch.ones(B, device=x.device) * self.treble_mask_prob).bool().cpu().tolist()  # (B,)
                    relation_keep = torch.bernoulli(torch.ones(B, device=x.device) * self.treble_mask_prob).bool().cpu().tolist()  # (B,)

                    masked_left_text = [text if keep else "" for text, keep in zip(y['text']['left'], left_keep)]
                    masked_right_text = [text if keep else "" for text, keep in zip(y['text']['right'], right_keep)]
                    masked_relation_text = [text if keep else "" for text, keep in zip(y['text']['two_hands_relation'], relation_keep)]
                else:
                    masked_left_text = y['text']['left']
                    masked_right_text = y['text']['right']
                    masked_relation_text = y['text']['two_hands_relation']

                left_hand_output = self.cross_attention_w_single_text(x, masked_left_text, [time_embedder_token, self.left_hand_cls_token.expand(B, -1, -1)], decoder=decoder, y_lengths=y['lengths']) # (B, T, D)
                right_hand_output = self.cross_attention_w_single_text(x, masked_right_text, [time_embedder_token, self.right_hand_cls_token.expand(B, -1, -1)], decoder=decoder, y_lengths=y['lengths']) # (B, T, D)
                two_hands_relation_output = self.cross_attention_w_single_text(x, masked_relation_text, [time_embedder_token, self.two_hands_relation_cls_token.expand(B, -1, -1)], decoder=decoder, y_lengths=y['lengths']) # (B, T, D)

                output = x + left_hand_output + right_hand_output + two_hands_relation_output # (B, T, D)

            elif self.arch == 'trans_dec_treble_concat':
                concated_texts = []
                for left_text, right_text, two_hands_relation_text in zip(y['text']['left'], y['text']['right'], y['text']['two_hands_relation']):
                    concated_texts.append('[LEFT] ' + left_text + ' [RIGHT] ' + right_text + ' [TWO_HANDS_RELATION] ' + two_hands_relation_text)

                output = self.cross_attention_w_single_text(x, concated_texts, [time_embedder_token], decoder=decoder, y_lengths=y['lengths']) # (B, T, D)


        else:
            raise NotImplementedError(f"Architecture {self.arch} is not implemented.")

        return output

    def forward(self, x, timesteps, y:dict=None, predict_contact=False, *arg, **kwargs):
        cls_token = self.embed_timestep(timesteps) # (B, 1, D)
        if not predict_contact:
            x = self.input_process(x) # (B, T, D)
            output = self._forward(x, cls_token, y=y, decoder=self.seq_trans_decoder, *arg, **kwargs)
            output = self.output_process(output) # (B, J, feats_per_joint, T)
            return output
        else:
            motion_embedding = self.input_process(x.clone().detach())
            output = self._forward(motion_embedding, cls_token.clone().detach(), y=y, decoder=self.contact_predict_decoder, *arg, **kwargs) # (B, T, D)
            contact_logits = self.contact_predict_head(output) # (B, T, P)
            return contact_logits


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        '''
        input_feats: num_joints * feat_dim_per_joint
        '''
        super(InputProcess, self).__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pose_embedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        '''
        x: (B, J, D, T)
        '''
        B, J, D, T = x.shape
        x = rearrange(x, 'b j d t -> b t (j d)') # (B, T, J*D)
        x = self.pose_embedding(x) # (B, T, D)
        return x

class OutputProcess(nn.Module):
    def __init__(self, latent_dim, njoints, nfeats):
        '''
        nfeats: feature dimension per joint
        '''
        super(OutputProcess, self).__init__()
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.pose_head = nn.Linear(self.latent_dim, self.njoints * self.nfeats)

    def forward(self, x):
        '''
        x: (B, T, D)
        '''
        B, T, D = x.shape
        x = self.pose_head(x)
        ret = rearrange(x, 'b t (j d) -> b j d t', j=self.njoints, d=self.nfeats) # (B, J, D, T)
        return ret

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer('pe', pe) # 不会计算梯度，但是会保存到state_dict中，移动设备时也会随模型一起移动

    def forward(self, x):
        '''
        x: (B, T, D)
        '''
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, positional_encode:torch.Tensor):
        '''
        positional_encode: (max_len, D)
        '''
        super(TimestepEmbedder, self).__init__()
        self.latent_dim = latent_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.register_buffer(
            'positional_encode',
            positional_encode
        )

    def forward(self, timesteps) -> torch.Tensor:
        '''
        timesteps: (B,)
        '''
        return self.time_embedding(
            self.positional_encode[timesteps] # (B, D)
        ).unsqueeze(1) # (B, 1, D)


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super(EmbedAction, self).__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input:torch.Tensor):
        '''
        input: (B,)
        '''
        idx = input.long()
        output = self.action_embedding[idx]
        return output
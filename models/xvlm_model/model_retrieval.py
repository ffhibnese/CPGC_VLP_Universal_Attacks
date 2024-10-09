import torch
from models.xvlm_model import XVLMBase, load_pretrained
import torch.nn.functional as F

class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx=None):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        return loss_itc, loss_itm
    
    def inference_text(self, text_input):
        text_embed = self.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_feat': text_feat, 'text_embed': text_embed}
    
    def inference_text_replace(self, text_input, embedding, position, padding_zero):
        for i in range(len(padding_zero)):
            pos = text_input.attention_mask[i].argmin().item()
            zero_num = sum(padding_zero[i])
            if zero_num != 0:
                if pos == 0:
                    text_input.attention_mask[i][-zero_num:] = 0
                else:
                    text_input.attention_mask[i][pos-zero_num: pos] = 0
        text_embed = self.get_text_embeds(text_input.input_ids, text_input.attention_mask, replace_embedding = embedding, replace_position = position, padding_zero = padding_zero)
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_feat': text_feat, 'text_embed': text_embed}

    def inference_image(self, image):
        image_embed, _ = self.get_vision_embeds(image)
        image_feat = F.normalize(self.vision_proj(image_embed[:, 0, :]), dim=-1)
        return {'image_feat': image_feat, 'image_embed': image_embed}
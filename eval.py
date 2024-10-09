import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.clip_model.simple_tokenizer import SimpleTokenizer
import utils
from utils import load_model, get_filter_words
from attacker import TextAttacker
from dataset import paired_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/Retrieval_flickr_test.yaml')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--text_encoder', default='bert-base-uncased', type=str)

parser.add_argument('--source_model', default='ALBEF', type=str)
parser.add_argument('--checkpoint', default='./checkpoint', type=str)  
parser.add_argument('--original_rank_index_path', default='./std_eval_idx/')  
parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
parser.add_argument('--load_dir', type=str)
parser.add_argument('--epoch', type=int, default=39)
parser.add_argument('--word_num', type=int, default=1)
args = parser.parse_args()

filter_words = get_filter_words()


def retrieval_eval(ALBEF_model, TCL_model, BLIP_model, XVLM_model, CLIP_VIT_model, CLIP_CNN_model, ref_model, tokenizer, blip_tokenizer, extra_transform, target_transform, data_loader, device, config):
    # test
    ALBEF_model.float()
    ALBEF_model.eval()
    TCL_model.float()
    TCL_model.eval()
    BLIP_model.float()
    BLIP_model.eval()
    XVLM_model.eval()
    XVLM_model.float()
    CLIP_VIT_model.float()
    CLIP_VIT_model.eval()
    CLIP_CNN_model.float()
    CLIP_CNN_model.eval()
    ref_model.eval()

    if args.source_model == 'ALBEF':
        source_model = ALBEF_model
    elif args.source_model == 'TCL':
        source_model = TCL_model
    elif args.source_model == 'BLIP':
        source_model = BLIP_model
    elif args.source_model == 'XVLM':
        source_model = XVLM_model  
    elif args.source_model == 'ViT-B/16':
        source_model = CLIP_VIT_model
    elif args.source_model == 'RN101':
        source_model = CLIP_CNN_model

    print('Computing features for evaluation adv...')

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
        load_dir = os.path.join(args.load_dir, args.source_model, config['dataset'])
    elif args.source_model == 'ViT-B/16':
        load_dir = os.path.join(args.load_dir, 'CLIP_VIT', config['dataset'])
    elif args.source_model == 'RN101':
        load_dir = os.path.join(args.load_dir, 'CLIP_CNN', config['dataset'])
    uap_noise = torch.load(os.path.join(load_dir, 'uap_noise-{}.pth'.format(args.epoch)), map_location=device)
    uap_embeddings = torch.load(os.path.join(load_dir, 'uap_embedding-{}.pth'.format(args.epoch)), map_location=device)
    adv_words = []
    for i in range(uap_embeddings.size(0)):
        uap_embedding = uap_embeddings[i]
        if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
            word_idxs = np.load('word_idx_{}.npy'.format(args.source_model))
            word_embeddings = torch.load('word_embedding_{}.pth'.format(args.source_model)).to(device)
        else:
            word_idxs = np.load('word_idx_CLIP.npy')
            word_embeddings = torch.load('word_embedding_CLIP.pth').to(device)
        available_word_embeddings = word_embeddings[word_idxs]
        similarity = torch.stack([torch.dist(uap_embedding, word_embedding, p=2) for word_embedding in available_word_embeddings])
        for i in range(similarity.size(0)):
            values, indices = torch.topk(similarity, k=i+1, largest=False, sorted=True)
            # values, indices = torch.topk(similarity, k=i+1, largest=True, sorted=True)
            index = indices[i]
            min_id = word_idxs[index]
            if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
                word = tokenizer._convert_id_to_token(min_id)
            elif args.source_model in ['ViT-B/16', 'RN101']:
                simple_tokenizer = SimpleTokenizer()
                word = simple_tokenizer.decoder[min_id]
                word = word.replace('</w>', '')
            print(word)
            if word not in filter_words:
                adv_words.append(word)
                break

    print(adv_words)

    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=args.word_num, adv_words=adv_words,
                                topk=10, threshold_pred_score=0.3)

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    CLIP_VIT_image_feats = torch.zeros(num_image, CLIP_VIT_model.visual.output_dim)
    CLIP_VIT_text_feats = torch.zeros(num_text, CLIP_VIT_model.visual.output_dim)

    CLIP_CNN_image_feats = torch.zeros(num_image, CLIP_CNN_model.visual.output_dim)
    CLIP_CNN_text_feats = torch.zeros(num_text, CLIP_CNN_model.visual.output_dim)

    ALBEF_image_feats = torch.zeros(num_image, config['embed_dim'])
    ALBEF_image_embeds = torch.zeros(num_image, 577, 768)
    ALBEF_text_feats = torch.zeros(num_text, config['embed_dim'])
    ALBEF_text_embeds = torch.zeros(num_text, 30, 768)
    ALBEF_text_atts = torch.zeros(num_text, 30).long()

    TCL_image_feats = torch.zeros(num_image, config['embed_dim'])
    TCL_image_embeds = torch.zeros(num_image, 577, 768)
    TCL_text_feats = torch.zeros(num_text, config['embed_dim'])
    TCL_text_embeds = torch.zeros(num_text, 30, 768)
    TCL_text_atts = torch.zeros(num_text, 30).long()

    BLIP_image_feats = torch.zeros(num_image, config['embed_dim'])
    BLIP_image_embeds = torch.zeros(num_image, 577, 768)
    BLIP_text_feats = torch.zeros(num_text, config['embed_dim'])
    BLIP_text_embeds = torch.zeros(num_text, 30, 768)
    BLIP_text_atts = torch.zeros(num_text, 30).long()
    BLIP_text_ids = []

    XVLM_image_feats = torch.zeros(num_image, config['embed_dim'])
    XVLM_image_embeds = torch.zeros(num_image, 145, 1024)
    XVLM_text_feats = torch.zeros(num_text, config['embed_dim'])
    XVLM_text_embeds = torch.zeros(num_text, 30, 768)
    XVLM_text_atts = torch.zeros(num_text, 30).long()

    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None

    print('Forward')
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(tqdm(data_loader)):
        texts_ids = []
        txt2img = []
        texts = []
        img2txt = []
        txt_id = 0
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i]*len(text_ids_groups[i])
            img2txt.append([])
            for j in range(len(texts_group[i])):
                img2txt[i].append(txt_id)
                txt_id = txt_id + 1

        images = images.to(device)       
        adv_images = images + uap_noise.expand(images.size())
        adv_texts = txt_attacker.get_adv_text(source_model, texts)

        with torch.no_grad():
            t_adv_img_list = []
            for itm in adv_images:
                t_adv_img_list.append(target_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)

            if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
                extra_adv_img_list = []
                for itm in adv_images:
                    extra_adv_img_list.append(extra_transform(itm))
                extra_adv_imgs = torch.stack(extra_adv_img_list, 0).to(device)

            if args.source_model == 'ALBEF':
                ALBEF_adv_images = adv_images
                TCL_adv_images = extra_adv_imgs
                BLIP_adv_images = extra_adv_imgs
                XVLM_adv_images = extra_adv_imgs
                CLIP_VIT_adv_imgs = t_adv_imgs
                CLIP_CNN_adv_imgs = t_adv_imgs
            elif args.source_model == 'TCL':
                ALBEF_adv_images = extra_adv_imgs
                TCL_adv_images = adv_images
                BLIP_adv_images = extra_adv_imgs
                XVLM_adv_images = extra_adv_imgs
                CLIP_VIT_adv_imgs = t_adv_imgs
                CLIP_CNN_adv_imgs = t_adv_imgs
            elif args.source_model == 'BLIP':
                ALBEF_adv_images = extra_adv_imgs
                TCL_adv_images = extra_adv_imgs
                BLIP_adv_images = adv_images
                XVLM_adv_images = extra_adv_imgs
                CLIP_VIT_adv_imgs = t_adv_imgs
                CLIP_CNN_adv_imgs = t_adv_imgs
            elif args.source_model == 'XVLM':
                ALBEF_adv_images = extra_adv_imgs
                TCL_adv_images = extra_adv_imgs
                BLIP_adv_images = extra_adv_imgs
                XVLM_adv_images = adv_images
                CLIP_VIT_adv_imgs = t_adv_imgs
                CLIP_CNN_adv_imgs = t_adv_imgs
            else:
                ALBEF_adv_images = t_adv_imgs
                TCL_adv_images = t_adv_imgs
                BLIP_adv_images = t_adv_imgs
                XVLM_adv_images = t_adv_imgs
                CLIP_VIT_adv_imgs = adv_images
                CLIP_CNN_adv_imgs = adv_images

            ALBEF_adv_images_norm = images_normalize(ALBEF_adv_images)
            ALBEF_adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
            ALBEF_output_img = ALBEF_model.inference_image(ALBEF_adv_images_norm)
            ALBEF_output_txt = ALBEF_model.inference_text(ALBEF_adv_texts_input)
            ALBEF_image_feats[images_ids] = ALBEF_output_img['image_feat'].cpu().detach()
            ALBEF_image_embeds[images_ids] = ALBEF_output_img['image_embed'].cpu().detach()
            ALBEF_text_feats[texts_ids] = ALBEF_output_txt['text_feat'].cpu().detach()
            ALBEF_text_embeds[texts_ids] = ALBEF_output_txt['text_embed'].cpu().detach()
            ALBEF_text_atts[texts_ids] = ALBEF_adv_texts_input.attention_mask.cpu().detach()               
            
            TCL_adv_images_norm = images_normalize(TCL_adv_images)
            TCL_adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
            TCL_output_img = TCL_model.inference_image(TCL_adv_images_norm)
            TCL_output_txt = TCL_model.inference_text(TCL_adv_texts_input)
            TCL_image_feats[images_ids] = TCL_output_img['image_feat'].cpu().detach()
            TCL_image_embeds[images_ids] = TCL_output_img['image_embed'].cpu().detach()
            TCL_text_feats[texts_ids] = TCL_output_txt['text_feat'].cpu().detach()
            TCL_text_embeds[texts_ids] = TCL_output_txt['text_embed'].cpu().detach()
            TCL_text_atts[texts_ids] = TCL_adv_texts_input.attention_mask.cpu().detach()  

            BLIP_adv_images_norm = images_normalize(BLIP_adv_images)
            BLIP_adv_texts_input = blip_tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
            BLIP_output_img = BLIP_model.inference_image(BLIP_adv_images_norm)
            BLIP_output_txt = BLIP_model.inference_text(BLIP_adv_texts_input)
            BLIP_image_feats[images_ids] = BLIP_output_img['image_feat'].cpu().detach()
            BLIP_image_embeds[images_ids] = BLIP_output_img['image_embed'].cpu().detach()
            BLIP_text_feats[texts_ids] = BLIP_output_txt['text_feat'].cpu().detach()
            BLIP_text_embeds[texts_ids] = BLIP_output_txt['text_embed'].cpu().detach()
            BLIP_text_atts[texts_ids] = BLIP_adv_texts_input.attention_mask.cpu().detach()   
            BLIP_text_ids.append(BLIP_adv_texts_input.input_ids)

            XVLM_adv_images_norm = images_normalize(XVLM_adv_images)
            XVLM_adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
            XVLM_output_img = XVLM_model.inference_image(XVLM_adv_images_norm)
            XVLM_output_txt = XVLM_model.inference_text(XVLM_adv_texts_input)
            XVLM_image_feats[images_ids] = XVLM_output_img['image_feat'].cpu().detach()
            XVLM_image_embeds[images_ids] = XVLM_output_img['image_embed'].cpu().detach()
            XVLM_text_feats[texts_ids] = XVLM_output_txt['text_feat'].cpu().detach()
            XVLM_text_embeds[texts_ids] = XVLM_output_txt['text_embed'].cpu().detach()
            XVLM_text_atts[texts_ids] = XVLM_adv_texts_input.attention_mask.cpu().detach()  
            
            CLIP_VIT_adv_images_norm = images_normalize(CLIP_VIT_adv_imgs)
            output = CLIP_VIT_model.inference(CLIP_VIT_adv_images_norm, adv_texts)
            CLIP_VIT_image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            CLIP_VIT_text_feats[texts_ids] = output['text_feat'].cpu().float().detach()

            CLIP_CNN_adv_images_norm = images_normalize(CLIP_CNN_adv_imgs)
            output = CLIP_CNN_model.inference(CLIP_CNN_adv_images_norm, adv_texts)
            CLIP_CNN_image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            CLIP_CNN_text_feats[texts_ids] = output['text_feat'].cpu().float().detach()

    BLIP_text_ids = torch.cat(BLIP_text_ids,dim=0)

    ALBEF_score_matrix_i2t, ALBEF_score_matrix_t2i = retrieval_score('ALBEF', ALBEF_model, ALBEF_image_feats, ALBEF_image_embeds, ALBEF_text_feats,
                                                         ALBEF_text_embeds, ALBEF_text_atts, None, num_image, num_text, device=device)
    TCL_score_matrix_i2t, TCL_score_matrix_t2i = retrieval_score('TCL', TCL_model, TCL_image_feats, TCL_image_embeds, TCL_text_feats,
                                                         TCL_text_embeds, TCL_text_atts, None, num_image, num_text, device=device)
    BLIP_score_matrix_i2t, BLIP_score_matrix_t2i = retrieval_score('BLIP', BLIP_model, BLIP_image_feats, BLIP_image_embeds, BLIP_text_feats,
                                                         BLIP_text_embeds, BLIP_text_atts, BLIP_text_ids, num_image, num_text, device=device)
    XVLM_score_matrix_i2t, XVLM_score_matrix_t2i = retrieval_score('XVLM', XVLM_model, XVLM_image_feats, XVLM_image_embeds, XVLM_text_feats,
                                                         XVLM_text_embeds, XVLM_text_atts, None, num_image, num_text, device=device)
    CLIP_VIT_sims_matrix = CLIP_VIT_image_feats @ CLIP_VIT_text_feats.t()

    CLIP_CNN_sims_matrix = CLIP_CNN_image_feats @ CLIP_CNN_text_feats.t()

    return ALBEF_score_matrix_i2t.cpu().numpy(), ALBEF_score_matrix_t2i.cpu().numpy(), \
            TCL_score_matrix_i2t.cpu().numpy(), TCL_score_matrix_t2i.cpu().numpy(), \
            BLIP_score_matrix_i2t.cpu().numpy(), BLIP_score_matrix_t2i.cpu().numpy(), \
            XVLM_score_matrix_i2t.cpu().numpy(), XVLM_score_matrix_t2i.cpu().numpy(), \
            CLIP_VIT_sims_matrix.cpu().numpy(), CLIP_VIT_sims_matrix.t().cpu().numpy(), \
            CLIP_CNN_sims_matrix.cpu().numpy(), CLIP_CNN_sims_matrix.t().cpu().numpy()
    

@torch.no_grad()
def retrieval_score(model_name, model, image_feats, image_embeds, text_feats, text_embeds, text_atts, text_ids, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        if model_name == 'BLIP':
            output = model.text_encoder(text_ids[topk_idx].to(device),
                                            attention_mask=text_atts[topk_idx].to(device),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True
                                            )
        else:
            output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                        attention_mask=text_atts[topk_idx].to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        if model_name == 'BLIP':
            output = model.text_encoder(text_ids[i].repeat(config['k_test'], 1).to(device),
                                            attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True
                                            )
        else:
            output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                        attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = os.path.join(args.original_rank_index_path, config['dataset'], args.mode)
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) # 在原来的分类成功的样本里，但是现在不在攻击后的成功分类集合里
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)

    eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
                   'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
                   'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
                   'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
                   'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
                   'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    return eval_result



def eval_asr(ALBEF_model, TCL_model, BLIP_model, XVLM_model, CLIP_VIT_model, CLIP_CNN_model, ref_model, tokenizer, blip_tokenizer, extra_transform, target_transform, data_loader, device, config):
    ALBEF_model = ALBEF_model.to(device)
    TCL_model = TCL_model.to(device)
    BLIP_model = BLIP_model.to(device)
    XVLM_model = XVLM_model.to(device)
    CLIP_VIT_model = CLIP_VIT_model.to(device)
    CLIP_CNN_model = CLIP_CNN_model.to(device)
    ref_model = ref_model.to(device)

    print("Start eval")
    start_time = time.time()
    
    ALBEF_score_i2t, ALBEF_score_t2i, TCL_score_i2t, TCL_score_t2i, BLIP_score_i2t, BLIP_score_t2i, XVLM_score_i2t, XVLM_score_t2i, CLIP_VIT_score_i2t, CLIP_VIT_score_t2i,\
    CLIP_CNN_score_i2t, CLIP_CNN_score_t2i = retrieval_eval(ALBEF_model, TCL_model, BLIP_model, XVLM_model, CLIP_VIT_model, CLIP_CNN_model, ref_model, tokenizer, blip_tokenizer, extra_transform, target_transform, data_loader, device, config)

    ALBEF_result = itm_eval(ALBEF_score_i2t, ALBEF_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'ALBEF')
    TCL_result = itm_eval(TCL_score_i2t, TCL_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'TCL')
    BLIP_result = itm_eval(BLIP_score_i2t, BLIP_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'BLIP')
    XVLM_result = itm_eval(XVLM_score_i2t, XVLM_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'XVLM')
    CLIP_VIT_result = itm_eval(CLIP_VIT_score_i2t, CLIP_VIT_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_ViT')
    CLIP_CNN_result = itm_eval(CLIP_CNN_score_i2t, CLIP_CNN_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_CNN')

    print('Performance on ALBEF: \n {}'.format(ALBEF_result))
    print('Performance on TCL: \n {}'.format(TCL_result))
    print('Performance on XVLM: \n {}'.format(XVLM_result))
    print('Performance on CLIP_VIT: \n {}'.format(CLIP_VIT_result))
    print('Performance on CLIP_CNN: \n {}'.format(CLIP_CNN_result))
    print('Performance on BLIP: \n {}'.format(BLIP_result))
    
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
device = torch.device('cuda')

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

ALBEF_ckpt = os.path.join(args.checkpoint, 'ALBEF', '{}.pth'.format(config['dataset']))
TCL_ckpt = os.path.join(args.checkpoint, 'TCL', '{}.pth'.format(config['dataset']))
BLIP_ckpt = os.path.join(args.checkpoint, 'BLIP', '{}.pth'.format(config['dataset']))
XVLM_ckpt = os.path.join(args.checkpoint, 'XVLM', '{}.pth'.format(config['dataset']))

print("Creating Source Model")
ALBEF_model, ref_model, tokenizer = load_model("ALBEF", ALBEF_ckpt, args.text_encoder, config, device)
TCL_model, _, _ = load_model("TCL", TCL_ckpt, args.text_encoder, config, device)
BLIP_model, _, blip_tokenizer = load_model("BLIP", BLIP_ckpt, args.text_encoder, config, device)
XVLM_model, _, _ = load_model("XVLM", XVLM_ckpt, args.text_encoder, config, device)
CLIP_VIT_model, _, _ = load_model("ViT-B/16", None, args.text_encoder, config, device)
CLIP_CNN_model, _, _ = load_model("RN101", None, args.text_encoder, config, device)

#### Dataset ####
print("Creating dataset")

if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
    source_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    extra_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    n_px = CLIP_VIT_model.visual.input_resolution
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(), 
    ])
else:
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    extra_transform = None

    n_px = CLIP_VIT_model.visual.input_resolution
    source_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])

test_dataset = paired_dataset(config['annotation_file'], source_transform, config['image_root'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=test_dataset.collate_fn)

eval_asr(ALBEF_model, TCL_model, BLIP_model, XVLM_model, CLIP_VIT_model, CLIP_CNN_model, ref_model, tokenizer, blip_tokenizer, extra_transform, target_transform, test_loader, device, config)


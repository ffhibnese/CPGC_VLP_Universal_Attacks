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
import pandas as pd

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
parser.add_argument('--output', type=str, default='result')
args = parser.parse_args()

filter_words = get_filter_words()

model_list = ['ALBEF', 'TCL', 'BLIP', 'XVLM', 'ViT-B/16', 'RN101']
record = dict()
for model_name in model_list:
    record[model_name] = dict()

def retrieval_eval(record, ref_model, tokenizer, blip_tokenizer, target_transform, data_loader, device, config):
    # test
    for model_name in model_list:
        record[model_name]['model'].float()
        record[model_name]['model'].eval()
    ref_model.eval()

    source_model = record[args.source_model]['model']

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

    for model_name in model_list:
        if model_name == 'BLIP':
            record[model_name]['text_ids'] = []
        if model_name in ['ALBEF', 'TCL', 'BLIP']:
            record[model_name]['image_feats'] = torch.zeros(num_image, config['embed_dim'])
            record[model_name]['image_embeds'] = torch.zeros(num_image, 577, 768)
            record[model_name]['text_feats'] = torch.zeros(num_text, config['embed_dim'])
            record[model_name]['text_embeds'] = torch.zeros(num_text, 30, 768)
            record[model_name]['text_atts'] = torch.zeros(num_text, 30).long()
        elif model_name == 'XVLM':
            record[model_name]['image_feats'] = torch.zeros(num_image, config['embed_dim'])
            record[model_name]['image_embeds'] = torch.zeros(num_image, 145, 1024)
            record[model_name]['text_feats'] = torch.zeros(num_text, config['embed_dim'])
            record[model_name]['text_embeds'] = torch.zeros(num_text, 30, 768)
            record[model_name]['text_atts'] = torch.zeros(num_text, 30).long()
        else:
            record[model_name]['image_feats'] = torch.zeros(num_image, record[model_name]['model'].visual.output_dim)
            record[model_name]['text_feats'] = torch.zeros(num_text, record[model_name]['model'].visual.output_dim)

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
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
        adv_texts = txt_attacker.get_adv_text(source_model, texts)

        with torch.no_grad():
            t_adv_img_list = []
            for itm in adv_images:
                t_adv_img_list.append(target_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)


            for model_name in model_list:
                if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                    if args.source_model in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                        adv_images = adv_images
                    else:
                        adv_images = t_adv_imgs
                    
                    adv_images_norm = images_normalize(adv_images)
                    if model_name == 'BLIP':
                        adv_texts_input = blip_tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
                        record[model_name]['text_ids'].append(adv_texts_input.input_ids)
                    else:
                        adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
                    output_img = record[model_name]['model'].inference_image(adv_images_norm)
                    output_txt = record[model_name]['model'].inference_text(adv_texts_input)
                    record[model_name]['image_feats'][images_ids] = output_img['image_feat'].cpu().detach()
                    record[model_name]['image_embeds'][images_ids] = output_img['image_embed'].cpu().detach()
                    record[model_name]['text_feats'][texts_ids] = output_txt['text_feat'].cpu().detach()
                    record[model_name]['text_embeds'][texts_ids] = output_txt['text_embed'].cpu().detach()
                    record[model_name]['text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
                else:
                    if args.source_model in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                        adv_images = t_adv_imgs
                    else:
                        adv_images = adv_images
            
                    adv_images_norm = images_normalize(adv_images)
                    output = record[model_name]['model'].inference(adv_images_norm, adv_texts)
                    record[model_name]['image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
                    record[model_name]['text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()


    record['BLIP']['text_ids'] = torch.cat(record['BLIP']['text_ids'], dim=0)

    for model_name in model_list:
        if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
            record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'] = retrieval_score(record, model_name, num_image, num_text, device=device)
        else:
            sims_matrix = record[model_name]['image_feats'] @ record[model_name]['text_feat'].t()
            record[model_name]['score_matrix_i2t'] = sims_matrix.cpu().numpy()
            record[model_name]['score_matrix_t2i'] = sims_matrix.t().cpu().numpy()
    return 
    
@torch.no_grad()
def retrieval_score(record, model_name, num_image, num_text, device=None):
    if device is None:
        device = record[model_name]['image_embeds'].device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    image_feats = record[model_name]['image_feats']
    text_feats = record[model_name]['text_feats']
    image_embeds = record[model_name]['image_embeds']
    text_embeds = record[model_name]['text_embeds']
    text_atts = record[model_name]['text_atts']
    model = record[model_name]['model']

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        if model_name == 'BLIP':
            text_ids = record[model_name]['text_ids']
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
            text_ids = record[model_name]['text_ids']
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

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


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

    # tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = os.path.join(args.original_rank_index_path, config['dataset'])
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) 
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    # ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)

    eval_result = {'txt_r1': asr_tr1,
                   'txt_r5': asr_tr5,
                   'txt_r10': asr_tr10,
                   'img_r1': asr_ir1,
                   'img_r5': asr_ir5,
                   'img_r10': asr_ir10}
    return eval_result



def eval_asr(record, ref_model, tokenizer, blip_tokenizer, target_transform, data_loader, device, config):
    for model_name in model_list:
        record[model_name]['model'] = record[model_name]['model'].to(device)
    ref_model = ref_model.to(device)

    print("Start eval")
    start_time = time.time()
    
    retrieval_eval(record, ref_model, tokenizer, blip_tokenizer, target_transform, data_loader, device, config)
    result = {}
    for model_name in model_list:
        if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, model_name)
        elif model_name == 'ViT-B/16':
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_ViT')
        else:
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_CNN')
        print('Performance on {}: \n {}'.format(model_name, record[model_name]['result']))
        i2t = model_name + ' i2t'
        t2i = model_name + ' t2i'
        result[i2t] = []
        result[t2i] = []
        result[i2t].append(record[model_name]['result']['txt_r1'])
        result[i2t].append(record[model_name]['result']['txt_r5'])
        result[i2t].append(record[model_name]['result']['txt_r10'])
        result[t2i].append(record[model_name]['result']['img_r1'])
        result[t2i].append(record[model_name]['result']['img_r5'])
        result[t2i].append(record[model_name]['result']['img_r10'])

    df = pd.DataFrame(result)

    save_dir = f'./outputs/{args.output}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/{args.source_model}_{config['dataset']}.xlsx"

    df.to_excel(save_path, index=False)

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

print("Creating Source Model")
for model_name in model_list:
    record[model_name]['ckpt'] = os.path.join(args.checkpoint, model_name, '{}.pth'.format(config['dataset']))
    if model_name == 'BLIP':
        record[model_name]['model'], _, blip_tokenizer = load_model(model_name, record[model_name]['ckpt'], args.text_encoder, config, device)
    else:
        record[model_name]['model'], ref_model, tokenizer = load_model(model_name, record[model_name]['ckpt'], args.text_encoder, config, device)

#### Dataset ####
print("Creating dataset")

if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
    source_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    n_px = record['ViT-B/16']['model'].visual.input_resolution
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

    n_px = record['ViT-B/16']['model'].visual.input_resolution
    source_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])

test_dataset = paired_dataset(config['annotation_file'], source_transform, config['image_root'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=test_dataset.collate_fn)

eval_asr(record, ref_model, tokenizer, blip_tokenizer, target_transform, test_loader, device, config)


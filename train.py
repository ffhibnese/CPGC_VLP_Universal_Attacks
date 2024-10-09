import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from torch.autograd import Variable
import ruamel.yaml as yaml
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from generator import Generator
import utils
from utils import load_model
from attacker import CPGCAttacker, ImageAttacker, TextAttacker
from dataset import paired_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/Retrieval_flickr_train.yaml')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--target_batch_size', default=4, type=int)
parser.add_argument('--source_model', default='ALBEF', type=str)
parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
parser.add_argument('--source_ckpt', default='./checkpoint/', type=str)
parser.add_argument('--eps', type=int, default=12)
parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
parser.add_argument('--word_num', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='output')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--load_dir', type=str)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()

device = torch.device('cuda')
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

if args.source_model == 'ViT-B/16':
    save_dir = os.path.join(args.save_dir, 'CLIP_VIT', config['dataset'])
elif args.source_model == 'RN101':
    save_dir = os.path.join(args.save_dir, 'CLIP_CNN', config['dataset'])
else:
    save_dir = os.path.join(args.save_dir, args.source_model, config['dataset'])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

def train(model, ref_model, tokenizer, data_loader, target_loader, device, args):
    target_dataiter = iter(target_loader)

    print("Start train")
    model.float()
    model.eval()
    ref_model.eval()
    image_G_input_dim = 100
    image_G_output_dim = 3
    image_num_filters = [[1024, 512], [256, 128], [64, 32]]
    if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
        context_dim = 256
        dim = 3
        word_embedding = torch.load("word_embedding_{}.pth".format(args.source_model))
    else:
        context_dim = 512
        dim = 2
        word_embedding = torch.load("word_embedding_CLIP.pth")
    norms = torch.norm(word_embedding, p=2, dim=1)
    max_norm, _ = torch.max(norms, dim=0)
    min_norm, _ = torch.min(norms, dim=0)
    image_netG = Generator(image_G_input_dim, image_num_filters, image_G_output_dim, args.batch_size * 5,
                           first_kernel_size=4, num_heads=1, context_dim=context_dim)
    image_z = torch.randn(args.batch_size * 5, image_G_input_dim, 3, 3)

    text_G_input_dim = 32
    text_G_output_dim = args.word_num
    text_num_filters = [[256, 128], [64, 32]]
    text_netG = Generator(text_G_input_dim, text_num_filters, text_G_output_dim, args.batch_size * 5,
                          first_kernel_size=1, num_heads=1, context_dim=context_dim)
    text_z = torch.randn(args.batch_size * 5, text_G_input_dim, 1, dim)
    if args.start_epoch > 0:
        if args.source_model == 'ViT-B/16':
            load_dir = os.path.join(args.load_dir, 'CLIP_VIT', config['dataset'])
        elif args.source_model == 'RN101':
            load_dir = os.path.join(args.load_dir, 'CLIP_CNN', config['dataset'])
        else:
            load_dir = os.path.join(args.load_dir, args.source_model, config['dataset'])
        image_z = torch.load(os.path.join(load_dir, 'image-z-{}.pth'.format(args.start_epoch - 1)), map_location=device)
        text_z = torch.load(os.path.join(load_dir, 'text-z-{}.pth'.format(args.start_epoch - 1)), map_location=device)
        image_netG.load_state_dict(
            torch.load(os.path.join(load_dir, 'image-model-{}.pth'.format(args.start_epoch - 1)), map_location=device))
        text_netG.load_state_dict(
            torch.load(os.path.join(load_dir, 'text-model-{}.pth'.format(args.start_epoch - 1)), map_location=device))
    image_netG = image_netG.to(device)
    text_netG = text_netG.to(device)
    text_z = Variable(text_z.to(device))
    image_z = Variable(image_z.to(device))

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_attacker = ImageAttacker(image_netG, images_normalize, args.temperature, z=image_z, model=args.source_model,
                                 eps=args.eps / 255, device=device, lr=args.lr, alpha=args.alpha)
    txt_attacker = TextAttacker(ref_model, tokenizer, text_netG, text_z, args.source_model, device, temperature=args.temperature,
                                alpha=args.alpha, min_norm=min_norm, max_norm=max_norm, lr=args.lr, number_perturbation=args.word_num)
    attacker = CPGCAttacker(model, img_attacker, txt_attacker)
    if args.scales == 'None':
        scales = None
    else:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)

    for epoch in range(args.start_epoch, args.epochs):
        image_running_loss = 0
        image_running_loss_MSE = 0
        image_running_loss_infoNCE = 0
        text_running_loss = 0
        text_running_loss_MSE = 0
        text_running_loss_infoNCE = 0
        for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(tqdm(data_loader)):
            txt2img = []
            texts = []
            txt_id = 0
            img2txt = []
            for i in range(len(texts_group)):
                texts += texts_group[i]
                txt2img += [i] * len(text_ids_groups[i])
                img2txt.append([])
                for j in range(len(texts_group[i])):
                    img2txt[i].append(txt_id)
                    txt_id = txt_id + 1
            images = images.to(device)

            try:
                target_imgs, target_texts_group, _, _ = next(target_dataiter)
            except StopIteration:
                target_dataiter = iter(target_loader)
                target_imgs, target_texts_group, _, _ = next(target_dataiter)
            target_texts = []
            target_img2txt = []
            txt_id = 0
            for i in range(len(target_texts_group)):
                target_texts += target_texts_group[i]
                target_img2txt.append([])
                for j in range(len(target_texts_group[i])):
                    target_img2txt[i].append(txt_id)
                    txt_id = txt_id + 1

            target_imgs = target_imgs.to(device)
            target_imgs_outputs = model.inference_image(images_normalize(target_imgs))
            target_img_supervisions = target_imgs_outputs['image_feat']

            target_txts_input = tokenizer(target_texts, padding='max_length', truncation=True, max_length=30,
                                          return_tensors="pt").to(device)
            target_txts_output = model.inference_text(target_txts_input)
            target_txt_supervisions = target_txts_output['text_feat']

            image_loss, image_loss_infoNCE, image_loss_MSE, uap_noise, text_loss, text_loss_infoNCE, text_loss_MSE, uap_embedding = attacker.attack(
                images, texts, img2txt, txt2img, target_img_supervisions, target_txt_supervisions, target_img2txt,
                device=device, max_lemgth=30, scales=scales)
            if batch_idx % 10 == 9:
                print(
                    'Epoch: {} \t Batch: {}/{} \t image infoNCE loss: {:.5f} \t image MSE loss: {:.5f} \t image total loss: {:.5f} \t text infoNCE loss: {:.5f} \t text MSE loss: {:.5f} \t text total loss: {:.5f}'.format(
                        epoch, batch_idx, len(data_loader), image_running_loss_infoNCE / 10,
                        image_running_loss_MSE / 10, image_running_loss / 10, text_running_loss_infoNCE / 10,
                        text_running_loss_MSE / 10, text_running_loss / 10))
                image_running_loss = 0
                image_running_loss_MSE = 0
                image_running_loss_infoNCE = 0
                text_running_loss = 0
                text_running_loss_MSE = 0
                text_running_loss_infoNCE = 0
            image_running_loss += image_loss.item()
            image_running_loss_MSE += image_loss_MSE.item()
            image_running_loss_infoNCE += image_loss_infoNCE.item()
            text_running_loss += text_loss.item()
            text_running_loss_MSE += text_loss_MSE.item()
            text_running_loss_infoNCE += text_loss_infoNCE.item()
        attacker.img_attacker.save_model('{}/image-model-{}.pth'.format(save_dir, epoch))
        torch.save(image_z, '{}/image-z-{}.pth'.format(save_dir, epoch))
        torch.save(uap_noise, '{}/uap_noise-{}.pth'.format(save_dir, epoch))
        attacker.txt_attacker.save_model('{}/text-model-{}.pth'.format(save_dir, epoch))
        torch.save(text_z, '{}/text-z-{}.pth'.format(save_dir, epoch))
        torch.save(uap_embedding, '{}/uap_embedding-{}.pth'.format(save_dir, epoch))
    torch.cuda.empty_cache()


print("Creating Source Model")
source_ckpt = os.path.join(args.source_ckpt, args.source_model, '{}.pth'.format(config['dataset']))
model, ref_model, tokenizer = load_model(args.source_model, source_ckpt, args.source_text_encoder, config, device)
model = model.to(device)
ref_model = ref_model.to(device)

print("Creating dataset")
if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
    s_test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
else:
    n_px = model.visual.input_resolution
    s_test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])

train_dataset = paired_dataset(config['annotation_file'], s_test_transform, config['image_root'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          num_workers=4, collate_fn=train_dataset.collate_fn, drop_last=True)
target_dataset = paired_dataset(config['annotation_file'], s_test_transform, config['image_root'])
target_loader = DataLoader(target_dataset, batch_size=args.target_batch_size,
                           num_workers=4, collate_fn=target_dataset.collate_fn, shuffle=True, drop_last=True)

train(model, ref_model, tokenizer, train_loader, target_loader, device, args)

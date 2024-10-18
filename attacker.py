import numpy as np
import torch
import copy
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils import get_filter_words

class CPGCAttacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, img2txt, txt2img, target_img_supervision, target_txt_supervision, target_img2txt,
               device='cpu', max_length=30, scales=None, **kwargs):
        with torch.no_grad():
            imgs_outputs = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = imgs_outputs['image_feat']
        txt_loss, txt_loss_infoNCE, txt_loss_MSE, uap_embedding = self.txt_attacker.img_guided_attack(self.model, txts,
                                                                                                      txt2img,
                                                                                                      img_embeds=img_supervisions,
                                                                                                      target_img_embeds=target_img_supervision)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True, max_length=max_length,
                                                     return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

        img_loss, img_loss_infoNCE, img_loss_MSE, uap_noise = self.img_attacker.txt_guided_attack(self.model, imgs,
                                                                                                  img2txt, txt2img,
                                                                                                  device,
                                                                                                  scales=scales,
                                                                                                  txt_embeds=txt_supervisions,
                                                                                                  target_txt_embeds=target_txt_supervision,
                                                                                                  target_img2txt=target_img2txt)
        return img_loss, img_loss_infoNCE, img_loss_MSE, uap_noise, txt_loss, txt_loss_infoNCE, txt_loss_MSE, uap_embedding


class ImageAttacker():
    def __init__(self, netG, normalization, temperature, z, model, eps, device='cuda', lr=2e-4, alpha=0.1):
        self.normalization = normalization
        self.eps = eps
        self.generator = netG
        if self.generator is not None:
            self.generator = self.generator.to(device)
            self.optimG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.z = z
        self.temperature = temperature
        self.alpha = alpha
        self.model = model

    def get_generator(self):
        return self.generator

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)

    def loss_func(self, adv_imgs_embeds, imgs_embeds, txts_embeds, txt2img, target_txt_embeds, target_img2txt,
                  temperature):
        device = adv_imgs_embeds.device
        target_it_sim_matrix = torch.exp((adv_imgs_embeds @ target_txt_embeds.T) / temperature)
        similarity = imgs_embeds @ target_txt_embeds.T
        average_similarity = torch.zeros(imgs_embeds.size(0), len(target_img2txt))
        for i in range(average_similarity.size(1)):
            average_similarity[:, i] = torch.mean(similarity[:, target_img2txt[i]], dim=-1)
        index = torch.min(average_similarity, dim=-1).indices
        target_it_labels = torch.zeros_like(target_it_sim_matrix).to(device)
        for i in range(target_it_labels.size(0)):
            target_it_labels[i][target_img2txt[index[i]]] = 1

        loss_target = (target_it_sim_matrix * target_it_labels).sum(-1).mean()

        it_sim_matrix = torch.exp((adv_imgs_embeds @ txts_embeds.T) / temperature)
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1

        loss_untarget = (it_sim_matrix * it_labels).sum(-1).mean()
        loss = torch.log(loss_untarget / (loss_untarget + loss_target))

        return loss

    def txt_guided_attack(self, model, imgs, img2txt, txt2img, device, scales=None, txt_embeds=None,
                          target_txt_embeds=None, target_img2txt=None):
        model.eval()
        b, _, _, _ = imgs.shape
        aug_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        aug_imgs = torch.clamp(aug_imgs, 0.0, 1.0)
        scaled_imgs = self.get_scaled_imgs(aug_imgs, scales, device)

        self.generator.train()
        self.optimG.zero_grad()
        for p in model.parameters():
            p.requires_grad = True

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1
        text_cond = []
        for i in range(len(img2txt)):
            text_cond.append(torch.mean(txt_embeds[img2txt[i]], dim=0))
        text_cond = text_cond * 5
        text_cond = torch.stack(text_cond, dim=0)

        with torch.enable_grad():
            x = Variable(scaled_imgs.to(device))
            uap_noise = self.generator(self.z, text_cond)
            if self.model in ['ViT-B/16', 'RN101']:
                uap_noise = F.interpolate(uap_noise, size=(224, 224), mode='bilinear')
            uap_noise = uap_noise.squeeze()
            uap_noise = torch.clamp(uap_noise, -self.eps, self.eps)

            # fake image
            adv_imgs = x + uap_noise.expand(scaled_imgs.size())

            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                imgs_output = model.inference_image(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.inference_image(adv_imgs)
                imgs_output = model.inference_image(scaled_imgs)

            adv_imgs_embeds = adv_imgs_output['image_feat']
            imgs_embeds = imgs_output['image_feat']

            criterion_MSE = torch.nn.MSELoss(reduce=True, size_average=False)
            loss_MSE = criterion_MSE(adv_imgs_embeds, imgs_embeds)

            loss_infoNCE = torch.tensor(0.0, dtype=torch.float32).to(device)
            for i in range(scales_num):
                loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], imgs_embeds[i * b:i * b + b], txt_embeds,
                                           txt2img, target_txt_embeds, target_img2txt, self.temperature)
                loss_infoNCE += loss_item
            loss = loss_infoNCE - self.alpha * loss_MSE
        loss.backward()
        self.optimG.step()

        return loss, loss_infoNCE, loss_MSE, uap_noise

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)

filter_words = get_filter_words()


class TextAttacker():
    def __init__(self, ref_net, tokenizer, netG=None, z=None, model_name=None, device='cuda',temperature=0.1,
                alpha=None, min_norm=None, max_norm=None, adv_words=None, lr=2e-4, cls=False,max_length=30, 
                number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.z = z
        self.model_name = model_name
        self.generator = netG
        if self.generator is not None:
            self.generator = self.generator.to(device)
            self.optimG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.temperature = temperature
        self.alpha = alpha
        self.adv_words = adv_words
        self.min_norm = min_norm
        self.max_norm = max_norm

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)

    def img_guided_attack(self, net, texts, txt2img, img_embeds=None, target_img_embeds=None):
        device = self.ref_net.device
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        self.generator.train()
        self.optimG.zero_grad()
        for p in net.parameters():
            p.requires_grad = True

        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()
        img_cond = img_embeds[txt2img]
        padding_zero = []
        position = []
        for i, text in enumerate(texts):
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            if self.model_name in ['ViT-B/16', 'RN101']:
                position.append([])
                for j in range(self.num_perturbation):
                    position[i].append(list_of_index[j][0] + 1)

            else:
                words, sub_words, keys = self._tokenize(text)

                position.append([])
                padding_zero.append([])
                for j in range(self.num_perturbation):
                    position[i].append(keys[list_of_index[j][0]][0])
                    padding_zero[i].append(keys[list_of_index[j][0]][1] - keys[list_of_index[j][0]][0] - 1)
        with torch.enable_grad():
            uap_embedding = self.generator(self.z, img_cond)[0]
            uap_embedding = uap_embedding.reshape(uap_embedding.size(0), -1)
            norm = torch.norm(uap_embedding, dim=1).view(-1, 1).expand_as(uap_embedding)
            clamp_norm = torch.clamp(norm, self.min_norm, self.max_norm)
            uap_embedding = clamp_norm * uap_embedding / norm

            adv_txt_output = net.inference_text_replace(text_inputs, uap_embedding, position, padding_zero)
            if self.cls:
                adv_txt_embeds = adv_txt_output['text_feat'][:, 0, :]
            else:
                adv_txt_embeds = adv_txt_output['text_feat'].flatten(1)
            loss_infoNCE = self.loss_func(adv_txt_embeds, img_embeds, txt2img, target_img_embeds, self.temperature)
            criterion_MSE = torch.nn.MSELoss(reduce=True, size_average=False)
            loss_MSE = criterion_MSE(adv_txt_embeds, origin_embeds)
            loss = loss_infoNCE - self.alpha * loss_MSE
        loss.backward()
        self.optimG.step()

        return loss, loss_infoNCE, loss_MSE, uap_embedding

    def loss_func(self, adv_txt_embeds, img_embeds, txt2img, target_img_embeds, temperature):
        device = adv_txt_embeds.device
        target_it_sim_matrix = torch.exp((adv_txt_embeds @ target_img_embeds.T) / temperature)
        similarity = adv_txt_embeds @ target_img_embeds.T
        index = torch.min(similarity, dim=-1).indices
        target_it_labels = torch.zeros_like(target_it_sim_matrix).to(device)
        for i in range(target_it_labels.size(0)):
            target_it_labels[i][index[i]] = 1

        loss_target = (target_it_sim_matrix * target_it_labels).sum(-1).mean()

        it_sim_matrix = torch.exp((adv_txt_embeds @ img_embeds.T) / temperature)
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[i, txt2img[i]] = 1

        loss_untarget = (it_sim_matrix * it_labels).sum(-1).mean()
        loss = torch.log(loss_untarget / (loss_untarget + loss_target))

        return loss

    def get_adv_text(self, net, texts):
        device = self.ref_net.device
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= len(self.adv_words):
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                final_words[top_index[0]] = self.adv_words[change]
                change = change + 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)

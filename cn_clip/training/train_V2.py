import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
import torch.nn.functional as F

from einops import rearrange
from cn_clip.clip.model import convert_state_dict


def is_master(args):
    return args.rank == 0

def get_loss(model, datas, masks, loss_img, loss_txt, args, accum_image_features=None, accum_text_features=None, accum_idx=-1, teacher_model=None, teacher_accum_image_features=None,
             labels=None):
    if args.accum_freq == 1:
        data_mask = {**datas, **masks}
        # TODO: 重点处理
        output, logit_scale = model(**data_mask)

        # image_features, text_features = output['model_us'], output['model_text']
        if 'model_us' in output.keys():
            output['model_us'] = rearrange(output['model_us'], '(b u) c -> b u c', b=args.batch_size).mean(dim=1)

        # Todo: 整合强弱模态
        # image_features, text_features = output['model_wsi'], output['model_text']
        image_features, text_features = [out_i[1] for out_i in output.items()][:2]

        # if args.origin_task:
        #     image_features, text_features, logit_scale = model(images, texts, args.mask_ratio)
        # else:   # Todo: 下面只是为了跑通代码; 这里应该直接输入以病人 ID 为 key 的字典，这样才能处理模态缺失的问题
        #     output = model(model_us=torch.randn(2, 3, 256, 256), model_mri=torch.randn(2, 1, 128, 128, 128), model_text=texts)
        #     image_features, text_features = output['model_us'], output['model_mri'] # 那么这里也要改，最终返回的应该是 强模态的特征、缺失模态下的特征

        #     # 还是说我直接把所有需要的 feature 都进行类似的处理？
        #     # 有多少模态就处理多少种模态
        #     logit_scale = torch.tensor([1.])

        if args.distillation:
            with torch.no_grad():
                # different teacher model has different output
                output = teacher_model.module.get_feature(images)
                if(isinstance(output, tuple)):
                    teacher_image_features = output[0]
                else:
                    teacher_image_features = output
    else:
        assert accum_image_features and accum_text_features and accum_idx != -1
        chunk_image_features, chunk_text_features, logit_scale = model(images, texts, args.mask_ratio)

        if args.distillation:
            with torch.no_grad():
                # different teacher model has different output
                output = teacher_model.module.get_feature(images)
                if(isinstance(output, tuple)):
                    teacher_chunk_image_features = output[0]
                else:
                    teacher_chunk_image_features = output
            teacher_image_features = torch.cat(
            teacher_accum_image_features[:accum_idx] + [teacher_chunk_image_features] + teacher_accum_image_features[accum_idx + 1:])
        
        image_features = torch.cat(
            accum_image_features[:accum_idx] + [chunk_image_features] + accum_image_features[accum_idx + 1:])
        text_features = torch.cat(
            accum_text_features[:accum_idx] + [chunk_text_features] + accum_text_features[accum_idx + 1:])
    logit_scale = logit_scale.mean()
    if args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        if args.gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_tumor_label = torch.cat(torch.distributed.nn.all_gather(labels['tumor']), dim=0)

            if args.distillation:
                all_teacher_image_features = torch.cat(torch.distributed.nn.all_gather(teacher_image_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            gathered_tumor_label = [
                torch.zeros_like(labels['tumor']) for _ in range(world_size)
            ]
            
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_tumor_label, labels['tumor'])

            all_image_features = torch.cat(
                [image_features]
                + gathered_image_features[:rank]
                + gathered_image_features[rank + 1 :]
            )
            all_text_features = torch.cat(
                [text_features]
                + gathered_text_features[:rank]
                + gathered_text_features[rank + 1 :]
            )
            all_tumor_label = torch.cat(
                [labels['tumor']]
                + gathered_tumor_label[:rank]
                + gathered_tumor_label[rank + 1 :]
            )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        loss_fn = torch.nn.functional.cross_entropy
        loss_c = loss_fn(all_image_features, all_tumor_label) + loss_fn(all_text_features, all_tumor_label)

        if args.distillation:
            gathered_teacher_image_features = [
                torch.zeros_like(teacher_image_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_teacher_image_features, teacher_image_features)
            all_teacher_image_features = torch.cat(
                [teacher_image_features]
                + gathered_teacher_image_features[:rank]
                + gathered_teacher_image_features[rank + 1 :]
            )
            kd_loss = cosineSimilarityLoss(all_teacher_image_features, all_image_features)

    else:   # note: 这里未更改
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        if args.distillation:
            kd_loss = cosineSimilarityLoss(teacher_image_features, image_features)

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2   #  + loss_c

    acc = None
    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    if args.distillation:
        total_loss += kd_loss * args.kd_loss_weight

    return total_loss, acc

def freeze_vision_bn(args, model):
    # freeze bn running mean and variance
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model, nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps, teacher_model=None):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)    # fixbug: 未被 pickle 序列化的对象

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        if args.distillation:
            teacher_accum_image_features = []

    end = time.time()
    epoch_trained_steps = 0
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        step = num_steps_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step, args.max_steps // args.accum_freq))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        # images, texts, eos_indices = batch

        # 以下都是标签、对模态是否缺失的说明
        tumor_dict = {'astrocytoma': 0, 'glioblastoma': 1, 'oligodendroglioma': 2}
        idh_dict   = {'IDH mutant': 0, 'IDH wild type': 1}
        p19q_dict  = {'无共缺失': 0, '共缺失': 1}

        batch_tumor_label = batch['label'][0]        
        batch_idh_label   = batch['label'][1]
        batch_1p19q_label = batch['label'][2]

        batch_tumor_label = torch.tensor([tumor_dict[i_cls] for i_cls in batch_tumor_label], dtype=torch.long).cuda(args.local_device_rank)
        batch_idh_label = torch.tensor([idh_dict[i_cls] for i_cls in batch_idh_label], dtype=torch.long).cuda(args.local_device_rank)
        batch_1p19q_label = torch.tensor([p19q_dict[i_cls] for i_cls in batch_1p19q_label], dtype=torch.long).cuda(args.local_device_rank)
        labels = {'tumor': batch_tumor_label, 'idh': batch_idh_label, '1p19q': batch_1p19q_label}

        batch_wsi         = [isinstance(moi, str) for moi in batch['case_id'][0]]
        batch_us          = [[isinstance(moi, str)]*args.us_N for moi in batch['case_id'][1]]
        batch_t1c         = batch['mri_modality'][0]
        batch_flair       = batch['mri_modality'][1]

        modality_masks = {
            'mask_wsi': torch.tensor(batch_wsi, dtype=torch.float32).cuda(args.local_device_rank),
            'mask_us': torch.tensor(batch_us, dtype=torch.float32).flatten().cuda(args.local_device_rank),
            'mask_mri_t1c': torch.tensor(batch_t1c, dtype=torch.float32).cuda(args.local_device_rank),
            'mask_mri_flair': torch.tensor(batch_flair, dtype=torch.float32).cuda(args.local_device_rank),
            'mask_text': torch.ones(len(batch_wsi), dtype=torch.float32).cuda(args.local_device_rank)  # 假设文本 跟随病理图像 始终存在
        }

        # TODO: 需要确定维度, 需要调整网络结构
        data_t1c            = batch['data_mri']['T1c'] / 255
        data_flair          = batch['data_mri']['Flair'] / 255
        data_wsi            = batch['data_wsi']             
        data_us             = batch['data_us'].float() / 255
        texts, eos_indices  = batch['text_tumor']           
        # data_us = data_us.view(-1, *data_us.shape[-2:])[:, None].repeat(1, 3, 1, 1) # 超声有多帧图像, 需要专门进行处理
        data_us = rearrange(data_us, 'b (c o) h w -> (b c) o h w', o=1).repeat(1, 3, 1, 1)
        data_t1c = data_t1c[:, None].cuda(args.local_device_rank, non_blocking=True)
        data_flair = data_flair[:, None].cuda(args.local_device_rank, non_blocking=True)
        data_wsi = [i_feat.cuda(args.local_device_rank, non_blocking=True) for i_feat in data_wsi]
        data_us = data_us.cuda(args.local_device_rank, non_blocking=True)
        # images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        # 定义强弱模态（这里只是示例，实际应根据具体情况定义）
        strong_modalities = ['model_wsi', 'model_text']
        weak_modalities = ['model_us', 'model_mri_t1c', 'model_mri_flair']
        all_modality = strong_modalities + weak_modalities
        np.random.seed(0)
        sel_modality = ['model_text', 'model_mri_flair']    # np.random.choice(all_modality, 2, p=[0.2, 0.2, 0.2, 0.2, 0.2])

        # 超声数据还没有经过处理, 因此在这里简单进行标准化, 以运行程序
        data_modalities = {'model_wsi': data_wsi, 'model_text': texts, 'model_us': data_us, 
                           'model_mri_t1c': data_t1c.float(), 'model_mri_flair': data_flair.float(),
                           }

        # TODO: images 需要确定怎么处理
        images = data_modalities['model_us']

        # 模拟缺失数据
        available_modalities = {f'{name}': data_modalities[f'{name}'] 
                                for name in sel_modality }
        available_masks = {f'{name.replace("model_", "mask_")}': modality_masks[f'{name.replace("model_", "mask_")}'] 
                            for name in sel_modality }
        data_time = time.time() - end

        m = model.module

        if args.accum_freq == 1:
            # with automatic mixed precision.
            if args.precision == "amp":
                with autocast():
                    if args.distillation:
                        total_loss, acc = get_loss(model, available_modalities, available_masks, loss_img, loss_txt, args, teacher_model=teacher_model,
                                                   labels=labels)
                    else:
                        total_loss, acc = get_loss(model, available_modalities, available_masks, loss_img, loss_txt, args, labels=labels)
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()

            else:
                if args.distillation:
                    total_loss, acc = get_loss(model, available_modalities, available_masks, loss_img, loss_txt, args, teacher_model=teacher_model)
                else:
                    total_loss, acc = get_loss(model, available_modalities, available_masks, loss_img, loss_txt, args)
                total_loss.backward()
                optimizer.step()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast(enabled=(args.precision == "amp")):
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                if args.distillation:
                    output = teacher_model.module.get_feature(images)
                    if(len(output) == 2):
                        teacher_chunk_image_features = output[0]
                    else:
                        teacher_chunk_image_features = output
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)
                if args.distillation:
                    teacher_accum_image_features.append(teacher_chunk_image_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast(enabled=(args.precision == "amp")):
                    # `total_loss` and `acc` are coarsely sampled, taking only the last result in the loop.
                    # Although each result should be the same in theory, it will be slightly different in practice
                    if args.distillation:
                        total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args, accum_image_features, accum_text_features, j, teacher_model, teacher_accum_image_features)
                    else:
                        total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args, accum_image_features, accum_text_features, j)
                if args.precision == "amp":
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
            if args.distillation:
                teacher_accum_image_features = []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

        if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))
        
    return epoch_trained_steps


def evaluate(model, data, epoch, args, steps):

    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    model.eval()

    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            # images, texts, eos_indices = batch

            # images = images.cuda(args.local_device_rank, non_blocking=True)
            # texts = texts.cuda(args.local_device_rank, non_blocking=True)
            # eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

            # image_features, text_features, logit_scale = model(images, texts)

            # images, texts, eos_indices = batch
            # 以下都是标签、对模态是否缺失的说明
            batch_tumor_label = batch['label'][0]
            batch_idh_label   = batch['label'][1]
            batch_1p19q_label = batch['label'][2]

            batch_wsi         = [isinstance(moi, str) for moi in batch['case_id'][0]]
            batch_us          = [[isinstance(moi, str)]*args.us_N for moi in batch['case_id'][1]]
            batch_t1c         = batch['mri_modality'][0]
            batch_flair       = batch['mri_modality'][1]

            modality_masks = {
                'mask_wsi': torch.tensor(batch_wsi, dtype=torch.float32).cuda(args.local_device_rank),
                'mask_us': torch.tensor(batch_us, dtype=torch.float32).flatten().cuda(args.local_device_rank),
                'mask_mri_t1c': torch.tensor(batch_t1c, dtype=torch.float32).cuda(args.local_device_rank),
                'mask_mri_flair': torch.tensor(batch_flair, dtype=torch.float32).cuda(args.local_device_rank),
                'mask_text': torch.ones(len(batch_wsi), dtype=torch.float32).cuda(args.local_device_rank)  # 假设文本始终存在
            }

            # 数据  Todo: 需要确定维度, 需要调整网络结构
            data_t1c            = batch['data_mri']['T1c'] / 255
            data_flair          = batch['data_mri']['Flair'] / 255
            data_wsi            = batch['data_wsi']
            data_us             = batch['data_us'].float() / 255
            texts, eos_indices  = batch['text_tumor']           
            # data_us = data_us.view(-1, *data_us.shape[-2:])[:, None].repeat(1, 3, 1, 1) # 超声有多帧图像, 需要专门进行处理
            data_us = rearrange(data_us, 'b (c o) h w -> (b c) o h w', o=1).repeat(1, 3, 1, 1)
            data_t1c = data_t1c.cuda(args.local_device_rank, non_blocking=True)
            data_flair = data_flair.cuda(args.local_device_rank, non_blocking=True)
            data_wsi = [i_feat.cuda(args.local_device_rank, non_blocking=True) for i_feat in data_wsi]
            data_us = data_us.cuda(args.local_device_rank, non_blocking=True)
            # images = images.cuda(args.local_device_rank, non_blocking=True)
            texts = texts.cuda(args.local_device_rank, non_blocking=True)
            eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

            # 定义强弱模态（这里只是示例，实际应根据具体情况定义）
            strong_modalities = ['model_wsi', 'model_text']
            weak_modalities = ['model_us', 'model_mri_t1c', 'model_mri_flair']
            all_modality = strong_modalities + weak_modalities
            np.random.seed(0)
            sel_modality = strong_modalities # np.random.choice(all_modality, 2, p=[0.2, 0.2, 0.2, 0.2, 0.2])

            # 超声数据还没有经过处理, 因此在这里简单进行标准化, 以运行程序
            data_modalities = {'model_wsi': data_wsi, 'model_text': texts, 'model_us': data_us.float(), 
                               'model_mri_t1c': data_t1c.float(), 'model_mri_flair': data_flair.float()
                               }

            images = data_modalities['model_us']

            # 模拟缺失数据
            available_modalities = {f'{name}': data_modalities[f'{name}'] 
                                    for name in sel_modality }
            available_masks = {f'{name.replace("model_", "mask_")}': modality_masks[f'{name.replace("model_", "mask_")}'] 
                               for name in sel_modality }
            
            data_mask = {**available_modalities, **available_masks}
            output, logit_scale = model(**data_mask)

            # image_features, text_features = output['model_us'], output['model_text']
            if 'model_us' in output.keys():
                output['model_us'] = rearrange(output['model_us'], '(b u) c -> b u c', b=args.batch_size).mean(dim=1)
            
            image_features, text_features = [out_i[1] for out_i in output.items()][:2]  # 随机选取两个模态的 特征
            # <---

            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(image_features)).long()
            ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(image_features)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

            cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
            cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

            if (i + 1) % 100 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))

        dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_i2t_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_t2i_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
        loss = cumulative_loss / num_elements
        i2t_acc = cumulative_i2t_acc / num_elements
        t2i_acc = cumulative_t2i_acc / num_elements

        assert num_elements.item() == dataloader.num_samples # sanity check

        logging.info(
            f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
            f"Valid Loss: {loss.item():.6f} | "
            f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
            f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
            f"logit_scale: {model.module.logit_scale.data:.3f} | "
            f"Valid Batch Size: {args.valid_batch_size}"
        )

def cosineSimilarityLoss(feature1, feature2):
    scale_factor_h = feature1.shape[0] / feature2.size(0)
    scale_factor_w = feature1.shape[1] / feature2.size(1)

    feature2_interpolated = F.interpolate(feature2.unsqueeze(0).unsqueeze(0),
                            size=(feature1.shape[0], feature1.shape[1]),
                            mode='bilinear',
                            align_corners=False)
    feature2_interpolated = feature2_interpolated.squeeze(0).squeeze(0)
    

    cosine_sim = F.cosine_similarity(feature1, feature2_interpolated, dim=1)
    similarity_loss = 1 - cosine_sim.mean()
    return similarity_loss
import os, sys
sys.path.append('/database/wuyonghuang/WSA')
sys.path.append('/database/wuyonghuang/WSA/relys')  # 记得换成你自己的路径！
sys.path.append('/database/wuyonghuang/WSA/cn_clip')
os.chdir('/database/wuyonghuang/WSA')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math # For isnan
import argparse
import torch
import pickle
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from cn_clip.training.data import get_data
from cn_clip.training.data import custom_collate_fn
from relys.SAM_Med3D_main import build_sam3D_vit_b_ori

from cn_clip.training.nano_train_utils import count_parameters, UNet3DEncoder, UNet2DEncoder


# Assume these U-Net encoders are pre-defined or imported
# They should output a feature vector (e.g., after pooling and flattening)
# For simplicity, let's assume they output a fixed-size feature vector, e.g., 512-dim.


class TumorClassifier(nn.Module):
    def __init__(self, mri_feature_dim=512, us_feature_dim=512, num_classes=3, reduction='attention', build_2d_sam=False):
        super().__init__()
        # self.mri_encoder = UNet3DEncoder(in_channels=2, out_features=mri_feature_dim) # 2 channels for T1c and Flair
        # self.us_encoder = UNet2DEncoder(in_channels=2, out_features=us_feature_dim)   # 2 channels for US data

        self.multi_task = True if isinstance(num_classes, list) else False
        if isinstance(num_classes, int):
            num_classes_task1 = num_classes
        elif isinstance(num_classes, list) and len(num_classes) == 3:
            num_classes_task1, num_classes_task2, num_classes_task3 = num_classes
        else: raise ValueError('num_classes must be an integer or a list of integers')

        self.reduction = reduction
        if self.reduction == 'attention':
            # 使 mri_feature_dim == us_feature_dim
            self.attention = nn.Sequential(
                nn.Linear(mri_feature_dim, mri_feature_dim // 2),
                nn.ReLU(),
                nn.Linear(mri_feature_dim // 2, 1)
            )
            self.out_dim = mri_feature_dim

        elif self.reduction == 'avg':
            # fusion_dim = mri_feature_dim + us_feature_dim
            self.out_dim = mri_feature_dim
            
        else:
            raise ValueError("Invalid reduction method. Choose 'attention' or 'avg'.")

        self.clip_encoder, self.us_preprocess = self.build_biomedclip()

        self.build_2d_sam = build_2d_sam
        if self.build_2d_sam:
            self.us_sam_encoder = self.build_sam2d()
            self.us_target_size = (256, 256)
        else:
            self.us_sam_encoder = self.clip_encoder
            self.us_target_size = (224, 224)

        self.mri_sam_encoder = self.build_sam3d()
        self.wsi_encoder = self.build_mil()

        self.classifier_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task1)   # 肿瘤的类别： 3
            )
        
        self.us_classifier_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task1)
            )
        
        if self.multi_task:
            self.classifier_head2 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task2)             # idh的类别： 2
            )
            self.classifier_head3 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task2)             # 1p19q的类别： 2
            )
            self.us_classifier_head2 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task3)
            )
            self.us_classifier_head3 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes_task3)
            )
    
    def build_biomedclip(self):
        from open_clip import create_model_from_pretrained, get_tokenizer
        biomedclip, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # texts = model.tokenizer(['this is ...', 'that is ...'], context_length=256)
        # image_features, text_features, logit_scale = model(torch.randn(2, 3, 224, 224), texts)
        # print(F.cosine_similarity(image_features.repeat(2, 1), text_features.repeat(2, 1))) # 查看匹配的情况
        # 定义标准化变换
        preprocess = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        # 应用到图像张量 (假设图像已经转换为tensor且值在[0,1]范围)
        # normalized_image = normalize(image_tensor)
        return biomedclip, preprocess

    def build_sam3d(self):
        sam3d = build_sam3D_vit_b_ori()
        # /database/wuyonghuang/WSA/relys/sam_med3d_turbo.pth
        sam3d_ckpt = torch.load('/database/wuyonghuang/WSA/relys/sam_med3d_turbo.pth')
        sam3d.load_state_dict(sam3d_ckpt['model_state_dict'], strict=False)
        # sam3d_img_encoder = deepcopy()
        return sam3d.image_encoder
        # res = sam3d.image_encoder(torch.randn(1, 1, 128, 128, 128)) # return: (1, 384, 8, 8, 8)
    
    def build_sam2d(self):
        from relys.SAM_Med2D_main import build_sam_vit_b
        class ARGS: pass
        args = ARGS()
        args.image_size = 256
        args.sam_checkpoint = None
        args.encoder_adapter = True
        sam2d = build_sam_vit_b(args)
        sam2d_ckpt = torch.load("/database/wuyonghuang/WSA/relys/sam-med2d_b.pth")
        sam2d.load_state_dict(sam2d_ckpt['model'], strict=False)
        # sam2d_img_encoder = deepcopy()
        return sam2d.image_encoder
    
    def build_mil(self):
        from relys.TransMIL import build_transmil
        model = build_transmil(input_dim=1536, n_classes=None, out_feat_dim=512)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)  # 更推荐的方式
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        return model.apply(init_weights)

    def forward(self, mri_data, us_data, wsi_data, mri_flags, us_available_flags, text_data=None, align_loss=None):
        mri_available_flags, mri2_flags = mri_flags
        # mri_data: (B, 2, 64, 64, 64) - T1c and Flair concatenated or handled
        # us_data: (B, 2, 256, 256)
        # mri_available_flags: list of bools (length B), True if MRI is present for patient i
        # us_available_flags: list of bools (length B), True if US is present for patient i

        batch_size = mri_data.size(0)
        device = mri_data.device

        # # Process MRI: # note: 忽略对 t1c和flair(mri2_flags)缺失的处理. 
        # mri_features = self.mri_encoder(mri_data) # (B, mri_feature_dim)
        # us_features = self.us_encoder(us_data)     # (B, us_feature_dim)

        # note: 使用 SAM-2d 和 3d
        ada_mri_data = F.adaptive_avg_pool3d(mri_data, (128, 128, 128))
        ada_mri_data = rearrange(ada_mri_data, 'b c d h w -> (b c) d h w')
        rearrange_ada_mri_data = ada_mri_data.unsqueeze(dim=1)
        rearrange_ada_mri_feat = self.mri_sam_encoder(rearrange_ada_mri_data)  # (B, 384)
        mri_features = rearrange(rearrange_ada_mri_feat, '(b c) d -> b c d', c=2).mean(dim=1) # (B, 786); 因为mri可能缺失, 因此先取平均, 然后再 调整 向量维度
        mri_features = F.adaptive_avg_pool1d(mri_features, 512)

        # note: 支持使用 biomedclip 来提取 us 数据
        us_N = us_data.shape[1]
        rearrange_ada_us_data = rearrange(us_data, 'b c h w -> (b c) h w').unsqueeze(dim=1)
        rearrange_ada_us_data = F.adaptive_avg_pool2d(rearrange_ada_us_data, self.us_target_size).repeat(1, 3, 1, 1)
        if self.build_2d_sam:
            rearrange_ada_us_feat = self.us_sam_encoder(rearrange_ada_us_data)   # (B, 256)
        else:
            rearrange_ada_us_data = self.us_preprocess(rearrange_ada_us_data/255)
            rearrange_ada_us_feat, _, _ = self.us_sam_encoder(rearrange_ada_us_data)

        # us_features = rearrange(rearrange_ada_us_feat, '(b c) d -> b (c d)', c=2)   # (B, 256 * us_N)   # note: 待改进
        us_features = torch.mean(rearrange(rearrange_ada_us_feat, '(b c) d -> b c d', c=us_N), dim=1)
        # if us_features.shape[-1] != 512:
        #     us_features = F.adaptive_avg_pool1d(us_features, 512)
        
        mri_available_flags = torch.tensor(mri_available_flags, device=device, dtype=torch.bool)  # (B,)
        mri_features = mri_features * mri_available_flags.unsqueeze(1)  # 广播操作
        us_available_flags = torch.tensor(us_available_flags, device=device, dtype=torch.bool)  # (B,)
        us_features = us_features * us_available_flags.unsqueeze(1)  # 广播操作

        wsi_features_lst = [self.wsi_encoder(i) if len(i)!=0 else torch.zeros(1, 512).to(device) for i in wsi_data]
        wsi_features = torch.concat(wsi_features_lst, dim=0)  # stack

        if text_data is not None:
            text_emb = self.tokenizer(text_data, context_length=256).to(device)
            _, text_feat, logit_scale = self.clip_encoder(None, text_emb)
            if text_feat.shape[-1] != 512:
                text_feat = F.adaptive_avg_pool1d(text_feat, 512)

        if self.reduction == 'attention':
            if text_data is not None:
                fused_features = torch.stack((mri_features, us_features, wsi_features, text_feat), dim=1)  # (B, 3, 512)
            else:
                fused_features = torch.stack((mri_features, us_features, wsi_features), dim=1)  # (B, 3, 512)

            attn_weights = self.attention(fused_features)  # (B, C, 1)
            attn_weights = F.softmax(attn_weights, dim=1)  # (B, C, 1)
            fused_features = torch.sum(fused_features * attn_weights, dim=1)  # (B, D)

        elif self.reduction == 'avg':
            if text_data is not None:
                fused_features = torch.stack((mri_features, us_features, wsi_features, text_feat), dim=1)  # (B, 3, 512)
            else:
                fused_features = torch.stack((mri_features, us_features, wsi_features), dim=1)  # (B, 3, 512)

            # 特征拼接
            fused_features = torch.mean(fused_features, dim=1)
        
        logits = self.classifier_head(fused_features) # (B, num_classes)
        us_logits = self.us_classifier_head(us_features)

        other_results_dict = {}

        if align_loss is not None:
            align_loss = F.mse_loss(us_features, fused_features.detach())
            other_results_dict.update({
                'align_loss': align_loss
            })
        else:
            other_results_dict.update({'align_loss': None})

        if self.multi_task:
            logits2 = self.classifier_head2(fused_features) # (B, num_classes)
            us_logits2 = self.us_classifier_head2(us_features)

            logits3 = self.classifier_head3(fused_features) # (B, num_classes)
            us_logits3 = self.us_classifier_head3(us_features)

            # 将四个logits放进 dict中
            other_results_dict.update({
                'logits2': logits2,
                'us_logits2': us_logits2,
                'logits3': logits3,
                'us_logits3': us_logits3
            })
        return logits, us_logits, other_results_dict


def get_mri_data(data, mri_modality='T1c', batchsize=1, batch_available_flags=None):
    tmp_mri = torch.zeros(batchsize, *(224, 224, 224))
    for i in range(batchsize):
        if batch_available_flags[i]:
            tmp_mri[i] = data['data_mri'][mri_modality][i]
    return tmp_mri


def process_mri_data(i_data, batchsize, device='cuda'):
    mri_t1c = get_mri_data(i_data, mri_modality='T1c', batchsize=batchsize, batch_available_flags=i_data['mri_modality'][0]).to(device)
    mri_flair = get_mri_data(i_data, mri_modality='Flair', batchsize=batchsize, batch_available_flags=i_data['mri_modality'][1]).to(device)

    # Reshape to add channel dimension if they are (B, D, H, W)
    if mri_t1c.ndim == 4: mri_t1c = mri_t1c.unsqueeze(1)
    if mri_flair.ndim == 4: mri_flair = mri_flair.unsqueeze(1)

    # Concatenate T1c and Flair to form a 2-channel 3D image
    mri_input = torch.cat((mri_t1c, mri_flair), dim=1) # (B, 2, 64, 64, 64)

    return mri_input


def load_args(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    args = load_args("args_config_20250512_095806.pkl")
    args.id_files = '/database/wuyonghuang/WSA/mine_task/three_modality.xlsx'
    print(args)

    # 使用 argparse 接收命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    parser.add_argument('--exp_name', '-e', type=str, default='debug')
    args2 = parser.parse_args()

    # 将args2更新到 args中
    args.__dict__.update(args2.__dict__)

    # 新的参数设置
    args.num_classes = [3, 2, 2]
    args.reduction = 'attention'
    args.task_wegiht = 1.
    args.ckpt = '/database/wuyonghuang/WSA/results/test1/tumor_classifier_mri_us_epoch15_all-0.812_us-0.812.pth'
    args.tune_mode = ['classifier', 'us_sam_encoder']  # 'all' ['clip_encoder' 'us_sam_encoder' 'mri_sam_encoder' 'wsi_encoder' 'classifier']

    device = 'cuda'
    batchsize = 2
    num_epochs = 100 # Example

    if args.debug:
        snap_dir = os.path.join('/database/wuyonghuang/WSA', 'results', 'debug')
    else:
        snap_dir = os.path.join('/database/wuyonghuang/WSA', 'results', args.exp_name)
    
    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)
    
    writer = SummaryWriter(log_dir=snap_dir)

    tumor_label_map = {'glioblastoma': 0, 'astrocytoma': 1, 'oligodendroglioma': 2}    
    num_tumor_classes = len(tumor_label_map)

    idh_label_map = {'IDH wild type': 0, 'IDH mutant': 1}
    pq_label_map = {'无共缺失': 0, '共缺失': 1}

    data = get_data(args, epoch_id=0, max_txt_length=args.context_length, is_mine=not args.origin_task, use_wsi_coord=args.use_wsi_coord, single_version=True)
    dataloader = DataLoader(
                data['train'].dataset,
                batch_size=batchsize,
                pin_memory=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
    val_dataloader = DataLoader(
                data['val'].dataset,
                batch_size=batchsize,
                pin_memory=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )

    model = TumorClassifier(mri_feature_dim=512, us_feature_dim=512, num_classes=args.num_classes, reduction=args.reduction, build_2d_sam=False) # 意味着使用 biomed_clip 而不是 sam2d
    
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt), strict=False)

    if not args.debug:
        def nan_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"NaN detected in {module.__class__.__name__}")
                    raise RuntimeError("NaN detected")

        # 为所有层添加钩子
        for name, module in model.named_modules():
            module.register_forward_hook(nan_hook)

    # model = torch.nn.DataParallel(model)
    model.cuda()

    if args.tune_mode == 'all':
        print(f"Total parameters: {count_parameters(model)}")

    elif isinstance(args.tune_mode, list):
        for name, param in model.named_parameters():
            if name.split('.')[0] in args.tune_mode:
                param.requires_grad = True
            else:
                param.requires_grad = False
            if 'classifier' in args.tune_mode and 'classifier' in name:
                param.requires_grad = True

    else: raise ValueError("tune_mode should be 'all' or a list of indices")

    count_parameters(model)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.functional.cross_entropy

    best_accuracy1, best_us_accuracy1 = 0., 0.
    best_accuracy2, best_us_accuracy2 = 0., 0.
    best_accuracy3, best_us_accuracy3 = 0., 0.

    global_step = 0  # 用于记录全局步骤
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions1, us_correct_predictions1 = 0, 0
        correct_predictions2, us_correct_predictions2 = 0, 0
        correct_predictions3, us_correct_predictions3 = 0, 0

        total_tumor_samples, total_idh_samples, total_pq_samples = 0, 0, 0

        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        for batch_idx, i_data in tqdm(enumerate(dataloader), desc='Training'):
            model.train()
            # --- 1. Prepare Inputs ---
            # MRI data (T1c and Flair)
            # data_mri is {'T1c': tensor, 'Flair': tensor}
            # mri_modality indicates if T1c (idx 0) or Flair (idx 1) is missing (False means missing, zero-filled)
            # Concatenate T1c and Flair along channel dimension.
            # If a sub-modality is missing, its tensor is already zero-filled by dataloader as per prior discussion.
            mri_input = process_mri_data(i_data, batchsize, device=device)
            us_input = i_data['data_us'].to(device) # (B, 2, 256, 256)
            wsi_input = [i.to(device) for i in i_data['data_wsi']]

            # Availability flags (check 'case_id' for nan)
            # case_id: [['wsi_id1', 'wsi_id2'], ['us_id1', 'us_id2'], ['mri_id1', 'mri_id2']]
            # MRI is at index 2, US is at index 1 in case_id list
            mri_ids_for_batch = i_data['case_id'][2]    # note: 这是对整体的mri的标记, 不为nan的话, 就说明存在一个以上的mri模态
            us_ids_for_batch = i_data['case_id'][1]

            mri_available_flags = [not (isinstance(id_val, float) and math.isnan(id_val)) for id_val in mri_ids_for_batch]
            t1c_flair_available_flags = i_data['mri_modality']
            us_available_flags = [not (isinstance(id_val, float) and math.isnan(id_val)) for id_val in us_ids_for_batch]
            
            # --- 2. Prepare Labels (Tumor Classification) ---
            # i_data['label'][0] contains string labels like ['glioblastoma', 'glioblastoma']
            tumor_labels_str = i_data['label'][0]
            tumor_labels_int = torch.tensor([tumor_label_map[label] for label in tumor_labels_str], dtype=torch.long).to(device)

            idh_labels_str = i_data['label'][1]
            idh_labels_int = torch.tensor([idh_label_map[label] for label in idh_labels_str], dtype=torch.long).to(device)

            pq_labels_str = i_data['label'][2]
            pq_labels_int = torch.tensor([pq_label_map[label] for label in pq_labels_str], dtype=torch.long).to(device)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()

            # with autocast():
            logits1, us_logits1, other_dict = model(mri_data=mri_input, 
                            us_data=us_input.float(), 
                            wsi_data=wsi_input, 
                            mri_flags=(mri_available_flags, t1c_flair_available_flags), 
                            us_available_flags=us_available_flags,
                            text_data=i_data['text_tumor'], align_loss=True)
            loss11 = criterion(logits1, tumor_labels_int)
            loss12 = criterion(us_logits1, tumor_labels_int)

            loss = loss11 + loss12
            writer.add_scalar('Loss/loss1', loss11.item(), global_step)
            writer.add_scalar('Loss/loss2', loss12.item(), global_step)

            if other_dict.get('align_loss') is not None:
                loss += other_dict['align_loss']
                writer.add_scalar('Loss/align_loss', other_dict['align_loss'].item(), global_step)

            if len(other_dict) != 0:
                loss21 = criterion(other_dict['logits2'], idh_labels_int)
                loss22 = criterion(other_dict['us_logits2'], idh_labels_int)
                loss31 = criterion(other_dict['logits3'], pq_labels_int)
                loss32 = criterion(other_dict['us_logits3'], pq_labels_int)

                loss += args.task_wegiht * (loss21 + loss22 + loss31 + loss32)

                writer.add_scalar('Loss/loss21', loss21.item(), global_step)
                writer.add_scalar('Loss/loss22', loss22.item(), global_step)
                writer.add_scalar('Loss/loss31', loss31.item(), global_step)
                writer.add_scalar('Loss/loss32', loss32.item(), global_step)

            global_step += 1  # 更新全局步骤

            # 检查loss
            if torch.isnan(loss):
                print(f"NaN loss at batch {batch_idx}")
                print(f"mri_input stats: min={mri_input.min()}, max={mri_input.max()}")
                print(f"us_input stats: min={us_input.min()}, max={us_input.max()}")
                print(f"wsi_input stats: min={min([i.min() for i in wsi_input])}, max={max([i.max() for i in wsi_input])}")
                print(f"Output stats: min={logits1.min()}, max={us_logits1.max()}")
                break

            # --- 5. Backward pass and optimize ---
            loss.backward()
            optimizer.step()
            # 缩放损失，反向传播
            # scaler.scale(loss).backward()

            # 检查梯度
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            if total_norm > 1000:  # 梯度过大
                print(f"Large gradient norm: {total_norm}")

            # 更新参数
            # scaler.step(optimizer)
            # # 更新缩放器
            # scaler.update()
            
            # --- 6. Logging ---
            total_loss += loss.item()
            _, predicted_classes = torch.max(logits1, 1)
            _, us_predicted_classes = torch.max(us_logits1, 1)

            # correct_predictions += (predicted_classes == tumor_labels_int).sum().item()
            # total_samples += tumor_labels_int.size(0)

            if len(other_dict) == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Loss1: {loss11.item():.4f}, Loss2: {loss12.item():.4f}")
            else:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, \
                      loss11: {loss11.item():.4f}, loss12: {loss12.item():.4f}, \
                      Loss21: {loss21.item():.4f}, Loss22: {loss22.item():.4f}, \
                      Loss31: {loss31.item():.4f}, Loss32: {loss32.item():.4f}, ")
            
            if args.debug and batch_idx == 2:
                break
        if args.debug:
            break

        with torch.no_grad():
            model.eval()
            for batch_idx, i_data in tqdm(enumerate(val_dataloader), desc='Validation'):
                mri_input = process_mri_data(i_data, batchsize, device=device)
                us_input = i_data['data_us'].to(device) # (B, 2, 256, 256)
                wsi_input = [i.to(device) for i in i_data['data_wsi']]

                mri_ids_for_batch = i_data['case_id'][2]
                us_ids_for_batch = i_data['case_id'][1]

                mri_available_flags = [not (isinstance(id_val, float) and math.isnan(id_val)) for id_val in mri_ids_for_batch]
                t1c_flair_available_flags = i_data['mri_modality']
                us_available_flags = [not (isinstance(id_val, float) and math.isnan(id_val)) for id_val in us_ids_for_batch]
                
                tumor_labels_str = i_data['label'][0]
                tumor_labels_int = torch.tensor([tumor_label_map[label] for label in tumor_labels_str], dtype=torch.long).to(device)

                idh_labels_str = i_data['label'][1]
                idh_labels_int = torch.tensor([idh_label_map[label] for label in idh_labels_str], dtype=torch.long).to(device)

                pq_labels_str = i_data['label'][2]
                pq_labels_int = torch.tensor([pq_label_map[label] for label in pq_labels_str], dtype=torch.long).to(device)

                logits1, us_logits1, other_dict = model(mri_input, us_input.float(), wsi_input, (mri_available_flags, t1c_flair_available_flags), us_available_flags,
                                                       text_data=i_data['text_tumor'])
                
                # 计算 肿瘤分类的 结果
                _, predicted_classes = torch.max(logits1, 1)
                correct_predictions1 += (predicted_classes == tumor_labels_int).sum().item()
                _, us_predicted_classes = torch.max(us_logits1, 1)
                us_correct_predictions1 += (us_predicted_classes == tumor_labels_int).sum().item()
                total_tumor_samples += tumor_labels_int.size(0)

                if len(other_dict) != 0 and len(args.num_classes) == 3:
                    # 计算 idh 分类的 结果
                    _, predicted_classes = torch.max(other_dict['logits2'], 1)
                    correct_predictions2 += (predicted_classes == idh_labels_int).sum().item()
                    _, us_predicted_classes = torch.max(other_dict['us_logits2'], 1)
                    us_correct_predictions2 += (us_predicted_classes == idh_labels_int).sum().item()
                    total_idh_samples += idh_labels_int.size(0)

                    # 计算 1p19q 分类的 结果
                    _, predicted_classes = torch.max(other_dict['logits3'], 1)
                    correct_predictions3 += (predicted_classes == pq_labels_int).sum().item()
                    _, us_predicted_classes = torch.max(other_dict['us_logits3'], 1)
                    us_correct_predictions3 += (us_predicted_classes == pq_labels_int).sum().item()
                    total_pq_samples += pq_labels_int.size(0)

                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(val_dataloader)}")
                if args.debug and batch_idx > 5:
                    break
        
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_accuracy1 = correct_predictions1 / total_tumor_samples
        epoch_us_accuracy1 = us_correct_predictions1 / total_tumor_samples
        
        if isinstance(args.num_classes, list):
            epoch_accuracy2 = correct_predictions2 / total_idh_samples
            epoch_us_accuracy2 = us_correct_predictions2 / total_idh_samples

            epoch_accuracy3 = correct_predictions3 / total_pq_samples
            epoch_us_accuracy3 = us_correct_predictions3 / total_pq_samples
            print(f"Epoch {epoch} Summary: Avg Loss: {avg_epoch_loss:.4f}, \
                  Val_all_Accuracy1: {epoch_accuracy1:.4f}, Val_us_Accuracy1: {epoch_us_accuracy1:.4f}, \
                  Val_all_Accuracy2: {epoch_accuracy2:.4f}, Val_us_Accuracy2: {epoch_us_accuracy2:.4f}, \
                  Val_all_Accuracy3: {epoch_accuracy3:.4f}, Val_us_Accuracy3: {epoch_us_accuracy3:.4f}, ")
        else:
            print(f"Epoch {epoch} Summary: Avg Loss: {avg_epoch_loss:.4f}, Val_all_Accuracy: {epoch_accuracy1:.4f}, Val_us_Accuracy: {epoch_us_accuracy1:.4f}")

        # 根据 epoch_accuracy 写earlystop, 当指标更好的时候就 保存模型
        is_save1, is_save2 = False, False
        if epoch_accuracy1 > best_accuracy1:
            best_accuracy1 = epoch_accuracy1
            is_save1 = True
        
        if epoch_us_accuracy1 > best_us_accuracy1:
            best_us_accuracy1 = epoch_us_accuracy1
            is_save2 = True
        
        if is_save1 or is_save2:
            torch.save(model.state_dict(), 
                os.path.join(snap_dir, f"tumor_cls_epoch{epoch}_all-{epoch_accuracy1:.03f}_us-{epoch_us_accuracy1:.03f}.pth"))
        
        is_save1, is_save2 = False, False
        if epoch_accuracy2 > best_accuracy2:
            best_accuracy2 = epoch_accuracy2
            is_save1 = True
        
        if epoch_us_accuracy2 > best_us_accuracy2:
            best_us_accuracy2 = epoch_us_accuracy2
            is_save2 = True
        
        if is_save1 or is_save2:
            torch.save(model.state_dict(), 
                os.path.join(snap_dir, f"idh_cls_epoch{epoch}_all-{epoch_accuracy2:.03f}_us-{epoch_us_accuracy2:.03f}.pth"))
        
        is_save1, is_save2 = False, False
        if epoch_accuracy3 > best_accuracy3:
            best_accuracy3 = epoch_accuracy3
            is_save1 = True
        
        if epoch_us_accuracy3 > best_us_accuracy3:
            best_us_accuracy3 = epoch_us_accuracy3
            is_save2 = True
        
        if is_save1 or is_save2:
            torch.save(model.state_dict(), 
                os.path.join(snap_dir, f"pq_cls_epoch{epoch}_all-{epoch_accuracy3:.03f}_us-{epoch_us_accuracy3:.03f}.pth"))

    writer.close()
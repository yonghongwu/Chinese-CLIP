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

        self.reduction = reduction
        if self.reduction == 'attention':
            # 使 mri_feature_dim == us_feature_dim
            self.attention = nn.Sequential(
                nn.Linear(mri_feature_dim, mri_feature_dim // 2),
                nn.ReLU(),
                nn.Linear(mri_feature_dim // 2, 1)
            )
            self.out_dim = mri_feature_dim

        elif self.reduction == 'concat':
            fusion_dim = mri_feature_dim + us_feature_dim
            self.out_dim = fusion_dim
            
        else:
            raise ValueError("Invalid reduction method. Choose 'attention' or 'concat'.")

        self.build_2d_sam = build_2d_sam
        if self.build_2d_sam:
            self.us_sam_encoder = self.build_sam2d()
            self.us_target_size = (256, 256)
        else:
            self.us_sam_encoder, self.us_preprocess = self.build_biomedclip()
            self.us_target_size = (224, 224)

        self.mri_sam_encoder = self.build_sam3d()
        self.wsi_encoder = self.build_mil()

        self.classifier_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.out_dim // 2, num_classes)
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

    def forward(self, mri_data, us_data, wsi_data, mri_flags, us_available_flags, text_data=None):
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
        rearrange_ada_us_data = rearrange(us_data, 'b c h w -> (b c) h w').unsqueeze(dim=1)
        rearrange_ada_us_data = F.adaptive_avg_pool2d(rearrange_ada_us_data, self.us_target_size).repeat(1, 3, 1, 1)
        if self.build_2d_sam:
            rearrange_ada_us_feat = self.us_sam_encoder(rearrange_ada_us_data)   # (B, 256)
        else:
            rearrange_ada_us_data = self.us_preprocess(rearrange_ada_us_data/255)
            rearrange_ada_us_feat, _, _ = self.us_sam_encoder(rearrange_ada_us_data)

        us_features = rearrange(rearrange_ada_us_feat, '(b c) d -> b (c d)', c=2)   # (B, 256 * us_N)   # note: 待改进
        if us_features.shape[-1] != 512:
            us_features = F.adaptive_avg_pool1d(us_features, 512)
        
        mri_available_flags = torch.tensor(mri_available_flags, device=device, dtype=torch.bool)  # (B,)
        mri_features = mri_features * mri_available_flags.unsqueeze(1)  # 广播操作
        us_available_flags = torch.tensor(us_available_flags, device=device, dtype=torch.bool)  # (B,)
        us_features = us_features * us_available_flags.unsqueeze(1)  # 广播操作

        wsi_features_lst = [self.wsi_encoder(i) if len(i)!=0 else torch.zeros(1, 512).to(device) for i in wsi_data]
        wsi_features = torch.concat(wsi_features_lst, dim=0)  # stack

        # note: 通道融合
        if self.reduction == 'attention':
            fused_features = torch.stack((mri_features, us_features, wsi_features), dim=1)  # (B, 3, 512)
            attn_weights = self.attention(fused_features)  # (B, C, 1)
            attn_weights = F.softmax(attn_weights, dim=1)  # (B, C, 1)
            fused_features = torch.sum(fused_features * attn_weights, dim=1)  # (B, D)
            # logits = self.classifier(x)  # (B, num_classes)

        elif self.reduction == 'concat':
            # 特征拼接
            fused_features = torch.cat((mri_features, us_features), dim=1) # (B, fusion_dim)
        
        logits = self.classifier_head(fused_features) # (B, num_classes)

        return logits


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

    if args.debug:
        snap_dir = os.path.join('/database/wuyonghuang/WSA', 'debug')
    else:
        snap_dir = os.path.join('/database/wuyonghuang/WSA', args.exp_name)
    
    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    batchsize = 2

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

    device = 'cuda'
    model = TumorClassifier(build_2d_sam=False) # 意味着使用 biomed_clip 而不是 sam2d

    def nan_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"NaN detected in {module.__class__.__name__}")
                raise RuntimeError("NaN detected")

    # 为所有层添加钩子
    for name, module in model.named_modules():
        module.register_forward_hook(nan_hook)

    model = torch.nn.DataParallel(model)
    model.cuda()

    # mri_sam_encoder_length = len(list(model.mri_sam_encoder.parameters()))
    # for idx, param in enumerate(model.mri_sam_encoder.parameters()):
    #     if mri_sam_encoder_length - idx >= 20:
    #         param.requires_grad = False

    # mri_us_encoder_length = len(list(model.us_sam_encoder.parameters()))
    # for idx, param in enumerate(model.us_sam_encoder.parameters()):
    #     if mri_us_encoder_length - idx >= 20:
    #         param.requires_grad = False

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.functional.cross_entropy

    num_epochs = 100 # Example
    best_accuracy = 0.
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

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
            text_input = model.tokenizer(i_data['text_tumor'], context_length=256).to(device)

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

            with autocast():
                logits = model(mri_data=mri_input, 
                               us_data=us_input.float(), 
                               wsi_data=wsi_input, 
                               mri_flags=(mri_available_flags, t1c_flair_available_flags), 
                               us_available_flags=us_available_flags,
                               text_data=None)
                loss = criterion(logits, tumor_labels_int)

                # 检查loss
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}")
                    print(f"mri_input stats: min={mri_input.min()}, max={mri_input.max()}")
                    print(f"us_input stats: min={us_input.min()}, max={us_input.max()}")
                    print(f"wsi_input stats: min={min([i.min() for i in wsi_input])}, max={max([i.max() for i in wsi_input])}")
                    print(f"Output stats: min={logits.min()}, max={logits.max()}")
                    break

            # --- 5. Backward pass and optimize ---
            # loss.backward()
            # optimizer.step()
            # 缩放损失，反向传播
            scaler.scale(loss).backward()

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
            scaler.step(optimizer)
            # 更新缩放器
            scaler.update()
            
            # --- 6. Logging ---
            total_loss += loss.item()
            _, predicted_classes = torch.max(logits, 1)
            # correct_predictions += (predicted_classes == tumor_labels_int).sum().item()
            # total_samples += tumor_labels_int.size(0)

            print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
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

                logits = model(mri_input, us_input.float(), wsi_input, (mri_available_flags, t1c_flair_available_flags), us_available_flags)
                
                _, predicted_classes = torch.max(logits, 1)
                correct_predictions += (predicted_classes == tumor_labels_int).sum().item()
                total_samples += tumor_labels_int.size(0)

                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(val_dataloader)}")

        avg_epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch} Summary: Avg Loss: {avg_epoch_loss:.4f}, Val_Accuracy: {epoch_accuracy:.4f}")
        # 根据 epoch_accuracy 写earlystop, 当指标更好的时候就 保存模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), 
                       os.path.join(snap_dir, f"tumor_classifier_mri_us_epoch{epoch}_{epoch_accuracy:.03f}.pth"))
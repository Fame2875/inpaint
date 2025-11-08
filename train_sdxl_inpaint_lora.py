# train_sdxl_inpaint_lora_customloss_fixed2.py

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoPipelineForInpainting
from peft import get_peft_model, LoraConfig
import wandb

# ====== CONFIG ======
IMG_DIR = "Data/target_only"
MASK_DIR = "Data/diffusion_masks"
BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
OUTPUT_DIR = "./lora_inpaint_sdxl_customloss_fixed2"
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
LORA_RANK = 4
LORA_ALPHA = 16

# Target modules for LoRA in the UNet (attention layers)
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]

W_L2_MASK     = 1.0
W_BACKGROUND  = 0.5
W_PERCEPTUAL  = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ====================

class InpaintDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images       = sorted(os.listdir(images_dir))
        self.masks        = sorted(os.listdir(masks_dir))
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.image_transform = image_transform
        self.mask_transform  = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir,  self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.slice(x)

def compute_perceptual_loss(fe, out, tgt):
    out_feat = fe(out)
    tgt_feat = fe(tgt)
    return nn.functional.mse_loss(out_feat, tgt_feat)

def main():
    wandb.init(project="inpainting_sdxl_lora_fixed2", config={
        "base_model": BASE_MODEL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
    })

    # Image transform: range [0,1] then normalized to roughly [-1,1]
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),                  # gives [C, H, W] in [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])  # now ~[-1,1]
    ])
    # Mask transform: single channel, no 3-channel normalization
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()                   # gives [1, H, W] in [0,1]
        # Optionally: normalize if you want but ensure correct channels
    ])

    dataset = InpaintDataset(IMG_DIR, MASK_DIR, image_transform, mask_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading base model:", BASE_MODEL)
    pipeline = AutoPipelineForInpainting.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to(DEVICE)

    unet = pipeline.unet
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    pipeline.unet = unet

    # Freeze non‐adapter parts
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False

    optimizer     = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    feat_extractor= VGGFeatureExtractor().to(DEVICE)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for step, (imgs, masks) in enumerate(loader):
            imgs  = imgs.to(DEVICE, dtype=torch.float16)
            masks = masks.to(DEVICE, dtype=torch.float16)

            out = pipeline(image=imgs, mask_image=masks, prompt="repair target person in background").images[0]
            out = out.unsqueeze(0) if out.dim()==3 else out

            mask_bool = masks > 0.5
            l2_mask  = nn.functional.mse_loss(out[mask_bool], imgs[mask_bool])

            mask_inv  = ~mask_bool
            if mask_inv.sum() > 0:
                l2_bg = nn.functional.mse_loss(out[mask_inv], imgs[mask_inv])
            else:
                l2_bg = torch.tensor(0.0, device=DEVICE)

            perc_loss = compute_perceptual_loss(feat_extractor, out, imgs)

            loss = W_L2_MASK * l2_mask + W_BACKGROUND * l2_bg + W_PERCEPTUAL * perc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                wandb.log({
                    "step":      step + epoch * len(loader),
                    "loss":      loss.item(),
                    "l2_mask":   l2_mask.item(),
                    "l2_bg":     l2_bg.item(),
                    "perc_loss": perc_loss.item()
                })

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — Avg Loss: {avg_loss:.6f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})

        ckpt_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        pipeline.save_pretrained(ckpt_path)
        wandb.save(os.path.join(ckpt_path, "pytorch_model.bin"))

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()

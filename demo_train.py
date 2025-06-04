import time, sys, os, glob
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.heads.dpt_head import DPTHead
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 1. Dataset
class ImageDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

        if self.labels:
            assert len(self.img_paths)==len(labels), \
                f"Number of image paths ({len(self.img_paths)}) does not match number of labels ({len(labels)})"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # ───────────────────────────── 1. Transform ─────────────────────────────
        if self.transform==load_and_preprocess_images:
            img = self.transform([self.img_paths[idx]])
        elif self.transform:
            img = self.transform(img)

        # ───────────────────────────── 2. Labels ─────────────────────────────
        if self.labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img

def train():
    pass



if __name__ == "__main__":
    # ───────────────────────────── 1. device / dtype ─────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Device: {device},  Mixed-Precision dtype: {dtype}")

    # ───────────────────────────── 2. load image paths ─────────────────────────────
    raw_dir = "/home/team17/vggt/examples/room_2/images"
    image_dir = os.path.expanduser(raw_dir)
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    image_paths = []
    for pat in patterns:
        image_paths += glob.glob(os.path.join(image_dir, pat))
    image_paths = sorted(image_paths)
    
    # <<< 调试：打印一下到底扫到了哪些文件
    print(f"Found {len(image_paths)} images:")
    for p in image_paths:
        print("   ", p)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}! 请检查路径和后缀。")

    # ───────────────────────────── 3. Dataset & Dataloader ─────────────────────────────
    # 1) Dataset
    dataset = ImageDataset(image_paths, labels=None, transform=load_and_preprocess_images)

    # <<< 调试：打印一下前几张图片的尺寸
    print("⏳ Loading & preprocessing images ...")
    t0 = time.time()
    images = None
    for img in dataset:
        if images is not None:
            images = torch.cat((images, img), 0)
        else:
            images=img
    # images = load_and_preprocess_images(image_paths).to(device)
    F, _, H, W = images.shape
    print(f"✔ {F} frames  ({H}×{W})   ({time.time()-t0:.1f}s)")
    del(images)

    # 2) Dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # ───────────────────────────── 3. Model & Optimizer & Loss ─────────────────────────────
    print("⏳ Loading model ...")
    # 1) Load pretrained VGGT model ###
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2) Create or load Gaussian Head ###
    # Gaussian Num Channels =
        # 3D mean offsets (3) +
        # Scales (3) +
        # Rotations (4) +
        # Spherical Harmonics (3 * sh_degree) +
        # Opacity (1)
    sh_degree = 0
    gaussian_num_channels = 3 + 3 + 4 + 3 * (sh_degree+1)**2 + 1
    gaussian_head = Gaussianhead(dim_in=2 * model.embed_dim, output_dim=gaussian_num_channels, activation="exp", conf_activation="expp1").to(device)
    print(f"✔ Model ready   ({time.time()-t0:.1f}s)")

    # 3) Optimizer & Loss
    optimizer = torch.optim.Adam(gaussian_head.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ───────────────────────────── 4. tensorboard & tqdm ─────────────────────────────
    writer = SummaryWriter("runs/gaussian_head_only")

    # ───────────────────────────── 5. train ─────────────────────────────
    num_epochs = 5
    for epoch in range(1, num_epochs+1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        epoch_loss = 0.0

        for imgs in loop:
            imgs = imgs.to(device)

            # 1) Extract features from pretrained VGGT model (no grads flowing into backbone)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predict attributes including cameras, depth maps, and point maps.
                    # If without batch dimension, add it
                    if len(imgs.shape) == 4:
                        imgs = imgs.unsqueeze(0)
                    # Predict tokens
                    aggregated_tokens_list, ps_idx = model.aggregator(imgs)
                    # Predict Depth Maps
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, imgs, ps_idx)
                    # Predict Point Maps
                    point_map, point_conf = model.point_head(aggregated_tokens_list, imgs, ps_idx)
                    # Predict Cameras
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])

            # 2) Forward & backward on Gaussian head only
            gaussian_map = gaussian_head(aggregated_tokens_list, imgs, ps_idx, point_map)
            print(gaussian_map.keys())


        #     preds = gaussian_head(feats)          # -> (B, gaussian_num_channels, H, W)
        #     loss  = criterion(preds, targets)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     batch_loss = loss.item()
        #     epoch_loss += batch_loss
        #     loop.set_postfix(loss=batch_loss)

        #     writer.add_scalar("Loss/train_batch", batch_loss, global_step)
        #     global_step += 1

        # avg = epoch_loss / len(loader)
        # writer.add_scalar("Loss/train_epoch", avg, epoch)
        # print(f"Epoch {epoch:02d} • avg loss: {avg:.4f}")

    writer.close()

""" 
# 2. Hyperparameters & setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels    = [0, 1]  # dummy classes
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])
dataset = ImageDataset(img_paths, labels, transform)
loader  = DataLoader(dataset, batch_size=2, shuffle=True)

model = VGGT.from_pretrained("facebook/VGGT-1B")\
             .to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/vggt_overfit_demo')

# 3. Training loop with tqdm & TensorBoard
num_epochs = 100
global_step = 0

for epoch in range(1, num_epochs+1):
    epoch_loss = 0.0
    loop = tqdm(loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
    for imgs, targets in loop:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)         # (B, num_classes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # update tqdm bar
        loop.set_postfix(loss=batch_loss)

        # log to TensorBoard
        writer.add_scalar('Loss/train_batch', batch_loss, global_step)
        global_step += 1

    avg_loss = epoch_loss / len(loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    print(f'Epoch {epoch} — avg loss: {avg_loss:.4f}')

writer.close()
 """
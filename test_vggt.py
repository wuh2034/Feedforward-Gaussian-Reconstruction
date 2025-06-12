import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 设备与数据类型
device = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability()[0] if device=="cuda" else 0
dtype = torch.bfloat16 if cap >= 8 else torch.float16

# 加载模型
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device=device, dtype=dtype)

# 准备图像
image_paths = ["/Users/wuhan/Desktop/VGGT_test/im0.png", "/Users/wuhan/Desktop/VGGT_test/im1.png", "/Users/wuhan/Desktop/VGGT_test/im2.png"]
images = load_and_preprocess_images(image_paths).to(device=device)

# 推理
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        preds = model(images)

print(preds)

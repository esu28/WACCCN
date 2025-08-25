#loaded in pre-trained .pt file weights; from https://github.com/dalifreire/tumor_regions_segmentation
# model name: ORCA__Size-640x640_Epoch-100_Images-4181_Batch-1__color_augmentation

import torch

ckpt_path = model_path
ckpt = torch.load(ckpt_path, map_location="cpu")

model = UNet(
    in_channels=ckpt["model_in_channels"],   # 3
    out_channels=ckpt["model_out_channels"], # 1
    up_mode=ckpt["model_up_mode"],           
    padding=bool(ckpt["model_padding"]),     
    img_input_size=(640,640)                
)

ckpt_path1 = model_pathf
ckpt1 = torch.load(ckpt_path1, map_location="cpu")

model.load_state_dict(ckpt["model_state_dict"])

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms as T

model = model1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def run_inference_rgb(pil_img, model, device):
    np_rgb = np.array(pil_img) / 255.0
    np_lab = rgb_to_lab(np_rgb).astype("float32")
    X = torch.from_numpy(np_lab).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        y_hat = model(X)
        prob = y_hat.squeeze().cpu().numpy()

    return prob

pil_patch = Image.open(test_imgb).convert("RGB").crop((652, 317, 1292, 957))
prob_map = run_inference_rgb(pil_patch, model, device)

plt.subplot(1,2,1); plt.imshow(pil_patch); plt.title("RGB patch")
plt.subplot(1,2,2); plt.imshow(prob_map, cmap="viridis", vmin=0, vmax=1); plt.colorbar(); plt.title("Predicted Prob")
plt.show()

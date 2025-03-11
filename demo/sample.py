import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import mar
from models.vae import AutoencoderKL
from util import download
import time
import random

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")


diffloss_d = 6
diffloss_w = 1024
model_type = "mar_base" #@param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 100 #@param {type:"slider", min:1, max:1000, step:1}
model = mar.__dict__[model_type](
  buffer_size=64,
  diffloss_d=diffloss_d,
  diffloss_w=diffloss_w,
  num_sampling_steps=str(num_sampling_steps_diffloss)
).to(device)
state_dict = torch.load(f"work/pretrained_models/mar/{model_type}/checkpoint-last.pth".format(model_type))["model_ema"]
model.load_state_dict(state_dict)
model.eval() # important!
vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="work/pretrained_models/vae/kl16.ckpt").cuda().eval()

# Set user inputs:
#seed = 2 #@param {type:"number"}
seed = int(time.time()) #随机种子
torch.manual_seed(seed)
np.random.seed(seed)
num_ar_steps = 64 #@param {type:"slider", min:1, max:256, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.0 #@param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = 208, 361, 988, 3, 35, 98, 923, 9 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}

with torch.cuda.amp.autocast():
  sampled_tokens = model.sample_tokens(
      bsz=len(class_labels), num_iter=num_ar_steps,
      cfg=cfg_scale, cfg_schedule=cfg_schedule,
      labels=torch.Tensor(class_labels).long().cuda(),
      temperature=temperature, progress=True)
  sampled_images = vae.decode(sampled_tokens / 0.2325)

# Save and display images:
save_image(sampled_images, "./work/images/sampleRandom.png", nrow=int(samples_per_row), normalize=True, value_range=(-1, 1))


print(f'^_^ v0.0.3 ^_^')
import streamlit as st
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="centered"
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #f0f0f0;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    text-align: center;
    background: linear-gradient(90deg, #c9a96e, #f0e6d3, #c9a96e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a07840);
    color: #0f0f0f;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85;
}

.upload-box {
    border: 2px dashed #333;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #666;
}

.result-label {
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.info-box {
    background: #1a1a1a;
    border-left: 3px solid #c9a96e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #aaa;
}

.stDownloadButton > button {
    background: #1a1a1a;
    color: #c9a96e;
    border: 1px solid #c9a96e;
    border-radius: 8px;
    width: 100%;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Exact model architecture from your notebook
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu   = nn.ReLU(True)
        upnorm   = nn.BatchNorm2d(nf)
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down   = [downconv]
            up     = [uprelu, upconv, nn.Tanh()]
            model  = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down   = [downrelu, downconv]
            up     = [uprelu, upconv, upnorm]
            model  = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down   = [downrelu, downconv, downnorm]
            up     = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model  = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model  = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2**i, num_filters * 2**(i+1),
                                  s=1 if i == (n_down-1) else 2)
                  for i in range(n_down)]
        model += [self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act:  layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss() if gan_mode == 'vanilla' else nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        return self.loss(preds, self.get_labels(preds, target_is_real))


def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    net.apply(init_func)
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

def build_res_unet(n_input=1, n_output=2, size=256):
    weights  = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
    body   = create_body(backbone, cut=-2)
    net_G  = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device    = device
        self.lambda_L1 = lambda_L1
        if net_G is None:
            self.net_G = init_model(
                build_res_unet(n_input=1, n_output=2, size=256), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D        = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion  = nn.L1Loss()
        self.opt_G   = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D   = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()

    def forward(self):
        self.fake_color = self.net_G(self.L)


# ─────────────────────────────────────────────
# Load model (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    model = MainModel(net_G=net_G)
    model.load_state_dict(
        torch.load("final_model.pt", map_location=device, weights_only=False)
    )
    model.eval()
    return model


# ─────────────────────────────────────────────
# Colorize function (same logic as your notebook)
# ─────────────────────────────────────────────
def colorize(image: Image.Image, model, size=256) -> Image.Image:
    img = image.convert("RGB").resize((size, size), Image.BICUBIC)
    img_np  = np.array(img)
    img_lab = rgb2lab(img_np).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.
    L = L.unsqueeze(0).to(device)

    model.net_G.eval()
    with torch.no_grad():
        ab = model.net_G(L)

    L_out  = (L + 1.) * 50.
    ab_out = ab * 110.
    Lab    = torch.cat([L_out, ab_out], dim=1)
    Lab    = Lab.squeeze(0).permute(1, 2, 0).cpu().numpy()
    colorized = lab2rgb(Lab)
    colorized = (np.clip(colorized, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(colorized)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">✦ Colorize</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bring black & white photos back to life using a GAN trained from scratch</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">Upload a grayscale or black & white image. Works best on outdoor scenes, portraits, and street photography.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"],
                             label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image.convert("L"), caption="")
        st.markdown('<div class="result-label">Input</div>', unsafe_allow_html=True)

    if st.button("🎨 Colorize"):
        with st.spinner("Colorizing..."):
            try:
                model = load_model()
                result = colorize(image, model)

                with col2:
                    st.image(result, caption="")
                    st.markdown('<div class="result-label">Colorized</div>', unsafe_allow_html=True)

                # Download button
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    label="⬇ Download Result",
                    data=buf.getvalue(),
                    file_name="colorized.png",
                    mime="image/png"
                )

            except FileNotFoundError:
                st.error("❌ `final_model.pt` not found. Make sure it's in the same folder as `app.py`.")

else:
    st.markdown("""
    <div class="upload-box">
        Drop an image above to get started
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#444; font-size:0.8rem;">Built with PyTorch · ResNet18-UNet · PatchGAN</div>',
    unsafe_allow_html=True
)
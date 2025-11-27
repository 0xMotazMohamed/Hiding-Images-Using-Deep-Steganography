import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,50,3,padding=1)
        self.conv2 = nn.Conv2d(3,10,4,padding=1)
        self.conv3 = nn.Conv2d(3,5,5,padding=2)
        self.conv4 = ResnetBlock(65,50)
        self.conv5 = nn.Conv2d(65,10,4,padding=1)
        self.conv6 = nn.Conv2d(65,5,5,padding=2)
        self.conv7 = nn.Conv2d(68,50,3,padding=1)
        self.conv8 = nn.Conv2d(68,10,4,padding=1)
        self.conv9 = nn.Conv2d(68,5,5,padding=2)
        self.conv10 = ResnetBlock(65,50)
        self.conv11 = nn.Conv2d(65,10,4,padding=1)
        self.conv12 = nn.Conv2d(65,5,5,padding=2)
        self.conv13 = nn.Conv2d(65,50,3,padding=1)
        self.conv14 = nn.Conv2d(65,10,4,padding=1)
        self.conv15 = nn.Conv2d(65,5,5,padding=2)
        self.conv16 = ResnetBlock(65,50)
        self.conv17 = nn.Conv2d(65,10,4,padding=1)
        self.conv18 = nn.Conv2d(65,5,5,padding=2)
        self.conv19 = nn.Conv2d(65,50,3,padding=1)
        self.conv20 = nn.Conv2d(65,10,4,padding=1)
        self.conv21 = nn.Conv2d(65,5,5,padding=2)
        self.conv22 = nn.Conv2d(65,3,3,padding=1)

    def forward(self, input_S, input_C):
        x1 = F.leaky_relu(self.conv1(input_S))
        x2 = F.leaky_relu(self.conv2(input_S))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv3(input_S))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv4(x4))
        x2 = F.leaky_relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x4 = torch.cat((input_C,x4),1)
        x1 = F.leaky_relu(self.conv7(x4))
        x2 = F.leaky_relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv10(x4))
        x2 = F.leaky_relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv13(x4))
        x2 = F.leaky_relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv16(x4))
        x2 = F.leaky_relu(self.conv17(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv18(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv19(x4))
        x2 = F.leaky_relu(self.conv20(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv21(x4))
        x4 = torch.cat((x1,x2,x3),1)
        return torch.tanh(self.conv22(x4))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,50,3,padding=1)
        self.conv2 = nn.Conv2d(3,10,4,padding=1)
        self.conv3 = nn.Conv2d(3,5,5,padding=2)
        self.conv4 = ResnetBlock(65,50)
        self.conv5 = nn.Conv2d(65,10,4,padding=1)
        self.conv6 = nn.Conv2d(65,5,5,padding=2)
        self.conv7 = nn.Conv2d(65,50,3,padding=1)
        self.conv8 = nn.Conv2d(65,10,4,padding=1)
        self.conv9 = nn.Conv2d(65,5,5,padding=2)
        self.conv10 = ResnetBlock(65,50)
        self.conv11 = nn.Conv2d(65,10,4,padding=1)
        self.conv12 = nn.Conv2d(65,5,5,padding=2)
        self.conv13 = nn.Conv2d(65,50,3,padding=1)
        self.conv14 = nn.Conv2d(65,10,4,padding=1)
        self.conv15 = nn.Conv2d(65,5,5,padding=2)
        self.conv16 = nn.Conv2d(65,3,3,padding=1)

    def forward(self,x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv3(x))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv4(x4))
        x2 = F.leaky_relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv7(x4))
        x2 = F.leaky_relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv10(x4))
        x2 = F.leaky_relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1)
        x1 = F.leaky_relu(self.conv13(x4))
        x2 = F.leaky_relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1))
        x3 = F.leaky_relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1)
        return torch.tanh(self.conv16(x4))


class Make_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_S, input_C):
        encoded = self.encoder(input_S, input_C)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Load weights
@st.cache_resource
def load_model():
    model = Make_model().to(device)
    state_dict = torch.load("AEmodel_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()



# Streamlit interface
st.title("Deep Steganography")

col1, col2 = st.columns(2)

with col1:
    uploaded_cover = st.file_uploader("Upload Cover Image", type=["png","jpg","jpeg"], key="cover")
    if uploaded_cover:
        cover_img = Image.open(uploaded_cover).convert("RGB")
        st.image(cover_img, caption="Cover Image", use_container_width=True)

with col2:
    uploaded_secret = st.file_uploader("Upload Secret Image", type=["png","jpg","jpeg"], key="secret")
    if uploaded_secret:
        secret_img = Image.open(uploaded_secret).convert("RGB")
        st.image(secret_img, caption="Secret Image", use_container_width=True)

if uploaded_cover and uploaded_secret:
    if st.button("Encode & Decode"):
        transform_img = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
        cover_tensor = transform_img(cover_img).unsqueeze(0).to(device)
        secret_tensor = transform_img(secret_img).unsqueeze(0).to(device)

        with torch.no_grad():
            encoded_C, decoded_S = model(secret_tensor, cover_tensor)

        def to_np(img):
            img = img.squeeze().cpu().numpy()
            return np.clip(np.transpose(img, (1,2,0)), 0, 1)

        st.subheader("Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.image(to_np(encoded_C), caption="Encoded Cover", use_container_width=True)
        res_col2.image(to_np(decoded_S), caption="Decoded Secret", use_container_width=True)

"""HW1 Q4 example on a bundled sample; replace --image with your own photo.
Requires numpy scipy matplotlib pillow. Results are not the user's homework images.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import cbook
p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--out',type=Path,default=Path('edge_result.png'))
a=p.parse_args()
img=Image.open(a.image if a.image else cbook.get_sample_data('grace_hopper.jpg')).convert('L')
x=np.asarray(img.resize((320,384)),dtype=float)/255
sm=ndi.gaussian_filter(x,1.3,mode='reflect')
pre_x=np.array([[-1,0,1]]*3)/3;pre_y=pre_x.T
px=ndi.correlate(sm,pre_x,mode='reflect');py=ndi.correlate(sm,pre_y,mode='reflect');pre=np.hypot(px,py)
sx=ndi.sobel(sm,axis=1,mode='reflect')/4;sy=ndi.sobel(sm,axis=0,mode='reflect')/4;sob=np.hypot(sx,sy)
# Canny includes directional NMS and hysteresis; matplotlib does not expose it.
# If OpenCV is installed, use its calibrated implementation; otherwise indicate its absence.
try:
 import cv2
 u8=np.asarray(img.resize((320,384)),dtype=np.uint8)
 can=cv2.Canny(u8,60,140,L2gradient=True).astype(float)/255
 label='Canny (60,140)'
except ImportError:
 raise SystemExit('Install opencv-python to run the Canny panel; see MATLAB alternative in the guide.')
fig,axs=plt.subplots(1,4,figsize=(12,3))
for ax,v,name in zip(axs,[x,pre,sob,can],['Input','Prewitt magnitude','Sobel magnitude',label]):
 ax.imshow(v,cmap='gray');ax.set_title(name);ax.axis('off')
fig.tight_layout();fig.savefig(a.out,dpi=180);print(a.out)

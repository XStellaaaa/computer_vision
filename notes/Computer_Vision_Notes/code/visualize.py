"""Rebuild course diagrams. Requires numpy, scipy, matplotlib, pillow."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import cbook
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 130})

def save(name):
    plt.tight_layout(); plt.savefig(OUT / name, bbox_inches='tight'); plt.close()

def show(ax, data, title, cmap='gray'):
    ax.imshow(data, cmap=cmap, vmin=0 if cmap=='gray' else None,
              vmax=255 if cmap=='gray' else None)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

# Matrix is a pixel image, not an abstract image placeholder.
I=np.array([[0,0,32,32],[0,64,128,192],[32,128,224,255],[0,64,128,255]])
fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(I,cmap='gray',vmin=0,vmax=255)
ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('column c →');ax.set_ylabel('row r ↓')
for (r,c),val in np.ndenumerate(I):ax.text(c,r,str(val),ha='center',va='center',color='white' if val<110 else 'black')
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='tab:red',lw=1);save('pixel_grid.png')

A=np.arange(1,26).reshape(5,5); K=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
fig,axes=plt.subplots(1,3,figsize=(10,3))
for c,ax in enumerate(axes,1):
    ax.imshow(A,cmap='Blues');ax.set_title(f'output (r=2, c={c}): {np.sum(A[1:4,c-1:c+2]*K)}')
    ax.set_xticks(range(5));ax.set_yticks(range(5));ax.grid(color='white')
    for (r,j),v in np.ndenumerate(A):ax.text(j,r,str(v),ha='center',va='center',fontsize=7)
    ax.add_patch(plt.Rectangle((c-1.5,.5),3,3,fill=False,ec='red',lw=3))
save('kernel_steps.png')

# Matplotlib-bundled historical public sample portrait; synthetic noise is reproducible.
photo=np.asarray(Image.open(cbook.get_sample_data('grace_hopper.jpg')).convert('L').resize((320,384)),dtype=float)
rng=np.random.default_rng(20260924); noisy=photo.copy();mask=rng.random(photo.shape)
noisy[mask<.02]=0;noisy[(mask>=.02)&(mask<.04)]=255
mean=ndi.uniform_filter(noisy,size=5,mode='reflect');median=ndi.median_filter(noisy,size=5,mode='reflect')
gauss=ndi.gaussian_filter(noisy,sigma=1.5,mode='reflect');sharp=np.clip(photo+1.2*(photo-ndi.gaussian_filter(photo,1.5)),0,255)
fig,axes=plt.subplots(1,5,figsize=(12,4));
for ax,im,title in zip(axes,[noisy,mean,median,gauss,sharp],['salt & pepper input','mean 5x5','median 5x5','Gaussian sigma=1.5','sharpen clean image']):show(ax,im,title)
save('filter_comparison.png')

# Sobel then a didactic directional NMS and two thresholds.
base=np.zeros((100,145));base[20:75,25:85]=150;base[38:90,58:120]+=75
noisy2=np.clip(base+rng.normal(0,10,base.shape),0,255)
smooth=ndi.gaussian_filter(noisy2,1.4);sx=ndi.sobel(smooth,axis=1)/8;sy=ndi.sobel(smooth,axis=0)/8;mag=np.hypot(sx,sy)
ang=np.mod(np.rad2deg(np.arctan2(sy,sx)),180)
maxima=np.zeros_like(mag,dtype=bool)
for j in range(1,mag.shape[0]-1):
 for i in range(1,mag.shape[1]-1):
  a=ang[j,i]
  if a<22.5 or a>=157.5: p,q=mag[j,i-1],mag[j,i+1]
  elif a<67.5: p,q=mag[j-1,i-1],mag[j+1,i+1]
  elif a<112.5:p,q=mag[j-1,i],mag[j+1,i]
  else:p,q=mag[j-1,i+1],mag[j+1,i-1]
  maxima[j,i]=mag[j,i]>=p and mag[j,i]>=q
thin=mag*maxima;strong=thin>18;candidate=thin>8
labels,num=ndi.label(candidate,structure=np.ones((3,3)))
keep=np.zeros_like(candidate)
for k in range(1,num+1):
 if np.any(strong[labels==k]):keep[labels==k]=True
fig,axes=plt.subplots(1,5,figsize=(12,3))
for ax,im,title in zip(axes,[noisy2,smooth,mag,thin,keep],['input','Gaussian','gradient magnitude','directional NMS','connected threshold']):
 ax.imshow(im,cmap='gray');ax.set_title(title);ax.axis('off')
save('edge_pipeline.png')

n=128; xx,yy=np.meshgrid(np.arange(n),np.arange(n));checker=((xx//3+yy//3)%2).astype(float)
direct=checker[::4,::4];pref=ndi.gaussian_filter(checker,1.7)[::4,::4]
fig,axes=plt.subplots(1,3,figsize=(8,3))
for ax,im,title in zip(axes,[checker,direct,pref],['original fine checks','direct /4: aliasing','blur then /4']):
 ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title);ax.axis('off')
save('aliasing.png')

M0=np.diag([2,1]);M1=np.diag([50,1]);M2=np.diag([50,40]); t=np.linspace(0,2*np.pi,300)
fig,axes=plt.subplots(1,3,figsize=(9,3))
for ax,m,title in zip(axes,[M0,M1,M2],['flat: both small','edge: one large','corner: both large']):
 # E(u,v)=1 contour; clipped to keep the flat panel legible.
 a,b=np.diag(m);ax.plot(np.cos(t)/np.sqrt(a),np.sin(t)/np.sqrt(b));ax.axhline(0,c='0.8');ax.axvline(0,c='0.8');ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('shift u')
save('harris.png')

fig,ax=plt.subplots(figsize=(8,2.9));ax.set_xlim(0,220);ax.set_ylim(0,55)
ax.add_patch(plt.Rectangle((0,0),100,50,fill=False,ec='black'))
ax.add_patch(plt.Rectangle((120,0),100,50,fill=False,ec='black'))
ax.text(50,53,'image 1',ha='center');ax.text(170,53,'image 2',ha='center')
left_points=[(20,15),(50,38),(80,20)]
right_points=[(152,15),(182,38),(212,20)]
for idx,((x,y),(xx,yy)) in enumerate(zip(left_points,right_points)):
 color=['red','blue','green'][idx];ax.scatter([x,xx],[y,yy],color=color,s=65,zorder=3)
 ax.plot([x,xx],[y,yy],color='forestgreen',alpha=.65,lw=1.8)
ax.plot([50,212],[38,20],color='crimson',ls='--',lw=1.7,label='false candidate')
ax.legend(loc='upper left',fontsize=8);ax.axis('off')
save('matching.png')

# An isolated dark blob produces scale-dependent LoG response.
z=np.arange(-45,46);X,Y=np.meshgrid(z,z);blob=((X*X+Y*Y)<12**2).astype(float)
sigmas=np.linspace(2,18,49)
responses=[abs(s*s*ndi.gaussian_laplace(blob,sigma=s)[45,45]) for s in sigmas]
fig,axes=plt.subplots(1,2,figsize=(7,2.7))
axes[0].imshow(blob,cmap='gray');axes[0].set_title('blob radius = 12 px');axes[0].axis('off')
axes[1].plot(sigmas,responses);axes[1].axvline(sigmas[np.argmax(responses)],color='red',ls='--')
axes[1].set_xlabel('Gaussian scale sigma (px)');axes[1].set_ylabel('|sigma² LoG response|')
axes[1].set_title('scale changes response')
save('scale_blob.png')
fig,ax=plt.subplots(figsize=(9,2));ax.axis('off')
items=['overlap','detect & describe','ratio matches','RANSAC geometry','warp & blend']
for j,item in enumerate(items):
 ax.text(j*.2+.08,.5,item,ha='center',va='center',bbox={'boxstyle':'round','facecolor':'#dbeafe'})
 if j<4:ax.annotate('',xy=(j*.2+.195,.5),xytext=(j*.2+.16,.5),arrowprops={'arrowstyle':'->'})
save('panorama_pipeline.png')
print('Generated',len(list(OUT.glob('*.png'))),'explanatory images under',OUT)

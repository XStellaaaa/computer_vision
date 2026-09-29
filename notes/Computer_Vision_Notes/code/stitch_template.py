"""Learning scaffold for HW2: two originals supplied separately by the class.
Usage: python stitch_template.py left.jpg right.jpg panorama.png
Requires opencv-python and numpy. Inspects correspondences; no course image is bundled.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
left=cv2.imread(str(a.left));right=cv2.imread(str(a.right))
if left is None or right is None:raise SystemExit('Cannot read both images; use accessible image paths.')
# OpenCV loads BGR; both images are uint8 HxWx3.
g1=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);g2=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create();kp1,d1=sift.detectAndCompute(g1,None);kp2,d2=sift.detectAndCompute(g2,None)
if d1 is None or d2 is None or len(d2)<2:raise SystemExit('Not enough features: inspect overlap, focus, and image texture.')
pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2,d1,k=2) # right -> left
best=[m for m,n in pairs if m.distance < .75*n.distance]
if len(best)<4:raise SystemExit(f'Only {len(best)} candidate matches; need at least 4 non-collinear pairs.')
src=np.float32([kp2[m.queryIdx].pt for m in best]).reshape(-1,1,2)
dst=np.float32([kp1[m.trainIdx].pt for m in best]).reshape(-1,1,2)
H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.0)
if H is None or mask is None or int(mask.sum())<4:raise SystemExit('Homography failed; inspect matched geometry and scene parallax.')
print('keypoints:',len(kp1),len(kp2),'ratio matches:',len(best),'inliers:',int(mask.sum()))
print('H maps right image coordinates to left image coordinates:\n',H)
# Transform both image corner sets to locate the whole canvas; translation avoids negative pixels.
h1,w1=left.shape[:2];h2,w2=right.shape[:2]
c1=np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
c2=np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
all_c=np.vstack([c1,cv2.perspectiveTransform(c2,H)]).reshape(-1,2)
lo=np.floor(all_c.min(0)).astype(int);hi=np.ceil(all_c.max(0)).astype(int)
width,height=(hi-lo).tolist()
if width<=0 or height<=0 or width*height>100_000_000:raise SystemExit('Implausible warp bounds; inspect H and input order.')
T=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1]],float)
# Warp colors and binary support masks separately; black pixels can be real image content.
wright=cv2.warpPerspective(right,T@H,(width,height))
mask_right=cv2.warpPerspective(np.ones((h2,w2),np.uint8),T@H,(width,height),flags=cv2.INTER_NEAREST).astype(bool)
wleft=cv2.warpPerspective(left,T,(width,height))
mask_left=cv2.warpPerspective(np.ones((h1,w1),np.uint8),T,(width,height),flags=cv2.INTER_NEAREST).astype(bool)
# Distance-transform feathering gives each image more weight away from its boundary.
a1=cv2.distanceTransform(mask_left.astype(np.uint8),cv2.DIST_L2,3)
a2=cv2.distanceTransform(mask_right.astype(np.uint8),cv2.DIST_L2,3)
a1=np.where(mask_left,a1+1e-5,0);a2=np.where(mask_right,a2+1e-5,0)
w=a1/(a1+a2+1e-8)
blend=w[...,None]*wleft.astype(float)+(1-w[...,None])*wright.astype(float)
out=np.clip(blend,0,255).astype(np.uint8)
if not cv2.imwrite(str(a.output),out):raise SystemExit('Could not write panorama.')
preview=cv2.drawMatches(right,kp2,left,kp1,best,None,matchesMask=mask.ravel().tolist(),flags=2)
cv2.imwrite(str(a.output.with_name(a.output.stem+'_matches.jpg')),preview)
print('saved',a.output,'and an inlier-match diagnostic')

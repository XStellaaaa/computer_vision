"""Feature-based two-image panorama. Run: python stitch.py picture1.jpg picture2.jpg --out results"""
from pathlib import Path
import argparse, json, platform
import cv2
import numpy as np

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Cannot read {path}')
    return im

def save(path, im):
    suffix = Path(path).suffix
    ok, data = cv2.imencode(suffix, im)
    if not ok: raise ValueError('Image encoding failed')
    data.tofile(str(path))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('image1'); ap.add_argument('image2')
    ap.add_argument('--out', default='results')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cv2.setRNGSeed(20260929); cv2.setNumThreads(1)
    ims=[read(args.image1), read(args.image2)]
    scales=[min(1.,1600/im.shape[1]) for im in ims]
    small=[cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA) for im,s in zip(ims,scales)]
    sift=cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.025)
    kd=[sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None) for im in small]
    k1,d1=kd[0]; k2,d2=kd[1]
    if d1 is None or d2 is None: raise ValueError('No descriptors')
    bf=cv2.BFMatcher(cv2.NORM_L2)
    matches=bf.knnMatch(d2,d1,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<0.75*n.distance]
    if len(good)<4: raise ValueError('Fewer than four matches')
    p2=np.float32([k2[m.queryIdx].pt for m in good]); p1=np.float32([k1[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(p2,p1,cv2.RANSAC,3.0,maxIters=10000,confidence=0.999)
    if H is None: raise ValueError('Homography estimation failed')
    inside=mask.ravel().astype(bool)
    if inside.sum()<10: raise ValueError('Too few inliers')
    pred=cv2.perspectiveTransform(p2[:,None,:],H)[:,0,:]
    errs=np.linalg.norm(pred-p1,axis=1)[inside]
    full=np.linalg.inv(np.diag([scales[0],scales[0],1]))@H@np.diag([scales[1],scales[1],1])
    full/=full[2,2]
    # Save a readable, evenly sampled subset of RANSAC inliers.
    kept=[m for m,b in zip(good,inside) if b]
    shown=[kept[i] for i in np.linspace(0,len(kept)-1,min(70,len(kept)),dtype=int)]
    vis=cv2.drawMatches(small[1],k2,small[0],k1,shown,None,matchColor=(30,220,50),flags=2)
    save(out/'matches.jpg',vis)
    if ims[0].shape!=ims[1].shape: raise ValueError('This shared-focal implementation expects equal image sizes')
    hh,ww=small[0].shape[:2]
    def calibration(f): return np.array([[f,0,ww/2],[0,f,hh/2],[0,0,1.]])
    # For a rotating camera H ~ K R K^-1, so its normalized singular values agree.
    candidates=np.linspace(0.2*ww,2*ww,4000)
    def focal_loss(f):
        K=calibration(f)
        return np.var(np.log(np.linalg.svd(np.linalg.inv(K)@H@K,compute_uv=False)))
    f0=min(candidates,key=focal_loss); K=calibration(f0)
    U,_,Vt=np.linalg.svd(np.linalg.inv(K)@H@K)
    R0=U@np.diag([1.,1.,np.linalg.det(U@Vt)])@Vt
    params=np.r_[cv2.Rodrigues(R0)[0].ravel(),np.log(f0)]
    src=np.c_[p2[inside],np.ones(inside.sum())]; target=p1[inside]
    def residual(v):
        K=calibration(np.exp(v[3])); R=cv2.Rodrigues(v[:3])[0]
        q=(K@R@np.linalg.inv(K)@src.T).T
        return (q[:,:2]/q[:,2,None]-target).ravel()
    damping=1e-3
    for _ in range(80):
        e=residual(params); J=np.empty((e.size,4))
        for j in range(4):
            delta=np.zeros(4);delta[j]=1e-5
            J[:,j]=(residual(params+delta)-residual(params-delta))/(2e-5)
        step=np.linalg.solve(J.T@J+damping*np.eye(4),-J.T@e)
        trial=params+step
        if not np.log(.15*ww)<trial[3]<np.log(3*ww): damping*=10;continue
        if np.sum(residual(trial)**2)<np.sum(e**2):
            params=trial;damping=max(damping/3,1e-8)
            if np.linalg.norm(step)<1e-9:break
        else:damping*=10
    focal=float(np.exp(params[3])); R=cv2.Rodrigues(params[:3])[0]
    camera_err=np.linalg.norm(residual(params).reshape(-1,2),axis=1)
    rotations=cv2.detail.waveCorrect([np.eye(3,dtype=np.float32),R.astype(np.float32)],cv2.detail.WAVE_CORRECT_HORIZ)
    full_focal=focal/scales[0]
    warper=cv2.PyRotationWarper('spherical',full_focal)
    corners=[];warped=[];masks=[]
    for im,rotation in zip(ims,rotations):
        h,w=im.shape[:2];K=np.float32([[full_focal,0,w/2],[0,full_focal,h/2],[0,0,1]])
        corner,imw=warper.warp(im,K,rotation,cv2.INTER_LINEAR,cv2.BORDER_REFLECT)
        _,maskw=warper.warp(np.ones((h,w),np.uint8)*255,K,rotation,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT)
        corners.append(corner);warped.append(imw);masks.append(maskw)
    low=np.min(corners,axis=0);high=np.max([np.array(c)+[im.shape[1],im.shape[0]] for c,im in zip(corners,warped)],axis=0)
    W,HH=map(int,high-low)
    accum=np.zeros((HH,W,3),np.float32);weights=np.zeros((HH,W),np.float32)
    for corner,im,maskw in zip(corners,warped,masks):
        x,y=np.array(corner)-low;h,w=maskw.shape
        distance=cv2.distanceTransform(np.pad(maskw,1),cv2.DIST_L2,3)[1:-1,1:-1]
        accum[y:y+h,x:x+w]+=im*distance[:,:,None]
        weights[y:y+h,x:x+w]+=distance
    panorama=np.clip(accum/np.maximum(weights[:,:,None],1e-6),0,255).astype(np.uint8)
    valid=weights>0
    panorama[~valid]=255
    save(out/'panorama.jpg',panorama)
    # Largest fully covered axis-aligned rectangle in O(HW), using histogram stacks.
    heights=np.zeros(W,dtype=np.int32); best=(0,0,0,0,0)
    for y,row in enumerate(valid):
        heights=np.where(row,heights+1,0)
        stack=[]
        for x in range(W+1):
            height=int(heights[x]) if x<W else 0; left=x
            while stack and stack[-1][1]>height:
                start,old=stack.pop(); area=old*(x-start)
                if area>best[0]: best=(area,start,y-old+1,x,y+1)
                left=start
            if not stack or stack[-1][1]<height: stack.append((left,height))
    _,x0,y0,x1,y1=best
    save(out/'panorama_crop.jpg',panorama[y0:y1,x0:x1])
    metrics={'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__,
      'input_sizes':[[im.shape[1],im.shape[0]] for im in ims], 'work_sizes':[[im.shape[1],im.shape[0]] for im in small],
      'keypoints':[len(k1),len(k2)],'ratio_matches':len(good),'inliers':int(inside.sum()),
      'inlier_ratio':float(inside.mean()),'reprojection_mean_work_px':float(errs.mean()),
      'reprojection_median_work_px':float(np.median(errs)), 'reprojection_rmse_work_px':float(np.sqrt(np.mean(errs**2))),
      'reprojection_max_work_px':float(errs.max()),'H_work_2_to_1':H.tolist(),'H_full_2_to_1':full.tolist(),
      'canvas_size':[W,HH],'crop_xyxy':[x0,y0,x1,y1],'crop_size':[x1-x0,y1-y0],
      'displayed_inliers':len(shown),'focal_initial_work_px':float(f0),'focal_work_px':focal,'focal_full_px':full_focal,
      'camera_reprojection_mean_work_px':float(camera_err.mean()),'camera_reprojection_rmse_work_px':float(np.sqrt(np.mean(camera_err**2))),
      'rotation_2_to_1':R.tolist(),'projection':'spherical','wave_correction':'horizontal'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()

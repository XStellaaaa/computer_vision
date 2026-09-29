# FCAB333 Homework 2 Panorama Stitching

## Run
Python 3.12 is recommended. From this folder:
    python -m pip install -r requirements.txt
    python stitch.py picture1.jpg picture2.jpg --out results

## Files
- stitch.py: complete executable implementation
- picture1.jpg, picture2.jpg: original assignment inputs
- requirements.txt: package versions
- results/: actual execution outputs and metrics.json

## Method
SIFT (8000 features, contrast threshold 0.025), L2 2-NN matching,
0.75 ratio test, homography RANSAC (3 work pixels, 10000 iterations,
confidence 0.999), shared-focal rotating-camera estimation, spherical
warping, horizontal wave correction, distance feathering, maximal valid
rectangle crop. Features are computed at width 1600; warping uses original
images. No manually selected matches or generative filling is used.

The implementation assumes equal-sized images captured with a shared focal
length. It is designed for the supplied image pair, not a universal stitcher.
Close objects can retain parallax or motion ghosts. White pixels outside the
warped masks are blank canvas. The crop trades field of view for no borders.
The homography is used for matching and camera initialization, not for final
planar extrapolation. Reprojection statistics use fitted inliers and are not
independent validation errors. See the report for formulas and discussion.

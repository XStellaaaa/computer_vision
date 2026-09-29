# FCAB333 Computer Vision Homework 2

本仓库保存“基于特征匹配的校园全景图像拼接”作业的源码、输入图片、实际运行结果和书面报告。

## 内容

- `src/stitch.py`：完整的图像拼接程序
- `images/`：两张原始输入图片
- `results/`：特征匹配图、完整全景图、裁剪全景图和运行指标
- `report/计算机视觉作业2_全景图像拼接.docx`：完整作业报告
- `archive/hw2_源码与运行结果.zip`：可直接提交或长期保存的压缩包

## 运行

```powershell
python -m pip install -r requirements.txt
python src/stitch.py images/picture1.jpg images/picture2.jpg --out results
```

程序使用 SIFT 特征、L2 最近邻匹配、0.75 比值检验、RANSAC、球面投影、水平校正和距离羽化融合。详细算法、参数、误差分析和结果讨论见报告。

## 实际运行环境

- Python 3.12.14
- OpenCV 4.12.0
- NumPy 2.2.6

本仓库默认作为课程作业私有存档使用。

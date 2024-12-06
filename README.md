## YOLO v10
### onnx weights
[download](https://github.com/THU-MIG/yolov10/releases)

### installation
```bash
conda install nvidia/label/cuda-12.4.1::cuda
conda install anaconda::cudnn   # 9.1.1
pip install onnxruntime-gpu     # 1.19
pip3 install opencv-python
```
### reference
https://github.com/ibaiGorordo/ONNX-YOLOv10-Object-Detection

## ByteTrack
### useage
```python
tracker = BYTETracker(track_buffer=5, frame_rate=25, match_thresh=0.8)
for frame in video:
  detections = YOLO(frame)
  # [x1, y1, x2, y2, conf, class_id]

  targets = tracker.update(detections)
  # [x1, y1, x2, y2, track_id, class_id, conf]
```

### reference
https://github.com/kadirnar/bytetrack-pip





























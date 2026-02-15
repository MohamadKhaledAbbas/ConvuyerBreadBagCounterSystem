ffmpeg -rtsp_transport tcp \
  -fflags +genpts -use_wallclock_as_timestamps 1 \
  -i "rtsp://admin:a1234567@192.168.2.108:554/cam/realmonitor?channel=1&subtype=0" \
  -c:v copy -an \
  -f segment -segment_time 1800 -reset_timestamps 1 -strftime 1 \
  "data/remote/output_%Y-%m-%d_%H-%M-%S.h264"


rclone sync "/home/$USER/BreadCounting/data/remote" "pcloud:/BreadCounting/remote" --progress


ffmpeg -r 14 -i input.h264 -c:v copy output.mp4


## 2. Start Model Conversion Environment (Docker)

Launch a Docker container with the OpenExplorer AI toolchain, mounting your model directory for conversion tasks.

```bash
docker run -it --rm -v "C:\0001_MyFiles\0016_Projects\0002_ProjectBased\0012_ConvuyerBreadBagCounterSystem\data:/data" openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

- `-it` : Interactive terminal
- `--rm` : Remove the container when it exits
- `-v ...` : Mount local directory to `/data` inside the container
- `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8` : Docker image containing model tools

---

## 3. Convert ONNX Model to BIN Format

Run the model converter script inside the Docker container:

```bash
python3 model_converter/mapper.py --onnx model/yolo_small_classify_v14.onnx --cal-images model_converter/calibration_data_bgr
```

- `model_converter/mapper.py` : Conversion script
- `--onnx model/best_classify.onnx` : Input ONNX model file
- `--cal-images model_converter/Classify_Calibration` : Calibration images directory

---

rsync -avz --progress --filter="merge rsync.rules" /mnt/c/0001_MyFiles/0016_Projects/0002_ProjectBased/0012_ConvuyerBreadBagCounterSystem/* sunrise@rdkboard:/home/sunrise/ConvuyerBreadCounting


# Run mapper.py with RGB input type and calibration images:
python3 model_converter/mapper_nv12.py \
  --onnx model/yolo_small_classify_v15.onnx \
  --cal-images model_converter/calibration_images \
  --save-cache
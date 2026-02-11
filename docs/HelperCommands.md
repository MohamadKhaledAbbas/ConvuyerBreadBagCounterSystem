ffmpeg -rtsp_transport tcp \
  -fflags +genpts -use_wallclock_as_timestamps 1 \
  -i "rtsp://admin:a1234567@192.168.2.108:554/cam/realmonitor?channel=1&subtype=0" \
  -c:v copy -an \
  -f segment -segment_time 1800 -reset_timestamps 1 -strftime 1 \
  "data/remote/output_%Y-%m-%d_%H-%M-%S.mkv"


rclone sync "/home/$USER/BreadCounting/data/remote" "pcloud:/BreadCounting/remote" --progress


ffmpeg -r 14 -i input.h264 -c:v copy output.mp4


## 2. Start Model Conversion Environment (Docker)

Launch a Docker container with the OpenExplorer AI toolchain, mounting your model directory for conversion tasks.

```bash
docker run -it --rm -v "C:\Users\Khaled\PyCharmMiscProject\data:/data" openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

- `-it` : Interactive terminal
- `--rm` : Remove the container when it exits
- `-v ...` : Mount local directory to `/data` inside the container
- `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8` : Docker image containing model tools

---

## 3. Convert ONNX Model to BIN Format

Run the model converter script inside the Docker container:

```bash
python3 model_converter/mapper.py --onnx model/best_classify.onnx --cal-images model_converter/Classify_Calibration
```

- `model_converter/mapper.py` : Conversion script
- `--onnx model/best_classify.onnx` : Input ONNX model file
- `--cal-images model_converter/Classify_Calibration` : Calibration images directory

---

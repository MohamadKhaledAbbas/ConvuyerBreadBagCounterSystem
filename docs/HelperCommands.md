ffmpeg -rtsp_transport tcp \
  -fflags +genpts -use_wallclock_as_timestamps 1 \
  -i "rtsp://admin:a1234567@192.168.2.108:554/cam/realmonitor?channel=1&subtype=0" \
  -c:v copy -an \
  -f segment -segment_time 1800 -reset_timestamps 1 -strftime 1 \
  "data/remote/output_%Y-%m-%d_%H-%M-%S.mkv"


rclone sync "/home/$USER/BreadCounting/data/remote" "pcloud:/BreadCounting/remote" --progress


ffmpeg -r 14 -i input.h264 -c:v copy output.mp4
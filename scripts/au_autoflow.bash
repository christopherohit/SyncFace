# Loop through your videos
for name in Macron May Obama1; do
  echo "=================================================="
  echo "PROCESSING: $name (Single-Threaded DGX Fix)"
  echo "=================================================="
  
  WORK_DIR="/raid/nobackup/llm_voice/data_syncFace/pretrain/$name"
  
  # 1. EXTRACT FRAMES
  mkdir -p "$WORK_DIR/temp_frames"
  docker run --rm \
    --user $(id -u):$(id -g) \
    -v "$WORK_DIR":/data \
    jrottenberg/ffmpeg:4.1-alpine \
    -i "/data/$name.mp4" -q:v 2 "/data/temp_frames/%06d.jpg"

  # 2. RUN OPENFACE (WITH THREAD LIMITERS)
  # Added -e flags to prevent the old software from spawning too many threads on the DGX
  docker run --rm \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e VECLIB_MAXIMUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -v "$WORK_DIR":/data \
    -w /home/openface-build/build/bin \
    --entrypoint /bin/bash \
    algebr/openface -c \
    "./FeatureExtraction -fdir /data/temp_frames -out_dir /data -of $name ; chown -R $(id -u):$(id -g) /data"

  # 3. CLEAN UP
  if [ -f "$WORK_DIR/$name.csv" ]; then
      mv "$WORK_DIR/$name.csv" "$WORK_DIR/au.csv"
      rm -rf "$WORK_DIR/temp_frames"
      echo "SUCCESS: Created au.csv for $name"
  else
      echo "FAILURE: OpenFace could not create the output file."
  fi
done
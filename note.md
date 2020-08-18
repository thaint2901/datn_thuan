# Install docker

docker pull tensorflow/serving

docker run -p 8500:8500 \
    -v "/mnt/sda1/External Drive/datn_thuan/saved_model:/models/VehicleDetector" \
    -e MODEL_NAME=VehicleDetector \
    tensorflow/serving &
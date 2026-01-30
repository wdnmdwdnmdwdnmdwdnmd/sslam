apt-install eigen version 3.3
source-build eigen version 3.5 fail
run example:  ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml ./dataset/rgbd_dataset_freiburg1_xyz

ffmpeg -i video.mp4 -r 30 %06d.png
ls *.png | sort | awk '{printf "%.6f %s\n",(NR-1)/30,$1}' > rgb.txt
sed -i '1i# timestamp filename\n# example\n# timestamp filename' rgb.txt




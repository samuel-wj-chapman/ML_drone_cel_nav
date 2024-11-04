#docker run -it --gpus all -v $(pwd):/app -v /home/sam/code/personal/drone/data/image_train_val:/dataset/train -v /home/sam/code/personal/drone/data/image_test:/dataset/test -v $(pwd)/clearml.conf:/root/clearml.conf pytorch_celnav /bin/bash


docker run -it --gpus all -v $(pwd):/app -v /home/sam/code/personal/drone/data/image_train_val2:/dataset/train -v /home/sam/code/personal/drone/data/image_test2:/dataset/test -v $(pwd)/clearml.conf:/root/clearml.conf pytorch_celnav /bin/bash
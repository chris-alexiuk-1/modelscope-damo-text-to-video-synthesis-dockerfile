# Run Damo's text-to-video-synthesis locally

Based off of the amazing work found [here](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) and adapted from the HuggingFace space implementation [here](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis) - this repo serves to allow users to quickly test text-to-video-synthesis.

You will need a RTX 3090 or 4090 to run this model on your local machine as the model is fairly large.

## Instructions:

1. Clone this repo using: 

`git clone https://github.com/chris-alexiuk/modelscope-damo-text-to-video-synthesis-dockerfile.git`


2. Create the image with 

`docker build -t damo-txt2vid .`


3. Run the container with 

`docker run --gpus all -v ${HOME}/.cache:/root/.cache -p 7860:7860 --shm-size 30g -it --rm txt2vid`

:tada: That's it, enjoy! :tada:


# FFGO: First Frame is the Place to Go For Video Content Custimization

**Official repository for the,  "First Frame is the Place to Go For Video Content Custimization"**

\[[Website](http://firstframego.github.io)\] 
\[[Paper](https://arxiv.org/abs/2511.15700)\] 



![teaser.gif](./asset/git.gif)


**🤗 Lora Adapters on Huggingface:**  
- [FFGO-Lora-Adapter](https://huggingface.co/Video-Customization/FFGO-Lora-Adapter)


#### Coming
- Training data details

### Setup
- Create Environment
```
conda create -n ffgo python=3.11
conda activate ffgo
```

- Clone Repository and Setup
```
git clone https://github.com/zli12321/FFGO-Video-Customization.git
cd FFGO-Video-Customization
bash setup.sh
```


### Test data
- Test data is available in [Data](https://github.com/zli12321/FFGO-Video-Customization/tree/main/Data/combined_first_frames) folder. All test data involving personal portrait rights has been removed. [0-data.csv](https://github.com/zli12321/FFGO-Video-Customization/blob/main/Data/combined_first_frames/0-data.csv) has the input image path and the caption to generate the video.
- Test data materials are available in [data_materials](https://github.com/zli12321/FFGO-Video-Customization/tree/main/Data/data_materials) folder. These are materials that can form the final input image for video generations.
- Get your own test data: find any images online and segment out the elements as RGBA layer, then combine it with a background using our [combine script]().


### Running Inference

- **All video results in the paper are generated at 1280 × 720 resolution with 81 frames, which requires an H200 GPU for inference unless memory-saving techniques are applied. For lower resource usage, 640 × 480 resolution videos can be generated without H200. However outputs at this lower resolution can differ significantly in content from the 1280 × 720 results as we shown in the paper.**

- **We are using H200 (141GB RAM) to run inference. If you are using A100 or H100, the memory saving such as cpu offload features need to be turned on.**

1. Download [Wan2.2-I2V-14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) from huggingface or modelscope and download our Lora adapters. 

```bash
bash download.sh
```

2. Run fun demo video inference

```
bash ./example_single_inference.sh
```

3. Run continuous inference on our example test dataset
```
bash example_inference.sh
```

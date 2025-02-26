
##  0 . Fine tuning restrictions
Fine tuning supports training for image understanding but not image generation

##  1. install ms-swift
use ms-swift Fine tune the Janus-Pro-7B model,
First, install ms-swift
----------------------------------------------
pip install git+https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
--------------------------------------------------------
##  2. Datasets
The dataset format is
{"messages": [{"role": "user", "content": "<image>Does the construction worker in this picture comply with the safety regulations for high-altitude operations？"}, {"role": "assistant", "content": "In the high-altitude work area, people entering the construction site must wear safety helmets, and high-altitude workers should wear safety belts. The other end of the safety belt must be hung higher than the human body, which is called high hanging and low use. The high-altitude workers in the picture did not wear safety belts, which does not meet the safety standards for high-altitude operations."}], "images": ["root/train/train_images/wpd-36.jpg"]}

##  3. Fine tuning
lora Fine tuning
swift sft --model_type deepseek_janus_pro --model  <Janus-Pro-7B model path> --dataset <dataset path> --target_modules all-linear

full Fine tuning
swift sft --model_type deepseek_janus_pro --model  <Janus-Pro-7B model path> --dataset <dataset path> --train_type  full



##  4. swift model export 
Export can merge two previously dispersed models into one model system
swift export  --ckpt_dir <swift model path>

##  5. swift model Service
swift deploy --ckpt_dir <Export or Swift model path>


##  6. swift model Proxy Service
Create an empty uploads directory
fastapi_swift.py

##  7. Client API fastapi_client. py
Submit questions and receive responses to the swift model Proxy Service 
using fastapi_client. py


##  Other1. 
fastap_client.py parameters(seed、top_p、temperature ) are no longer useful, 
but in order to maintain interface reuse, 
they are retained

##  Other2.
If no export is performed, then:
If the Swift model needs to change the directory, the configuration file needs to be changed
Adapterconfig.json Modify 'base_madel_name_or_path'
Args.json modifies 'model'
Specify the Janus-Pro-7B directory for classical gravity

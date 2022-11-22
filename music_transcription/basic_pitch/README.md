# [OpenVINO] Basic pitch (audio to midi)

## 1. Environment setting
- For installing, follow these instructions

```
conda create -n openvino python=3.9
conda activate openvino
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch <- accroding your version

# install openvino (https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_pip.html)

pip install -r requirements.txt
```

## 2. Tensorflow model to onnx  
- Download basic-pitch [pretrained model](https://github.com/spotify/basic-pitch/tree/main/basic_pitch/saved_models/icassp_2022/nmp)  
- Install [tf2onnx](https://github.com/onnx/tensorflow-onnx)  
- Use following command to get onnx file
    ```
    python -m tf2onnx.convert --saved-model save_model --opset 16 --output model.onnx --inputs input_2:0[-1,43844,1] --rename-inputs input
    ```
  
  (Note that 43844 is calculated from `sample rate * 2 - 256`)  

## 3. Run inference 
  - Just simply run:
    ```
    python openvino_inference.py --input_audio [audio file path] --model_onnx [onnx file path] --save_dictionary [saving result dictionary]
    
    ```
    
## Contact
If you have any question, feel free to contact qaz5517359@gmail.com  

![visitors](https://visitor-badge.glitch.me/badge?page_id=openvino_basic_pitch_github)  
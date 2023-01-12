# [OpenVINO] <br/>CyberAudio  
[![pdf](https://img.shields.io/badge/PDF-Paper-brightgreen)]() 
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1LDmDUT5zwMKbjmsCHKE89hE5jh0sS-8r/view?usp=share_link)  

## 1. Environment setting
- For building environment, follow these instructions

```
(python 3.8 will occur dll missing problems)  
conda create -n openvino python=3.6
conda activate openvino
pip install openvino-dev[tensorflow2,onnx]
pip install -r requirements.txt
```

- For installing
```
git clone https://github.com/FanChiMao/Competition-2022-OpenVINO-Devcup.git
```

## 2. Run inference 
  - Make sure you have downloaded the pretrained onnx models in [**here**](https://github.com/FanChiMao/Competition-2022-OpenVINO-IntelDevCUP/tree/all_process/music_source_separation/umx_openvino/models)  
  
  - Just simply run:
    ```
    python run_convert.py --input_audio [audio file path] --type [separation type] --result_dir [saving result dictionary]
    ```
  
    For example:  
    ```
    python openvino_inference.py --input_audio ./sample_audio/Faded.wav --type vocals --result_dir ./result
    ```

## 3. Example transcription results
  TODO: 

## 4. Reference  
- OpenVINO: https://github.com/openvinotoolkit/openvino  
- tf2onnx: https://github.com/onnx/tensorflow-onnx
- basic pitch: https://github.com/spotify/basic-pitch  

## 5. Competition results


## Contact
If you have any question, feel free to contact qaz5517359@gmail.com  

![visitors](https://visitor-badge.glitch.me/badge?page_id=openvino_basic_pitch_github)  

# [OpenVINO] CyberAudio 2022 -- digital audio system  
## Team Members: Kelvin, Harry, Henry, Edward, Joe, [Jonathan](https://github.com/FanChiMao)  
- [**Intel DevCUP**](https://makerpro.cc/intel-devcup/)  
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1LDmDUT5zwMKbjmsCHKE89hE5jh0sS-8r/view?usp=share_link)  

> Abstract : 本案採用 Music Source Separation(MSS) 與 Music Transcription(MT) 兩項技術分離
樂曲音源並生成樂譜，首先在 MSS 技術實作上，我們將樂曲透過短時距傅立葉變換
Short Time Fourier Transform ( 轉換至頻域並獲取完整音源訊號 ，藉由 MSS
模型推論，可獲得人聲 (Vocals)、貝斯 (Bass)、鼓 (Drums)，以及其他背景音源
(Others) 再 經由 iSTFT 轉回時域並獲得分離後的音源輸出；接著於 MT 的技術實作上，
接收音源分離後的音檔後，會先對音源進行常數 Q 轉換 (Constant Q Transform
(CQT))，此轉換方式與常見的傅立葉轉換的主要差異在於頻率軸為對數標度 (log
scale)，且窗口長度會隨著頻率而改變。而後再對齊和諧相關頻率 (harmonically
related frequencies) 取得和諧頻率資訊。最後利用轉換過後的 2D 特徵，經 5 層卷積
神經網路取得 MIDI 音源輸出 。 使用 OpenVINO TM 的技術應用，可在短時間內迅速地
將任意一首歌曲分離出不同的音訊，並生成對應樂譜，提供使用者全方面的音樂學習
體驗。

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

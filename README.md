# DCASE2023 - Task 7 - Eval FAD

The code of this repository is mostly from [google-research/google-research/frechet_audio_distance](https://github.com/google-research/google-research/tree/master/frechet_audio_distance). Please check the detail of this code from the original repository. We made slight modifications to the code to enable it to run on the more recent Tensorflow version and reduce the number of steps required for execution. We don't have any license for this code.

## Set up

* Clone the repository: 

  ```
  git clone https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_eval_fad.git
  ```
* Install python requirements referring to the [original repository](https://github.com/google-research/google-research/tree/master/frechet_audio_distance). We checked the code runs without an error with the TensorFlow version below. 

  ```
  tensorflow==2.11.0
  ```
* Download a VGG model checkpoint file from the [original repository](https://github.com/google-research/google-research/tree/master/frechet_audio_distance) and save it at `./data/vggish_model.ckpt`.
* Put generated audio files and ground truth files following the folder structure below. For this challenge, the number of audio samples in each dir should be 100.

  ```
  .
  ├── ground truth folder
  │   ├── dog_bark
  │   ├── footstep
  │   ├── gunshot
  │   ├── keyboard
  │   ├── moving_motor_vehicle
  │   ├── rain
  │   └── sneeze_cough
  │
  └── generated folder
      ├── dog_bark
      ├── footstep
      ├── gunshot
      ├── keyboard
      ├── moving_motor_vehicle
      ├── rain
      └── sneeze_cough
  ```
  ## Usage
  * You can modify the config by changing the class variables of class FADWrapper. 
    * Please modify `self.audio_dir:Dict[str,str]` for setting the ground truth folder and generated folder. The initial value is set like below.
    ```
    self.audio_dir:Dict[str,str] = {"gt":f"./eval/","generated":f"./synthesized/"}
    ```
    * Please modify `self.output_dir:str` for setting the ground truth folder and generated folder. The initial value is set like below.
    ```
    self.output_dir:str = "./results/baseline"
    ```
  * Run the code below
  ```
  python FADWrapper.py
  ```
  * You can check the result from 
  ```
  self.output_dir/fad.csv
  ```
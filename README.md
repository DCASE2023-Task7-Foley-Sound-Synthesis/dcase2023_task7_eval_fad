# DCASE2023 - Task 7 - Eval FAD

The code of this repository is mostly from [google-research/google-research/frechet_audio_distance](https://github.com/google-research/google-research/tree/master/frechet_audio_distance). Please check the detail of this code from the original repository. We made slight modifications to the code to enable it to run on the more recent Tensorflow version and reduce the number of steps required for execution. We don't have any license for this code.

## Set up

* Clone the repository: 

  ```
  git clone https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_eval_fad.git
  ```
* Install python requirements and download a VGG model checkpoint file by running the command below. This command will install all the packages by pip install. 

  ```
  python Setup.py
  ```
* Put generated audio files following the folder structure below (You should follow the folder names.). For this challenge, the number of audio samples in each dir should be 100.

  ```
  .
  └── generated_audio
      ├── dog_bark
      ├── footstep
      ├── gunshot
      ├── keyboard
      ├── moving_motor_vehicle
      ├── rain
      └── sneeze_cough
  ```
## Usage
* Run the code below
  ```
  python FADWrapper.py
  ```
* You can check the result from 
  ```
  ./result/fad.csv
  ```
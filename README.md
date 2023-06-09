# DCASE2023 - Task 7 - Eval FAD

The code of this repository is largely from [google-research/google-research/frechet_audio_distance](https://github.com/google-research/google-research/tree/master/frechet_audio_distance). Please check out the details of the original repository if needed. We made slight modifications to the code to enable it to run on the more recent Tensorflow version and reduce the number of steps required for execution. We do not claim any license for this code repository.

## Set up

* Clone the repository: 

  ```
  git clone https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_eval_fad.git
  ```
* Install python requirements and download a VGG model checkpoint file by running the command below. This command will install all the packages by **pip install**.
* Tested on Python==3.8.

  ```
  python setup.py
  ```
* Put generated audio files under the following folder structure below (You should follow the folder names of each category). 
* There should be 100 .wav files in each directory. The names of the audio files do not matter. 

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
  python FADWrapper.py --dir='./generated_audio'
  ```
* You can check the result from 
  ```
  ./result/fad.csv
  ```

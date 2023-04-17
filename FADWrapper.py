from typing import Dict
import os
import pickle
import numpy as np
from create_embeddings_main import main
import fad_utils
import pandas as pd

class FADWrapper:
    def __init__(self) -> None:
        self.name_of_sound_list:list = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
        self.audio_dir:Dict[str,str] = {"gt":f"./data/eval/","generated":f"./generated_audio/"}
        self.output_dir:str = "./result"
        self.exist_mean_var_dict:Dict[str,bool] = {"gt":True,"generated":False}
    
    def compute_fad(self):
        result_dict:dict = {"Category":list(),"FAD":list()}
        for name_of_sound in self.name_of_sound_list:
            print(f'calculate fad of {name_of_sound}')
            mu_sigma_dict:dict = {set_type:{'mu':0,'sigma':0}for set_type in self.audio_dir}
            
            for set_type in self.audio_dir:
                
                if not self.exist_mean_var_dict[set_type]:
                    embeddig_output_dir:str = f'{self.output_dir}/{name_of_sound}'
                    os.makedirs(embeddig_output_dir, exist_ok=True)
                    os.system(f'ls --color=never {self.audio_dir[set_type]}/{name_of_sound}/*  > {embeddig_output_dir}/{set_type}_files.cvs')
                    main(input_file_list=f'{embeddig_output_dir}/{set_type}_files.cvs',output_path=f'{embeddig_output_dir}/{set_type}')
                else:
                    embeddig_output_dir:str = f'{self.audio_dir[set_type]}{name_of_sound}'
                with open(f'{embeddig_output_dir}/{set_type}_embedding.pkl', 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
                mu_sigma_dict[set_type]['mu'] = np.array(data['mu'].float_list.value)
                emb_len = np.array(data['embedding_length'].int64_list.value)[0]
                mu_sigma_dict[set_type]['sigma'] = (np.array(data['sigma'].float_list.value)).reshape((emb_len,emb_len))
            
            fad = fad_utils.frechet_distance(mu_sigma_dict["gt"]['mu'], mu_sigma_dict["gt"]['sigma'], mu_sigma_dict["generated"]['mu'], mu_sigma_dict["generated"]['sigma'])
            result_dict["Category"].append(name_of_sound)
            result_dict["FAD"].append(fad)
            pd.DataFrame(result_dict).to_csv(f'{self.output_dir}/fad.csv',index=False)

if __name__ == '__main__':
    fad_wrapper = FADWrapper()
    fad_wrapper.compute_fad()
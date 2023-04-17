from typing import Dict

import os

class Setup:
    def __init__(self) -> None:
        self.package_install_command_dict:Dict[str,str] = {
            'numpy1.22.4':'pip install numpy==1.22.4',
            'absl-py1.4.0':'pip install absl-py==1.4.0',
            'apache-beam2.45.0':'pip install apache-beam==2.45.0',
            'scipy1.10.1':'pip install scipy==1.10.1',
            'tensorflow2.11.0': 'pip install tensorflow==2.11.0',
            'tf-slim1.1.0': 'pip install tf-slim==1.1.0',
            'pandas1.5.2': 'pip install pandas==1.5.2'
        }
        self.vggish_model_checkpoint_url:str = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
    
    def install_all_package(self) -> None:
        for package_name in self.package_install_command_dict:
            print(f'Install {package_name}')
            os.system(self.package_install_command_dict[package_name])
    
    def download_vggish_model_checkpoint(self) -> None:
        print('download vggish_model_checkpoint')
        os.system(f'wget -P {self.vggish_model_checkpoint_url}')
    
    def setup(self):
        self.install_all_package()
        self.download_vggish_model_checkpoint()
        

if __name__ == '__main__':
    print('setup start')
    Setup().setup()

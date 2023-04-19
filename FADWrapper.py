from typing import Dict, List
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import librosa

from create_embeddings_main import main
import fad_utils


class FADWrapper:
    def __init__(
        self,
        ground_truth_audio_samples_dir: str = "./data/eval",
        generated_audio_samples_dir: str = "./generated_audio",
    ) -> None:
        self.name_of_sound_list: list = [
            "dog_bark",
            "footstep",
            "gunshot",
            "keyboard",
            "moving_motor_vehicle",
            "rain",
            "sneeze_cough",
        ]
        self.audio_dir: Dict[str, str] = {
            "gt": ground_truth_audio_samples_dir,
            "generated": generated_audio_samples_dir,
        }
        self.output_dir: str = "./result"
        self.exist_mean_var_dict: Dict[str, bool] = {"gt": True, "generated": False}
        self.sample_rate: int = 22050
        self.number_of_audio_in_each_dir: int = 100
        self.audio_length_sec: float = 4.0

    def compute_fad(self):
        result_dict: dict = {"Category": list(), "FAD": list()}
        for name_of_sound in self.name_of_sound_list:
            print(f"calculate fad of {name_of_sound}")
            mu_sigma_dict: dict = {
                set_type: {"mu": 0, "sigma": 0} for set_type in self.audio_dir
            }

            for set_type in self.audio_dir:
                if not self.exist_mean_var_dict[set_type]:
                    embeddig_output_dir: str = f"{self.output_dir}/{name_of_sound}"
                    files_path_list: list = [
                        f"{self.audio_dir[set_type]}/{name_of_sound}/{file_name}"
                        for file_name in os.listdir(
                            f"{self.audio_dir[set_type]}/{name_of_sound}"
                        )
                        if os.path.splitext(file_name)[-1] == ".wav"
                    ]
                    self.sanity_check(files_path_list)
                    self.write_meta_data(
                        save_dir=embeddig_output_dir,
                        save_file_name=f"{set_type}_files.cvs",
                        file_path_list=files_path_list,
                    )
                    main(
                        input_file_list_path=f"{embeddig_output_dir}/{set_type}_files.cvs",
                        output_path=f"{embeddig_output_dir}/{set_type}",
                    )
                else:
                    embeddig_output_dir: str = (
                        f"{self.audio_dir[set_type]}/{name_of_sound}"
                    )
                with open(
                    f"{embeddig_output_dir}/{set_type}_embedding.pkl", "rb"
                ) as pickle_file:
                    data = pickle.load(pickle_file)
                mu_sigma_dict[set_type]["mu"] = np.array(data["mu"].float_list.value)
                emb_len = np.array(data["embedding_length"].int64_list.value)[0]
                mu_sigma_dict[set_type]["sigma"] = (
                    np.array(data["sigma"].float_list.value)
                ).reshape((emb_len, emb_len))

            fad = fad_utils.frechet_distance(
                mu_sigma_dict["gt"]["mu"],
                mu_sigma_dict["gt"]["sigma"],
                mu_sigma_dict["generated"]["mu"],
                mu_sigma_dict["generated"]["sigma"],
            )
            result_dict["Category"].append(name_of_sound)
            result_dict["FAD"].append(fad)
            pd.DataFrame(result_dict).to_csv(f"{self.output_dir}/fad.csv", index=False)

    def sanity_check(self, file_path_list: List[str]) -> None:
        assert (
            len(file_path_list) == self.number_of_audio_in_each_dir
        ), f"[Error]The number of audio is {len(file_path_list)}. there should be exactly {self.number_of_audio_in_each_dir} wav files in each subfolder."
        for file_path in file_path_list:
            audio_data, sr = librosa.load(file_path, sr=None, mono=False)
            assert (
                sr == self.sample_rate
            ), f"sample rate should be {self.sample_rate}, but found {sr} at {file_path}."
            assert (
                audio_data.ndim == 1
            ), f"audio should be mono, but this file seems not: {file_path}"
            assert (
                len(audio_data) == int(self.sample_rate * self.audio_length_sec)
            ), f"length is expected to be 88200, but found {len(audio_data)}."

    def write_meta_data(
        self, save_dir: str, save_file_name: str, file_path_list: List[str]
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{save_file_name}", "w") as file:
            for i, file_path in enumerate(file_path_list):
                if i != 0:
                    file.write("\n")
                file.write(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./generated_audio")
    args = parser.parse_args()
    fad_wrapper = FADWrapper(generated_audio_samples_dir=args.dir)
    fad_wrapper.compute_fad()

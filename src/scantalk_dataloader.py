import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import sys
import trimesh
sys.path.append('./model/diffusion-net/src')
import diffusion_net


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertices = self.data[index]["vertices"]
        template = self.data[index]["template"]
        mass = self.data[index]["mass"]
        L = self.data[index]["L"]
        evals = self.data[index]["evals"]
        evecs = self.data[index]["evecs"]
        gradX = self.data[index]["gradX"]
        gradY = self.data[index]["gradY"]
        faces = self.data[index]["faces"]
        dataset = self.data[index]["dataset"]
        
        return {"audio": torch.FloatTensor(audio),
                "vertices": torch.FloatTensor(vertices),
                "template": torch.FloatTensor(template),
                "mass": torch.FloatTensor(np.array(mass)).float(),
                "L": L.float(),
                "evals": torch.FloatTensor(np.array(evals)),
                "evecs": torch.FloatTensor(np.array(evecs)),
                "gradX": gradX.float(),
                "gradY": gradY.float(),
                "file_name": file_name,
                "faces": faces.float(),
                "dataset": dataset
               }
        
    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path_voca = args.wav_path_voca
    vertices_path_voca = args.vertices_path_voca
    audio_path_biwi = args.wav_path_biwi
    vertices_path_biwi = args.vertices_path_biwi
    audio_path_multiface = args.wav_path_multiface
    vertices_path_multiface = args.vertices_path_multiface
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    #Read data from vocaset dataset
    reference = trimesh.load(args.template_file_voca, process=False)
    template_tri = reference.faces
    template_file = args.actor_file_voca
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
            
    k=0
    subject_id_list = []
    mass_dict = {}
    L_dict = {}
    evals_dict = {}
    evecs_dict = {}
    gradX_dict = {}
    gradY_dict = {}
    for r, ds, fs in os.walk(audio_path_voca):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = audio_feature
                data[key]["name"] = f
                
                
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["template"] = temp
        

                vertices_path_ = os.path.join(vertices_path_voca, f.replace("wav", "npy"))
                
                if subject_id not in subject_id_list:
                    subject_id_list.append(subject_id)
                    frame, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(torch.tensor(temp), faces=torch.tensor(template_tri), k_eig=128)
                    mass_dict[subject_id] = mass
                    L_dict[subject_id] = L
                    evals_dict[subject_id] = evals
                    evecs_dict[subject_id] = evecs
                    gradX_dict[subject_id] = gradX
                    gradY_dict[subject_id] = gradY
               
                data[key]["mass"] = mass_dict[subject_id]
                data[key]["L"] = L_dict[subject_id]
                data[key]["evals"] = evals_dict[subject_id]
                data[key]["evecs"] = evecs_dict[subject_id]
                data[key]["gradX"] = gradX_dict[subject_id]
                data[key]["gradY"] = gradY_dict[subject_id]
                data[key]["faces"] = torch.tensor(template_tri)
                data[key]["dataset"] = "vocaset"

                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    
                    vertices = np.load(vertices_path_, allow_pickle=True)[::2, :]
                    data[key]["vertices"] = np.reshape(vertices, (vertices.shape[0], 5023, 3))

                        
    #Read data from BIWI dataset
    reference = trimesh.load(args.template_file_biwi, process=False)
    template_tri = reference.faces
        
    k=0
    for r, ds, fs in os.walk(audio_path_biwi):
        for f in tqdm(fs):
            if f.endswith("wav") and f[3] != 'e':
                
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = audio_feature
                data[key]["name"] = f
        
                subject_id = key.split("_")[0]
                temp = trimesh.load(os.path.join(args.actor_file_biwi, subject_id + '.obj'), process=False).vertices
                data[key]["template"] = temp
                    

                vertices_path_ = os.path.join(vertices_path_biwi, f.replace("wav", "npy"))
                
                if subject_id not in subject_id_list:
                    subject_id_list.append(subject_id)
                    frame, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(torch.tensor(temp), faces=torch.tensor(template_tri), k_eig=128)
                    mass_dict[subject_id] = mass
                    L_dict[subject_id] = L
                    evals_dict[subject_id] = evals
                    evecs_dict[subject_id] = evecs
                    gradX_dict[subject_id] = gradX
                    gradY_dict[subject_id] = gradY
               
                data[key]["mass"] = mass_dict[subject_id]
                data[key]["L"] = L_dict[subject_id]
                data[key]["evals"] = evals_dict[subject_id]
                data[key]["evecs"] = evecs_dict[subject_id]
                data[key]["gradX"] = gradX_dict[subject_id]
                data[key]["gradY"] = gradY_dict[subject_id]
                data[key]["faces"] = torch.tensor(template_tri)
                data[key]["dataset"] = "BIWI"
                
                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    data[key]["vertices"] = np.load(vertices_path_, allow_pickle=True)
                
    #Read data from Multiface dataset
    reference = trimesh.load(args.template_file_multiface, process=False)
    template_tri = reference.faces
        
    k=0
    for r, ds, fs in os.walk(audio_path_multiface):
        for f in tqdm(fs):
            if f.endswith("wav"):
                
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = audio_feature
                data[key]["name"] = f
        
                subject_id = key.split("_")[0]
                temp = trimesh.load(os.path.join(args.actor_file_multiface, subject_id + '.ply'), process=False).vertices
                data[key]["template"] = temp

                vertices_path_ = os.path.join(vertices_path_multiface, f.replace("wav", "npy"))
                
                if subject_id not in subject_id_list:
                    subject_id_list.append(subject_id)
                    frame, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(torch.tensor(temp), faces=torch.tensor(template_tri), k_eig=128)
                    mass_dict[subject_id] = mass
                    L_dict[subject_id] = L
                    evals_dict[subject_id] = evals
                    evecs_dict[subject_id] = evecs
                    gradX_dict[subject_id] = gradX
                    gradY_dict[subject_id] = gradY
               
                data[key]["mass"] = mass_dict[subject_id]
                data[key]["L"] = L_dict[subject_id]
                data[key]["evals"] = evals_dict[subject_id]
                data[key]["evecs"] = evecs_dict[subject_id]
                data[key]["gradX"] = gradX_dict[subject_id]
                data[key]["gradY"] = gradY_dict[subject_id]
                data[key]["faces"] = torch.tensor(template_tri)
                data[key]["dataset"] = "multiface"
                
                if not os.path.exists(vertices_path_):
                    del data[key]
                else:
                    data[key]["vertices"] = np.load(vertices_path_, allow_pickle=True)


    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    for k, v in data.items():
        if data[k]["dataset"] == "BIWI" or data[k]["dataset"] == "multiface":
            subject_id = k.split("_")[0]
        if data[k]["dataset"] == "vocaset":
            subject_id = "_".join(k.split("_")[:-1])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True)
    test_data = Dataset(test_data)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    return dataset


if __name__ == "__main__":
    get_dataloaders()

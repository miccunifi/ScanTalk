import os
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from scantalk_dataloader import get_dataloaders
import scipy
import sys
from model.scantalk import ScanTalk
sys.path.append('./model/diffusion-net/src')

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(args):

    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        os.makedirs(os.path.join(args.result_dir, 'vocaset'))
        os.makedirs(os.path.join(args.result_dir, 'BIWI'))
        os.makedirs(os.path.join(args.result_dir, 'multiface'))

    scantalk = ScanTalk(args.in_channels, args.out_channels, args.latent_channels, args.lstm_layers).to(args.device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    scantalk.load_state_dict(checkpoint['autoencoder_state_dict'])
    
    scantalk.eval()

    print("model parameters: ", count_parameters(scantalk))
    
    dataset = get_dataloaders(args)
    
    pbar = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
    for b, sample in pbar:
        audio = sample["audio"].to(device)
        vertices = sample["vertices"].to(device).squeeze(0)
        template = sample["template"].to(device)
        mass = sample["mass"].to(device)
        L = sample["L"].to(device)
        evals = sample["evals"].to(device)
        evecs = sample["evecs"].to(device)
        gradX = sample["gradX"].to(device)
        gradY = sample["gradY"].to(device)
        faces = sample["faces"].to(device)
        dataset_type = sample["dataset"][0]
        vertices_pred = scantalk.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
        vertices = vertices.detach().cpu().numpy()
        vertices_pred = vertices_pred.detach().cpu().numpy()
        np.save(os.path.join(args.result_dir, dataset_type, filename[0][:-4] + ".npy"), vertices_pred)
    
        
def main():
    parser = argparse.ArgumentParser(description='ScatTalk Test')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./results/Scantalk_Masked_Velocity_Cosine_Loss_150.pth.tar")
    parser.add_argument("--result_dir", type=str, default='./predictions')
    parser.add_argument("--actor_file_voca", type=str, default="../Datasets/vocaset/templates.pkl", help='templates')
    parser.add_argument("--actor_file_biwi", type=str, default="../Datasets/Biwi_6/templates", help='templates')
    parser.add_argument("--actor_file_multiface", type=str, default="../Datasets/Multiface/templates", help='templates')
    parser.add_argument("--template_file_voca", type=str, default="../Datasets/vocaset/flame_model/FLAME_sample.ply", help='faces to animate')
    parser.add_argument("--template_file_biwi", type=str, default="../Datasets/Biwi_6/templates/F1.obj", help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str, default="../Datasets/Multiface/templates/20180227.ply", help='faces to animate')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA F1 F2 F3 F4 F5 F6 M1 M2 M3 M4"
                                                              " 20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927 20190529")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA F7 M5 20190828 20190521")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA F8 M6 20181017 20180105")
    parser.add_argument("--wav_path_voca", type=str, default="../Datasets/vocaset/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_voca", type=str, default="../Datasets/vocaset/vertices_npy", help='path of the ground truth')
    parser.add_argument("--wav_path_biwi", type=str, default="../Datasets/Biwi_6/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_biwi", type=str, default="../Datasets/Biwi_6/vertices", help='path of the ground truth')
    parser.add_argument("--wav_path_multiface", type=str, default="../Datasets/Multiface/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_multiface", type=str, default="../Datasets/Multiface/vertices", help='path of the ground truth')

    parser.add_argument("--info", type=str, default="", help='experiment info')
    
    ##Diffusion Net hyperparameters
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    

    args = parser.parse_args()

    test(args)

if __name__ == "__main__":
    main()

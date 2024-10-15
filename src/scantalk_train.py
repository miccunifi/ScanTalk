import os
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import librosa
import random
from transformers import Wav2Vec2Processor
from scantalk_dataloader import get_dataloaders
from model.scantalk import ScanTalk

class Masked_Loss(nn.Module):
    def __init__(self, voca_mask, biwi_mask, multiface_mask):
        super(Masked_Loss, self).__init__()
        self.voca_mask = voca_mask
        self.biwi_mask = biwi_mask
        self.multiface_mask = multiface_mask
        self.mse = nn.MSELoss(reduction='none')


    def forward(self, target, predictions, dataset_type):
        
        rec_loss = torch.mean(self.mse(predictions, target))
        
        if dataset_type == 'vocaset':
            mouth_loss = torch.mean(self.mse(predictions[:, self.voca_mask, :], target[:, self.voca_mask, :]))
        
        if dataset_type == 'BIWI':
            mouth_loss = torch.mean(self.mse(predictions[:, self.biwi_mask, :], target[:, self.biwi_mask, :]))
        
        if dataset_type == 'multiface':
            mouth_loss = torch.mean(self.mse(predictions[:, self.multiface_mask, :], target[:, self.multiface_mask, :]))
            
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))
        
        return rec_loss + mouth_loss + vel_loss 
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):

    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    lip_mask_voca = scipy.io.loadmat('../Datasets/vocaset/FLAME_lips_idx.mat')
    lip_mask_voca = lip_mask_voca['lips_idx'] - 1 
    lip_mask_voca = np.reshape(np.array(lip_mask_voca, dtype=np.int64), (lip_mask_voca.shape[0]))
        
    lip_mask_biwi = np.load('../Datasets/Biwi_6/mouth_indices.npy')
    
    lip_mask_multiface = np.load('../Datasets/Multiface/mouth_indices.npy')

    scantalk = ScanTalk(args.in_channels, args.out_channels, args.latent_channels, args.lstm_layers).to(args.device)

    print("model parameters: ", count_parameters(scantalk))

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")    
    
    dataset = get_dataloaders(args)

    criterion = nn.MSELoss()
    #criterion = Masked_Loss(lip_mask_voca, lip_mask_biwi, lip_mask_multiface)

    optim = torch.optim.Adam(scantalk.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        scantalk.train()
        tloss = 0
        
        pbar_talk = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_talk:
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
            displacements = vertices - template
            vertices_pred, displacements_pred = scantalk.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
            optim.zero_grad()

            loss = criterion(vertices, vertices_pred, displacements, displacements_pred, dataset_type) 
            torch.nn.utils.clip_grad_norm_(scantalk.parameters(), 10.0)
            loss.backward()
            optim.step()
            tloss += loss.item()
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss/(b+1)))
        
        if epoch % 10 == 0:
            scantalk.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_talk:
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
                    displacements = vertices - template
                    vertices_pred, displacements_pred = scantalk.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
                    loss = criterion(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                                    "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss)/(b+1)))
                
                
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': scantalk.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.result_dir, 'Scantalk.pth.tar'))
        
    
        
def main():
    parser = argparse.ArgumentParser(description='ScanTalk: 3D Talking Heads from Unregistered Meshes')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default='./results')
    parser.add_argument("--template_file_voca", type=str, default="../Datasets/vocaset/flame_model/FLAME_sample.ply", help='faces to animate')
    parser.add_argument("--template_file_biwi", type=str, default="../Datasets/Biwi_6/templates/F1.obj", help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str, default="../Datasets/Multiface/templates/20180227.ply", help='faces to animate')
    parser.add_argument("--actor_file_voca", type=str, default="../Datasets/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--actor_file_biwi", type=str, default="../Datasets/Biwi_6/templates", help='faces to animate')
    parser.add_argument("--actor_file_multiface", type=str, default="../Datasets/Multiface/templates", help='faces to animate')
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

    train(args)

if __name__ == "__main__":
    main()

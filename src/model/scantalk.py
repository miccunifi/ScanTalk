import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear  
from hubert.modeling_hubert import HubertModel
import sys
sys.path.append('./model/diffusion-net/src')
import diffusion_net
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ScanTalk(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, lstm_layers):
        super(ScanTalk, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.lstm_layers = lstm_layers

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                          C_out=self.latent_channels,
                                          C_width=self.latent_channels, 
                                          N_block=4, 
                                          outputs_at= 'vertices', 
                                          dropout=False)
        # decoder
        self.decoder = diffusion_net.layers.DiffusionNet(C_in=self.latent_channels*2,
                                          C_out=self.out_channels,
                                          C_width=self.latent_channels, 
                                          N_block=4, 
                                          outputs_at='vertices', 
                                          dropout=False)
        
        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))


        nn.init.constant_(self.decoder.last_lin.weight, 0)
        nn.init.constant_(self.decoder.last_lin.bias, 0)
        
        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=int(latent_channels/2), num_layers=self.lstm_layers, batch_first=True, bidirectional=True)


    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset, hks=None):
        hidden_states = self.audio_encoder(audio, dataset, frame_num=len(vertices)).last_hidden_state
        audio_emb = self.audio_embedding(hidden_states)
        if hks is None:
            actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        else:
            actor_vertices_emb = self.encoder(hks, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        latent, _ = self.lstm(audio_emb)
        combination = torch.cat([actor_vertices_emb.expand((1, latent.shape[1], actor_vertices_emb.shape[1], actor_vertices_emb.shape[2])), latent.unsqueeze(2).expand(1, latent.shape[1], actor_vertices_emb.shape[1], latent.shape[2])], dim=-1)
        
        combination = combination.squeeze(0)    
        mass = mass.expand(latent.shape[1], mass.shape[1])
        L = L.to_dense().expand(latent.shape[1], L.shape[1], L.shape[2])
        evals = evals.expand(latent.shape[1], evals.shape[1])
        evecs = evecs.expand(latent.shape[1], evecs.shape[1], evecs.shape[2])
        gradX = gradX.to_dense().expand(latent.shape[1], gradX.shape[1], gradX.shape[2])
        gradY = gradY.to_dense().expand(latent.shape[1], gradY.shape[1], gradY.shape[2])
        faces = faces.expand(latent.shape[1], faces.shape[1], faces.shape[2])
                
        pred_disp = self.decoder(combination, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        pred_sequence = pred_disp + actor
        
        return pred_sequence
    
    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces, dataset, hks=None):
        hidden_states = self.audio_encoder(audio, dataset).last_hidden_state
        audio_emb = self.audio_embedding(hidden_states)
        if hks is None:
            actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        else:
            actor_vertices_emb = self.encoder(hks, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        latent, _ = self.lstm(audio_emb)
        combination = torch.cat([actor_vertices_emb.expand((1, latent.shape[1], actor_vertices_emb.shape[1], actor_vertices_emb.shape[2])), latent.unsqueeze(2).expand(1, latent.shape[1], actor_vertices_emb.shape[1], latent.shape[2])], dim=-1)
        
        combination = combination.squeeze(0)    
        mass = mass.expand(latent.shape[1], mass.shape[1])
        L = L.to_dense().expand(latent.shape[1], L.shape[1], L.shape[2])
        evals = evals.expand(latent.shape[1], evals.shape[1])
        evecs = evecs.expand(latent.shape[1], evecs.shape[1], evecs.shape[2])
        gradX = gradX.to_dense().expand(latent.shape[1], gradX.shape[1], gradX.shape[2])
        gradY = gradY.to_dense().expand(latent.shape[1], gradY.shape[1], gradY.shape[2])
        faces = faces.expand(latent.shape[1], faces.shape[1], faces.shape[2])
                
        pred_disp = self.decoder(combination, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        pred_sequence = pred_disp + actor
        
        return pred_sequence




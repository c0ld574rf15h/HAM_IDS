import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import read_config

class Attention(nn.Module):
    def __init__(self, of):
        super().__init__()
        self.conf = read_config()['model']

        self.lin_1 = nn.Linear(
            in_features = eval(self.conf['{}GRUHiddenDim'.format(of)]) * 2,
            out_features = eval(self.conf['{}LinearHiddenDim'.format(of)])
        )
        self.lin_2 = nn.Linear(
            in_features = eval(self.conf['{}LinearHiddenDim'.format(of)]),
            out_features = 1
        )

    def forward(self, output):
        hidden = torch.tanh(self.lin_1(output))
        attention_score = self.lin_2(hidden)
        
        return F.softmax(attention_score, dim = 1)


class HAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf = read_config()['model']

        self.byte_embedder = nn.Embedding(256, eval(self.conf['ByteEmbeddingDim']))
        self.byte_biGRU = nn.GRU(
            input_size = eval(self.conf['ByteEmbeddingDim']),
            hidden_size = eval(self.conf['ByteGRUHiddenDim']),
            bidirectional = True
        )

        self.packet_biGRU = nn.GRU(
            input_size = eval(self.conf['PacketEmbeddingDim']),
            hidden_size = eval(self.conf['PacketGRUHiddenDim']),
            bidirectional = True
        )

        self.byte_attn = Attention(of='Byte')
        self.packet_attn = Attention(of='Packet')

        self.final_classification = nn.Linear(
            in_features = eval(self.conf['PacketGRUHiddenDim']) * 2,
            out_features = eval(self.conf['NumClasses'])
        )


    def forward(self, flow):
        num_packets, batch_sz = flow.shape[0], flow.shape[1]
        byte_embeddings = self.byte_embedder(flow)
        packet_embeddings = torch.zeros(
            num_packets,
            batch_sz,
            eval(self.conf['PacketEmbeddingDim'])
        ).to('cuda')

        for idx, byte_embedding in enumerate(byte_embeddings):
            h_0 = torch.zeros((2, batch_sz, eval(self.conf['ByteGRUHiddenDim']))).to('cuda')
            byte_embedding_transposed = byte_embedding.transpose(0, 1) # (sequence len, batch, input dim)
            output, _ = self.byte_biGRU(byte_embedding_transposed, h_0)

            attention_scores = self.byte_attn(output) # (sequence len, batch, 1)
            packet_embeddings[idx, :, :] = (output * attention_scores).sum(dim = 0)

        h_0 = torch.zeros((2, batch_sz, eval(self.conf['PacketGRUHiddenDim']))).to('cuda')
        output, _ = self.packet_biGRU(packet_embeddings, h_0)

        attention_scores = self.packet_attn(output)
        flow_embedding = (output * attention_scores).sum(dim = 1)

        final_output = self.final_classification(flow_embedding)

        return final_output

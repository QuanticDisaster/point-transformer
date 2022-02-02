import torch
import torch.nn as nn

import sys
sys.path.append(r'/home/tidop/Documents/pt_transfo_adapted/point-transformer')
sys.path.append('/home/tidop/anaconda3/envs/pt/lib/python3.7/site-packages/pointops-0.0.0-py3.7-linux-x86_64.egg')

from lib.pointops.functions import pointops
import pointops_cuda

import time
import logging
logger = logging.getLogger('main-logger')




class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            #logger.info("linear layer : " + str(self.linear))
            #logger.info(" x given : " + str(x.shape))
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                backup = x_b.clone()
                try:
                    x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                except:
                    import pdb; pdb.set_trace()
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        #p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        
        if True:
            offset = []
            if pxo.batch is None:
                offset = [pxo.pos.shape[0]]
            else:
                for i in range(1,pxo.batch.shape[0]):
                    if pxo.batch[i] != pxo.batch[i-1]:
                        offset.append(i)
                offset.append( pxo.batch.shape[0])
            p0, x0, o0 = pxo.pos, pxo.x, torch.IntTensor(offset).cuda()
            mini_pts = offset[0]
            for i in range(1, len(offset)):
                if offset[i] - offset[i-1] < mini_pts:
                    mini_pts = offset[i] - offset[i-1]
            #print('mini : ' + str(mini_pts))
        else:
            p0, x0, o0 = pxo
        
        if len(o0) > 2:
            pass#import pdb; pdb.set_trace()
        #logger.info("batch coord :" + str( p0.shape[0]))
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        #import pdb; pdb.set_trace()
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


custom = False
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU
from torch.nn import Identity

import torch_geometric.transforms as T
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool import knn
from torch_geometric.nn.conv import PointTransformerConv


from torch_cluster import knn_graph
from torch_cluster import fps
from torch_scatter import scatter_max
import time


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels, bias=False)
        self.lin_out = Lin(out_channels, out_channels, bias=False)

        self.pos_nn = Seq(
            MLP([3, 3]), 
            Lin(3, out_channels)
        )

        self.attn_nn = Seq(
            BN(out_channels),
            ReLU(),
            MLP([out_channels, out_channels // 8]),
            Lin(out_channels // 8, out_channels) 
            # NB : last should be // 8 according to original code but it does a
            #weird thing at the same time. To be investigated
        )

        self.transformer = PointTransformerConv(
            in_channels,
            out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn
        )
        
        self.bn1 = BN(in_channels)
        self.bn2 = BN(in_channels)
        self.bn3 = BN(in_channels)

    def forward(self, x, pos, edge_index):
        x_skip = x.clone()
        x = self.bn1(self.lin_in(x)).relu()
        x = self.bn2(self.transformer(x, pos, edge_index)).relu()
        x = self.bn3(self.lin_out(x))
        x = (x + x_skip).relu()
        return x

if custom:
    class TransitionDown(torch.nn.Module):
        '''
            Samples the input point cloud by a ratio percentage to reduce
            cardinality and uses an mlp to augment features dimensionnality
        '''
    
        def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
            super().__init__()
            self.k = k
            self.ratio = ratio
            self.mlp = MLP([3+in_channels, out_channels], bias=False)
    
        def forward(self, x, pos, batch):
            # FPS sampling
            id_clusters = fps(pos, ratio=self.ratio, batch=batch)
    
            # compute for each cluster the k nearest points
            sub_batch = batch[id_clusters] if batch is not None else None
    
            # beware of self loop
            sub_pos = pos[id_clusters]
            
            #pos2 = pos.clone()
            #sub_pos2 = sub_pos.clone()
            #batch2 = batch.clone()
            #sub_batch2 = sub_batch.clone()
            
            torch.cuda.synchronize()
            start = time.time()
            id_k_neighbor = knn(pos, sub_pos, k=self.k,
                                batch_x=batch, batch_y=sub_batch)
            torch.cuda.synchronize()
            end = time.time()
            logger.info("k neighbors : " + str(end - start) + " for " + str(pos.shape[0]) + " and " + str(sub_pos.shape[0]) + " points")

            
            #start = time.time()
            #id_k_neighbor_false = knn_graph(pos, k=self.k, batch=batch)
            
            #logger.info("knn_graph neighbors : " + str(time.time() - start) )
            
            
            
            #logger.info("compute k neighbors (GPU) TransitionDown :" + str(time.time() - start) +  " for " + str(pos.shape[0]) + " and " + str(pos[id_clusters].shape[0]) + " points")
            #logger.info(pos.shape)
            
            
            # start = time.time()
            
            # m = pos.shape[0]
            # nsample = 16
            # pointops_cuda.knnquery_cuda(m, nsample, pos.clone(), pos[id_clusters].clone(), torch.IntTensor(1), torch.IntTensor(1), torch.cuda.IntTensor(m, nsample).zero_(), torch.cuda.FloatTensor(m, nsample).zero_())
            # logger.info("compute knn_cuda in TransitionDown :" + str(time.time() - start) +  " for " + str(pos.shape[0]) + " and " + str(pos[id_clusters].shape[0]) + " points")
            # logger.info(pos.shape)
            #raise Exception 
            
            #start = time.time()
            #pointops.queryandgroup(self.k, pos, pos[id_clusters], x[id_clusters], idx=None, offset=torch.IntTensor(1), new_offset=torch.IntTensor(1), use_xyz=True) 
            #logger.info("compute query and group in  TransitionDown :" + str(time.time() - start) +  " for " + str(pos.shape[0]) + " and " + str(pos[id_clusters].shape[0]) + " points")
            
            
            # transformation of features through a simple MLP
            x = self.mlp(torch.cat([pos,x], axis=1))
    
            # Max pool onto each cluster the features from knn in points
            x_out, _ = scatter_max(x[id_k_neighbor[1]],
                                    id_k_neighbor[0],
                                    dim_size=id_clusters.size(0),
                                    dim=0)
    
            # keep only the clusters and their max-pooled features
            sub_pos, out = pos[id_clusters], x_out
            return out, sub_pos, sub_batch
    
    
    def MLP(channels, batch_norm=True, bias=True):
        return Seq(*[
            Seq(
                Lin(channels[i - 1], channels[i], bias=bias),
                BN(channels[i]) if batch_norm else Identity(),
                ReLU()
            )
            for i in range(1, len(channels))
        ])
    
    class TransitionUp(torch.nn.Module):
        '''
            Reduce features dimensionnality and interpolate back to higher
            resolution and cardinality
        '''
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.mlp_sub = MLP([in_channels, out_channels])
            self.mlp = MLP([out_channels, out_channels])
    
        def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
            # transform low-res features and reduce the number of features
            x_sub = self.mlp_sub(x_sub)
    
            # interpolate low-res feats to high-res points
            x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                              batch_x=batch_sub, batch_y=batch)
    
            x = self.mlp(x) + x_interpolated
    
            return x

class TransitionSummit(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp_sub = Seq(Lin(in_channels, in_channels), ReLU())
        self.mlp = MLP([2*in_channels, in_channels])

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.mlp(torch.cat((x, self.mlp_sub(x)),1))
        return x

import logging
logger = logging.getLogger('main-logger')

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )
        
        #blocks = [2,3,4,6,3]
        blocks = [1,2,3,5,2]

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        
        
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(Seq(
                TransitionDown(in_channels=dim_model[i],
                                out_channels=dim_model[i + 1], k=self.k),
                *[TransformerBlock(in_channels=dim_model[i + 1],
                                  out_channels=dim_model[i + 1]) for k in range(blocks[1:][i])])
            )
        
        # summit layers
        #self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)
        self.mlp_summit = TransitionSummit(dim_model[-1])
        
        self.transformer_summit = Seq(*[TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        ) for i in range(1)])
        
        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(Seq(
                TransitionUp(in_channels=dim_model[n - i],
                              out_channels=dim_model[n - i -1]),
                *[TransformerBlock(in_channels=dim_model[n - i -1],
                                  out_channels=dim_model[n - i -1]) for k in range(1)])
            )
            
        # class score computation
        self.mlp_output = Seq( MLP([dim_model[0], dim_model[0]]),
                              Lin(dim_model[0], out_channels))
            
    def forward(self, x, pos, batch=None): #data):#
        #x, pos, batch = data.x, data.pos, data.batch
        #logger.info("batch coord :" + str( pos.shape[0]))
        # add dummy features in case there is none
        #logger.info("\n")
        #if x is None:
        #    x = torch.ones((pos.shape[0], 1)).to(pos.get_device())
        x = pos if x is None else torch.cat((pos, x), 1)


        start = time.time()
        out_x = []
        out_pos = []
        out_batch = []
        edges_index = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)
        #logger.info('first block : ' + str(time.time() - start))
        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)
        edges_index.append(edge_index)

        start = time.time()
        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):
            #logger.info(str(pos.shape))
            if i == 0:
                start = time.time()   
            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            
            if i == 0:
                pass#logger.info("first transition down " + str(time.time() - start))
            #logger.info('Transition down : ' + str(time.time() - start))
            #start = time.time()   
            #logger.info("pos : " + str(pos.shape[0]) + ' batch : ' + str(batch.shape))
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            if i == 0:
                pass#logger.info("first knn_graph " + str(time.time() - start))
            #logger.info('knn_graph : ' + str(time.time() - start))
            #start = time.time()   
            for layer in self.encoders[i][1:]:
                x = layer(x, pos, edge_index)
            if i == 0:
                pass#logger.info("first transformers " + str(time.time() - start))
            #logger.info('transformers : ' + str(time.time() - start) + '\n')
            
            if i == 0:
                pass#logger.info("first encoder " + str(time.time() - start))
            

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
            edges_index.append(edge_index)

        #logger.info('encoders : ' + str(time.time() - start))
        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        for layer in self.transformer_summit:
            x = layer(x, pos, edge_index)
            

        # backbone up : augment cardinality and reduce dimensionnality
        start = time.time()
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                            pos=out_pos[-i - 2],
                                            pos_sub=out_pos[-i - 1],
                                            batch_sub=out_batch[-i - 1],
                                            batch=out_batch[-i - 2])


            edge_index = edges_index[-i - 2]   #knn_graph(out_pos[-i - 2], k=self.k,
                                #    batch=out_batch[-i - 2])

            for layer in self.decoders[i][1:]:
                x = layer(x, out_pos[-i - 2], edge_index)

        #logger.info('decoders : ' + str(time.time() - start))
        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


def pointtransformer_seg_repro(**kwargs):
    if custom:
        model = Net(6, 13, dim_model=[32, 64, 128, 256, 512], k=16)
        #arch = torch.load(r'/home/tidop/Downloads/point_transfo/pt_weights/model_custom.pt')
        #del arch['mlp_output.1.weight']
        #del arch['mlp_output.1.bias']
        #arch.mlp_output.1.weight
        #model.load_state_dict(arch, strict=False)
    else:
        model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model

# Name: src.py
# Description: The tools to to train and validate the model
# Author: Behzad Valipour Sh. <behzad.valipour@outlook.com>
# Date: 04.09.2022

'''
lines (17 sloc)  1.05 KB
MIT License
Copyright (c) 2022 Behzad Valipour Sh.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import rasterio as rio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window,from_bounds
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import Point, box

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics

from skimage import exposure

import numpy as np
import pandas as pd
import os



def extract_to_point(b, rc):
    """
    Extract the value for the points
    """
    extracted_values = [b[coord[0], coord[1]] for coord in rc]
    return extracted_values

def patch_samples(raster, patch_size, number_patches,strata=False):
    
    raster = rasterio.open(raster)
    Px_SIZE = raster.transform[0]
    xmin, ymin, xmax, ymax = raster.bounds
    PATCH_Px = patch_size
    NUM_PATCHES = number_patches
    PATCH_S = PATCH_Px * Px_SIZE
    
    if strata is False:
        X = np.random.rand(NUM_PATCHES) * ((xmax-(PATCH_S) ) - (xmin + (PATCH_S))) + xmin
        Y = np.random.rand(NUM_PATCHES) * ((ymax - (PATCH_S)) - (ymin + (PATCH_S))) + ymin
        PATCH_CENTER = gpd.GeoSeries.from_xy(X,Y)
        rowcol_tuple = np.transpose(raster.index(PATCH_CENTER.x, PATCH_CENTER.y))

        values = extract_to_point(raster.read()[0,...], rowcol_tuple)
        PATCH_GEOM = PATCH_CENTER.apply(lambda p: box(p.x - PATCH_S/2, p.y - PATCH_S/2, p.x + PATCH_S/2, p.y + PATCH_S/2))

        gdf = gpd.GeoDataFrame({'value':values, 'geometry':PATCH_GEOM},crs=raster.crs)
        gdf = gdf[gdf.value > 0].reset_index(drop=True)
    
    else:
        
        raster_arr = raster.read(masked=True)
        categories = np.unique(raster_arr.flatten())
        categories = categories[~categories.mask]
        selected = np.zeros((0, 2)).astype("int")
        
        for cat in categories:

            # get row,col positions for cat strata
            ind = np.transpose(np.nonzero(raster_arr == cat))

            if NUM_PATCHES > ind.shape[0]:
                msg = (
                    "Sample size is greater than number of pixels in " "strata {}"
                ).format(str(ind))

                msg = os.linesep.join([msg, "Sampling using replacement"])
                Warning(msg)

            # random sample
            samples = np.random.uniform(0, ind.shape[0], NUM_PATCHES).astype("int")
            xy = ind[samples, 1:]
            
            selected = np.append(selected, xy, axis=0)
            rowcol_idx = np.column_stack((selected[:,0], selected[:,1]))
            
            
            
        values = extract_to_point(raster.read()[0,...], rowcol_idx)
        
        X,Y = rasterio.transform.xy(transform=raster.transform, rows=selected[:, 0], cols=selected[:, 1])
        PATCH_CENTER = gpd.GeoSeries.from_xy(X,Y)
        PATCH_GEOM = PATCH_CENTER.apply(lambda p: box(p.x - PATCH_S/2, p.y - PATCH_S/2, p.x + PATCH_S/2, p.y + PATCH_S/2))
        
        gdf = gpd.GeoDataFrame({'value':values, 'geometry':PATCH_GEOM},crs=raster.crs)
    
    return gdf


def _stretch_im(arr, str_clip):
    """Stretch an image in numpy ndarray format using a specified clip value.
    Parameters
    ----------
    arr: numpy array
        N-dimensional array in rasterio band order (bands, rows, columns)
    str_clip: int
        The % of clip to apply to the stretch. Default = 2 (2 and 98)
    Returns
    ----------
    arr: numpy array with values stretched to the specified clip %
    """
    s_min = str_clip
    s_max = 100 - str_clip
    arr_rescaled = np.zeros_like(arr)
    for ii, band in enumerate(arr):
        lower, upper = np.nanpercentile(band, (s_min, s_max))
        arr_rescaled[ii] = exposure.rescale_intensity(
            band, in_range=(lower, upper)
        )
    return arr_rescaled.copy()


# import matplotlib.animation as animation
# from celluloid import Camera
# from datetime import datetime, timedelta

# fig = plt.figure(figsize=(10,4))
# camera = Camera(fig)
# plt.title("Animated SWE Over Alaska")

# for i in range(1000):
#     plt.imshow(swe[i,:,:],cmap='gist_rainbow', vmin=-10, vmax=50)
#     plt.text(145, 10, "{}".format(find_day(i)), {'backgroundcolor':"lightgrey"})
#     camera.snap()
    
# animation = camera.animate()



class LoadImageData(Dataset):
    def __init__(self,dataset_path,image_path, transform=None):
        # data loading
        self.dataset = gpd.read_file(dataset_path)
        self.image_path = image_path
        self.n_samples = len(self.dataset)
        
        self.transform = transform
    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        image = rio.open(self.image_path)
        minx,miny,maxx,maxy = self.dataset.loc[idx, 'geometry'].bounds
        window = from_bounds(minx, miny, maxx, maxy, transform=image.transform)
        self.X = np.moveaxis(image.read(window=window ,resampling=0),0,-1)
        
        self.y = self.dataset.loc[idx,'value']
        
        if self.transform is not None:
            transformed = self.transform(image = self.X)
            self.X = transformed["image"]
            self.y = torch.tensor(self.y)
        
        return self.X,self.y
    
    
class PatchWiseClassModel01(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
                                          # input patches: 4*16*16 
                                          nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 16*12*12
                                          
                                          nn.MaxPool2d(kernel_size=2, stride=2), # out size: 16*6*6
            
                                          nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 16*4*4
            
                                          nn.MaxPool2d(kernel_size=2, stride=2), # out size: 16*2*2
            
                                          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True) # out size: 32*1*1
        )
        
        self.linear_layers = nn.Sequential(
                                            nn.Linear(32*1*1, 6),
        )
        
    def forward(self,inPut):
            X = self.cnn_layers(inPut)
            X = X.view(-1,32*1*1)
            estimated  = self.linear_layers(X)
            return estimated
    
    
class PatchWiseClassModel02(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
                                          # input patches: 4*16*16 
                                          nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 16*12*12
                                                      
                                          nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 16*10*10
                                          
                                          #nn.BatchNorm2d(16),
            
                                          nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 16*8*8
            
                                          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 32*6*6
            
                                          #nn.BatchNorm2d(32),

                                          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 32*4*4
            
                                          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 32*2*2
            
                                          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding='valid'),
                                          nn.ReLU(inplace=True), # out size: 32*1*1
            
                                          #nn.BatchNorm2d(32),
        )
        self.linear_layers = nn.Sequential(
                                            nn.Linear(32*1*1, 6),
        )
        
    def forward(self,inPut):
            X = self.cnn_layers(inPut)
            X = X.view(-1,32*1*1)
            estimated  = self.linear_layers(X)
            return estimated
        

class PatchWiseClassModel(pl.LightningModule):
    def __init__(self,learning_rate = 0.01):
        super(PatchWiseClassModel,self).__init__()
        
        
        self.model = PatchWiseClassModel02()
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.Accuracy = torchmetrics.Accuracy()
        self.Accuracy_val = torchmetrics.Accuracy()
        self.Accuracy_test = torchmetrics.Accuracy()
        self.save_hyperparameters()
        
    def forward(self,inPut):
        return self.model(inPut)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        
        pred = self(x)
        loss = self.loss(pred,y)
        acc = self.Accuracy(pred,y.int())
        self.log('Train_Loss', loss, on_epoch=True, on_step=True)
        self.log('Train_ACC_Step',acc)
        return loss
    
    def training_epoch_end(self, outputs):
        self.log('train_ACC_epoch', self.Accuracy.compute(),prog_bar=True)
        self.Accuracy.reset()
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        pred = self(x)
        loss = self.loss(pred,y)
        acc = self.Accuracy_val(pred,y.int())
        self.log('validation_Loss', loss, on_epoch=True, on_step=True)
        self.log('validation_ACC_Step',acc)
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_ACC_epoch', self.Accuracy_val.compute(),prog_bar=True)
        self.Accuracy_val.reset()
        
        
    def test_step(self, batch, batch_idx):
        x,y = batch
        
        pred = self(x)
        loss = self.loss(pred,y)
        acc = self.Accuracy_test(pred,y.int())
        self.log('Test_Loss', loss, on_epoch=True, on_step=True)
        self.log('Test_ACC_Step',acc)
        return loss
    
    def test_epoch_end(self, outputs):
        self.log('test_ACC_epoch', self.Accuracy_test.compute())
        self.Accuracy_test.reset()
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer
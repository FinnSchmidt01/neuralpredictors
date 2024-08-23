from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from scipy.special import digamma
from scipy.optimize import root
from .base import Encoder
from .mean_variance_functions import (
    fitted_zig_mean,
    fitted_zig_variance,
    fitted_zil_mean,
    fitted_zil_variance,
)
from collections import OrderedDict

class PositionalEncoding(nn.Module):
    # Positional Encoding from Attewntion is all you need paper 
    def __init__(self, d_model, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]

class Response_Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim,mice_dim,hidden_gru, output_dim,n_neurons,n_hidden_layers=3,norm="batch",non_linearity = True, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mice_dim = mice_dim
        self.linear = nn.ModuleDict((datakey,nn.Linear(n_neurons[datakey], hidden_dim-mice_dim)) for datakey in n_neurons)
        #self.linear1 = nn.ParameterDict((datakey,nn.Parameter(torch.rand(n_neurons[datakey]))) for datakey in n_neurons)
        #self.linear2= nn.ParameterDict((datakey,nn.Parameter(torch.rand(hidden_dim))) for datakey in n_neurons)
        #self.bias = nn.ParameterDict((datakey,nn.Parameter(torch.rand(hidden_dim))) for datakey in n_neurons)
        #self.linear = nn.Linear(input_dim,hidden_dim-mice_dim)
        if non_linearity:
            self.elu = nn.ELU()
        else:
            self.elu = None
        if norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim-mice_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm([hidden_dim-mice_dim,80])
        else:
            self.norm = None
        self.GRU = nn.GRU(hidden_dim,hidden_gru,num_layers=n_hidden_layers, dropout=0.1, batch_first=True)
        self.final_linear = nn.Linear(hidden_gru,output_dim)

    def forward(self, x, datakey, mice):
        #reshape input
        B, n_neurons, in_features = x.size()
        x_reshaped = x.permute(0, 2, 1)#.reshape(-1, n_neurons)
        
        #if n_neurons < self.input_dim:
            #padding_size = self.input_dim - n_neurons
            # Apply padding on the last dimension
            #pad_layer = nn.ConstantPad3d((0, 0, 0, padding_size, 0, 0), 0)  # (padding_left, padding_right)
            #x = pad_layer(x)
        
        # Apply the MLP, each Mouse has individual linear layer
        #v = self.linear1[datakey].view(-1, 1)  # Reshape v to (output_features, 1)
        # Compute the outer product of w and v
        #W = torch.matmul(self.linear2[datakey].view(-1, 1), v.T)  # Resulting shape (output_features, input_features)
        #W = W.unsqueeze(0)
        # Expand W to match the batch and time dimensions of x
        # x is (B, T, neurons), W needs to be (B, T, hidden, neurons)
        #W_expanded = W.expand(x.size(0), x.size(2), self.hidden_dim, n_neurons)
        #x = torch.einsum('btij,bjt->bti', W_expanded, x).permute(0,2,1) +self.bias[datakey].unsqueeze(0).unsqueeze(2)
        #self.linear[datakey] = self.linear[datakey].to(x.device)
        
        x = self.linear[datakey](x.permute(0,2,1)).permute(0,2,1) #after permutation (B,hidden,time)
        #x = self.batch_norm(x)
        if self.norm: 
            x = self.norm(x)
        if self.elu:
            x = self.elu(x).permute(0,2,1) #after permutiation (B,time,hidden) this form is needed for GRU input
        else:
            x = x.permute(0,2,1)
        mice_expanded = mice.unsqueeze(0).repeat(B, 1, 1) #expand it from (time,hidden) to (batch,time,hidden)
        x = torch.cat((x,mice_expanded), dim=2)

        x = self.GRU(x)[0]
        x = self.final_linear(x)

        return x 

class Transformer_Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim,att_heads,n_neurons, n_hidden_layers=3,max_len = 80, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = OrderedDict((datakey,nn.Linear(n_neurons[datakey], hidden_dim)) for datakey in n_neurons)
        self.pos_enc = PositionalEncoding(d_model = hidden_dim, max_len = max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead = att_heads,
                                                            dim_feedforward= 4*hidden_dim, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_hidden_layers)

    def forward(self, x, datakey):
        #reshape input
        B, n_neurons, in_features = x.size()
        x_reshaped = x.permute(0, 2, 1)
        '''
        if n_neurons < self.input_dim:
            padding_size = self.input_dim - n_neurons
            # Apply padding on the last dimension
            pad_layer = nn.ConstantPad3d((0, 0, 0, padding_size, 0, 0), 0)  # (padding_left, padding_right)
            x = pad_layer(x)
        '''
        # Apply the MLP and positional encoding
        x = self.linear[datakey](x.permute(0,2,1))
        x = self.pos_enc(x)
        x = self.encoder(x)

        return x 


class ZeroInflationEncoderBase(Encoder):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        shifter=None,
        modulator=None,
    ):

        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = offset
        self.zero_thresholds = zero_thresholds

        if not loc_image_dependent:
            if isinstance(zero_thresholds, dict):
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            torch.ones(1, ro.outdims) * np.log(zero_thresholds[data_key]),
                            requires_grad=False,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            elif zero_thresholds is None:
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            (torch.rand(1, ro.outdims) + 1) * np.log(0.1),
                            requires_grad=True,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            else:
                raise ValueError("zero_thresholds should either be of type {data_key: zero_shreshold_value} or None.")

        else:
            if zero_thresholds is not None:
                warn("zero thresholds are set but will be ignored because loc_image_dependent is True")

        if not q_image_dependent:
            self.q = nn.ParameterDict(
                {data_key: nn.Parameter(torch.rand(1, ro.outdims) * 2 - 1) for data_key, ro in self.readout.items()}
            )

    def loc_nl(self, logloc):
        loc = torch.exp(logloc)
        assert not torch.any(loc == 0.0), "loc should not be zero! Because of numerical instability. Check the code!"
        return loc

    def q_nl(self, q):
        return torch.sigmoid(q) * 0.99999 + self.offset

    def forward_base(
        self,
        x,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):

        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()
        ###new part, reshape core output as in sensorium VideoFiringRateEncoder class
        x = torch.transpose(x, 1, 2)
        batch_size = x.shape[0]
        time_points = x.shape[1]
        ###
        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            ###new part, reshape core output as in sensorium VideoFiringRateEncoder class
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape(((-1,) + pupil_center.size()[2:]))
            ###
            shift = self.shifter[data_key](pupil_center, trial_idx)

        ###new part, reshape core output as in sensorium VideoFiringRateEncoder class sgsd   
            x = x.reshape(((-1,) + x.size()[2:]))
        ###
        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"],shift=shift, batch_size = batch_size, time_points = time_points)
        else:
            x = self.readout(x, data_key=data_key, shift=shift,batch_size = batch_size, time_points = time_points)

        # keep batch dimension if only one image was passed
        params = []
        for param in x:
            params.append(param[None, ...] if len(param.shape) == 1 else param)
        x = torch.stack(params)

        if self.modulator:
            x = self.modulator[data_key](x, behavior=behavior)

        readout_out_idx = 0
        if "logloc" in dir(self):
            logloc = getattr(self, "logloc")[data_key].repeat(batch_size*time_points, 1)
        else:
            logloc = x[readout_out_idx]
            readout_out_idx += 1

        if "q" in dir(self):
            q = getattr(self, "q")[data_key].repeat(batch_size*time_points, 1)
        else:
            q = x[readout_out_idx]
            readout_out_idx += 1

        return x, q, logloc, readout_out_idx, batch_size, time_points


class ZIGEncoder(ZeroInflationEncoderBase):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        init_ks=None,
        theta_image_dependent=True,
        k_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        shifter=None,
        modulator=None,
        moment_matching = None,
        mle_fitting = None,
        no_mixture_mean = False,
        latent = False,
        encoder = None,
        decoder = None,
        norm_layer = "batch",
        non_linearity = True
    ):
    #out_predicts: if True outputs mean predictions, otherwise outputs parameters of ZIG distribution
    # moment_matching is a dict that contains mean and variances numpy arrays of all neurons, the keys are the mice
    # If it is not None, it is used to calculate k via moment matching
    # mle fitting computes k via MLE of Gamma distribution solved for k, it contains k values as dict for each mice
    # no_mixture_mean if true the models predicts either the mean of the uniform (if q<0.5) or the mean of the gamma distr. if q>0.5
    # if false it predicts the mean of the mixture (the ZIG) distribution
    #latent is bool decides if latent sapce is added or not to model
    #encoder/decoder is a dict containing inforamtion about Encoder/Decoder for latent space
    # if norma_layer = "batch" the Encoder will use batch normailzation at beginning otherwise it uses layernorm
    # non_linearity: if True nonlinearity is applied after linear layer and normalization at beginning of Encoder
        super().__init__(
            core, readout, zero_thresholds, loc_image_dependent, q_image_dependent, offset, shifter, modulator
        )
        self.sigma = nn.Parameter(torch.ones(1))
        self.moment_matching = moment_matching
        self.mle_fitting = mle_fitting
        self.no_mixture_mean = no_mixture_mean
        self.latent = latent

        if latent:
            n_neurons = OrderedDict((data_key, ro.outdims) for data_key, ro in self.readout.items())
            #define individual parameters that are fed into the latent GRU for each mouse 
            self.mice = nn.ParameterDict(
                        {
                            data_key: nn.Parameter(torch.zeros(80,encoder["mice_dim"]))
                           for data_key, _ in self.readout.items()
                       })

            if "transformer" in encoder:
                self.encoder = Transformer_Encoder(encoder["input_dim"], encoder["hidden_dim"],encoder["att_heads"], 
                                                    n_neurons,n_hidden_layers=encoder["hidden_layers"])
                if decoder: 
                    if "transformer" in decoder:
                        self.encoder = Transformer_Encoder(encoder["hidden_dim"], decoder["hidden_dim"],decoder["att_heads"], 
                                                    n_neurons,n_hidden_layers=decoder["hidden_layers"])
                        out_dim = decoder["hidden_dim"]                          
                    else:
                        self.decoder = nn.GRU(encoder["hidden_dim"],decoder["hidden_dim"],num_layers=decoder["hidden_layers"], dropout=0.1, batch_first=True)
                        out_dim = decoder["hidden_dim"]
                else:
                    self.decoder = None
                    out_dim = encoder["hidden_dim"]

            else:
                self.encoder = Response_Encoder(encoder["input_dim"],encoder["hidden_dim"],encoder["mice_dim"],encoder["hidden_gru"],encoder["output_dim"],
                                                n_neurons,n_hidden_layers=encoder["hidden_layers"],norm=norm_layer, non_linearity = non_linearity)
                self.samples = encoder["n_samples"]
                if decoder:
                    self.decoder = nn.GRU(encoder["output_dim"],decoder["hidden_dim"],num_layers=decoder["hidden_layers"], dropout=0.1, batch_first=True)
                    out_dim = decoder["hidden_dim"]
                else:
                    self.decoder = None
                    out_dim = encoder["output_dim"]
            self.latent_feature = nn.ParameterDict(
                        {
                            data_key+suffix: nn.Parameter(torch.zeros(out_dim, ro.outdims))
                            #data_key+suffix: nn.Parameter(torch.zeros(out_dim, 1))
                            for data_key, ro in self.readout.items()
                            for suffix in ["_q","_theta"]
                        }) #means has shape (batch,time,hidden_dim)
        
        if not theta_image_dependent:
            self.logtheta = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

        if not k_image_dependent:
            if isinstance(init_ks, dict):
                self.logk = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(torch.ones(1, ro.outdims) * init_ks[data_key])
                        for data_key, ro in self.readout.items()
                    }
                )
            elif init_ks is None:
                self.logk = nn.ParameterDict(
                    {data_key: nn.Parameter(torch.ones(1, ro.outdims) * 0.0) for data_key, ro in self.readout.items()}
                )
            else:
                raise ValueError("init_ks should either be of type {data_key: init_k_value} or None.")

        else:
            if init_ks is not None:
                warn("init_ks are set but will be ignored because k_image_dependent is True")
        

    def theta_nl(self, logtheta):
        theta = nn.functional.elu(logtheta) + 1 + self.offset
        return theta

    def k_nl(self, logk):
        if self.zero_thresholds is not None:
            k = nn.functional.elu(logk) + 1 + self.offset
        else:
            k = nn.functional.elu(logk)+1+self.offset
        return k

    def forward(
        self, x, data_key, behavior=None, pupil_center=None, trial_idx=None, shift=None, detach_core=False, out_predicts = True,**kwargs
    ):
    # latent_params is a tuple containing tensor with means, variance factor and number of samples for mc sampling
        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")

        x, q, logloc, readout_out_idx, batch_size, time_points = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior,
            pupil_center=pupil_center,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )


        if "logtheta" in dir(self):
            logtheta = getattr(self, "logtheta")[data_key].repeat(batch_size*time_points, 1)
        else:
            logtheta = x[readout_out_idx]
            readout_out_idx += 1

        if "logk" in dir(self):
            logk = getattr(self, "logk")[data_key].repeat(batch_size*time_points, 1)
        else:
            logk = x[readout_out_idx].mean(dim=0).repeat(batch_size*time_points, 1) #average k over time, it should be constant for each neuron (not time dependent as theta and q)
            readout_out_idx += 1

        if self.latent:
            total_time = kwargs["responses"].shape[2] #total number of time points before core 
            means = self.encoder(kwargs["responses"],data_key, self.mice[data_key][0:total_time,:])
            sigma_squared = self.sigma ** 2
            # chop of the first time steps since they are also choped of by the core 
            means = means[:,-time_points:,:]

            B, time, hidden = means.shape
    
            if not out_predicts:
                # Sample epsilon from a standard normal distribution
                means = means.unsqueeze(-1)
                epsilon = torch.randn(B, time, hidden, self.samples).to(means.device)
                samples = means+ torch.sqrt(sigma_squared) * epsilon  # Broadcasting sigma across all dimensions and samples
                
                if self.decoder:
                    samples = samples.permute(0,3,1,2)
                    samples = samples.reshape(-1,samples.shape[2],samples.shape[3]) #after that it has form (B*n_samples,time,hidden), every sample goes independetly through decoder
                    samples = self.decoder(samples)[0]
                    samples = samples.reshape(B,self.samples,samples.shape[1],samples.shape[2]) #(B,n_samples,time,hidden) after reshape
                    samples = samples.permute(0,2,3,1) #(B,time,hidden,n_samples) after permute 
                    samples = samples.reshape(-1,samples.shape[2],samples.shape[3]) #reshape samples to (B*time,hidden,n_samples)
                else:
                    samples = samples.reshape(-1,samples.shape[2],samples.shape[3]) #reshape samples to (B*time,hidden,n_samples)
                
                # (B*time,hidden,samples) (hidden,neurons) -> (B*time,neurons,samples)
                latent_repr_theta =  torch.einsum('bhi,hj->bji', samples, self.latent_feature[data_key+"_theta"]) #this computes w_z*z and has shape (B*time,n_neurons,samples)
                logtheta = logtheta.unsqueeze(-1) + latent_repr_theta #shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations

                latent_repr_q =  torch.einsum('bhi,hj->bji', samples, self.latent_feature[data_key+"_q"]) #this computes w_z*z and has shape (B*time,n_neurons,samples)
                q = q.unsqueeze(-1) + latent_repr_q #shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations

                #return (logtheta, k, loc, q, means, sigma_squared, self.latent_feature, self.samples)

                
            else:
                if self.decoder:
                    means = self.decoder(means)[0]
                    means = means.reshape(-1,means.shape[2])
                else:
                    means = means.reshape(-1,means.shape[2])
                # (B*time,hidden) (hidden,neurons) -> (B*time,neurons)
                latent_repr_theta =  torch.einsum('bh,hj->bj', means, self.latent_feature[data_key+"_theta"]) #this computes w_z*z and has shape (B*time,n_neurons,samples)
                logtheta = logtheta + latent_repr_theta #shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations

                latent_repr_q =  torch.einsum('bh,hj->bj', means, self.latent_feature[data_key+"_q"]) #this computes w_z*z and has shape (B*time,n_neurons,samples)
                q = q + latent_repr_q #shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations


        if out_predicts:
            theta = self.theta_nl(logtheta)
            loc = self.loc_nl(logloc)
            if self.moment_matching:
                mean = self.moment_matching[data_key+"_mean"]
                variance = self.moment_matching[data_key+"_variance"]
                k = (mean-loc[0,:])**2 / variance
                k = k.repeat(batch_size*time_points, 1)
            elif self.mle_fitting:
                k = self.mle_fitting[data_key+"fitted_k"]
                k = k.repeat(batch_size*time_points, 1)
                # fix loc
                loc = torch.tensor(0.005).repeat(batch_size*time_points, 1).to(theta.device)
            else:
                k = self.k_nl(logk)

            q = self.q_nl(q)
            x = fitted_zig_mean(theta, k, loc, q, no_mixture_mean = self.no_mixture_mean)
            return x.reshape(((batch_size, time_points) + x.size()[1:]))
        else:
            theta = self.theta_nl(logtheta).reshape(((batch_size, time_points) + logtheta.size()[1:]))
            loc = self.loc_nl(logloc).reshape(((batch_size, time_points) + x.size()[2:]))

            if self.moment_matching:
                mean = self.moment_matching[data_key+"_mean"]
                variance = self.moment_matching[data_key+"_variance"]

                #fix loc
                loc = torch.tensor(0.005).repeat(batch_size,time_points, x.shape[2]).to(theta.device)

                k = (mean-loc[0,0,:])**2 / variance
                k = k.repeat(batch_size*time_points, 1).reshape(((batch_size, time_points) + x.size()[2:]))
                
            elif self.mle_fitting:
                k = self.mle_fitting[data_key+"fitted_k"]
                k = k.repeat(batch_size*time_points, 1).reshape(((batch_size, time_points) + x.size()[2:]))
                #fix loc
                loc = torch.tensor(0.005).repeat(batch_size,time_points, x.shape[2]).to(k.device)
            else:
                k = self.k_nl(logk).reshape(((batch_size, time_points) + x.size()[2:]))

            
            q = self.q_nl(q).reshape(((batch_size, time_points) + q.size()[1:]))
            if self.latent:
                return (theta, k, loc, q, means, sigma_squared, self.latent_feature, self.samples)
            else:
                return (theta, k, loc, q)
            

    def predict_mean(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_mean(theta, k, loc, q)

    def predict_variance(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_variance(theta, k, loc, q)


class ZILEncoder(ZeroInflationEncoderBase):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        mu_image_dependent=True,
        sigma2_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-12,
        shifter=None,
        modulator=None,
    ):

        super().__init__(
            core, readout, zero_thresholds, loc_image_dependent, q_image_dependent, offset, shifter, modulator
        )

        if not mu_image_dependent:
            self.mu = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

        if not sigma2_image_dependent:
            self.logsigma2 = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

    def sigma2_nl(self, logsigma2):
        return nn.functional.elu(logsigma2) + 1 + self.offset

    def mu_nl(self, mu):
        return mu

    def forward(
        self, x, data_key, behavior=None, pupil_center=None, trial_idx=None, shift=None, detach_core=False, **kwargs
    ):
        batch_size = x.shape[0]
        x, q, logloc, readout_out_idx = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior,
            pupil_center=pupil_center,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )

        if "logsigma2" in dir(self):
            logsigma2 = getattr(self, "logsigma2")[data_key].repeat(batch_size, 1)
        else:
            logsigma2 = x[readout_out_idx]
            readout_out_idx += 1

        if "mu" in dir(self):
            mu = getattr(self, "mu")[data_key].repeat(batch_size, 1)
        else:
            mu = x[readout_out_idx]
            readout_out_idx += 1

        return (
            self.mu_nl(mu),
            self.sigma2_nl(logsigma2),
            self.loc_nl(logloc),
            self.q_nl(q),
        )

    def predict_mean(self, x, data_key, *args, **kwargs):
        mu, sigma2, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zil_mean(mu, sigma2, loc, q)

    def predict_variance(self, x, data_key, *args, **kwargs):
        mu, sigma2, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zil_variance(mu, sigma2, loc, q)

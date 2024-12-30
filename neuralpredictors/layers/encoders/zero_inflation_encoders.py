from collections import OrderedDict
from warnings import warn

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.optimize import root
from scipy.special import digamma

from .base import Encoder
from .mean_variance_functions import (
    fitted_zig_mean,
    fitted_zig_variance,
    fitted_zil_mean,
    fitted_zil_variance,
)


class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        in_channels = input_size

        # Iterate over the provided layer sizes to build the network
        for out_channels in layer_sizes:
            layers.append(
                nn.Linear(
                    in_channels,
                    out_channels,
                )
            )  # Add a linear layer
            layers.append(nn.ELU())  # Add an ELU activation function
            in_channels = out_channels

        # Convert the list of layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.float())


class AffineTransform(nn.Module):
    def __init__(self, input_dim, positive_scale=True, positive_offset=False):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim

        # Learnable scale parameter
        self.scale = nn.Parameter(torch.randn(input_dim) + torch.ones(input_dim))
        if positive_scale:
            # Use softplus to ensure positivity for scale
            self.softplus = nn.Softplus()
        else:
            self.softplus = None

        # Learnable offset parameter
        self.offset = nn.Parameter(torch.randn(input_dim))
        self.positive_offset = positive_offset

    def forward(self, z):
        # Apply softplus to the scale to ensure positivity if required
        scale = self.softplus(self.scale) if self.softplus else self.scale

        # Ensure the first offset is positive
        if self.positive_offset:
            offset = torch.abs(self.offset)  # Ensure offset is positive
        else:
            offset = self.offset

        # Affine transformation: z' = scale * z + offset
        z_transformed = scale * z + offset

        # Compute the log determinant of the Jacobian: log|det(scale)| = sum(log(scale))
        log_det = torch.log(torch.abs(scale)) - torch.abs(scale)  # derivative is the same across batches and timepoints
        log_det = log_det.view(1, 1, -1)
        log_det = log_det.expand(z.shape[0], z.shape[1], log_det.shape[2])  # Expand to match shape (B, time, neurons)

        return z_transformed, log_det

    def inverse(self, y):
        # Apply inverse of affine transformation: z = (y - offset) / scale
        scale = self.softplus(self.scale) if self.softplus else self.scale
        if self.positive_offset:
            offset = torch.abs(self.offset)  # Ensure offset is positive
        else:
            offset = self.offset

        offset = offset.view(1, 1, -1, 1)
        scale = scale.view(1, 1, -1, 1)
        z = (y - offset) / scale
        return z


class FlowLayer(nn.Module):
    def __init__(self, input_dim):
        super(FlowLayer, self).__init__()
        # First affine transformation with positive scale and positive offset
        self.affine1 = AffineTransform(input_dim, positive_scale=True, positive_offset=True)

        # Other transformations with positive scale only
        self.elu1 = nn.ELU()
        self.affine2 = AffineTransform(input_dim, positive_scale=True)
        self.elu2 = nn.ELU()
        self.affine3 = AffineTransform(input_dim, positive_scale=True)
        self.affine4 = AffineTransform(input_dim, positive_scale=True)
        self.softplus = nn.Softplus(1)
        self.affine5 = AffineTransform(input_dim, positive_scale=True, positive_offset=True)
        self.tanh = nn.Tanh()

    def forward(self, z, zero_mask):
        # return z, torch.zeros_like(z, device=z.device)
        # Set z to zero where zero_mask is True
        log_det_soft = -torch.log(
            torch.sigmoid(self.invert_softplus(z))
        )  # sigmoid is derivative of sofplus f'^-1 = 1/f'(f())
        z = self.invert_softplus(z)

        z, log_det1 = self.affine1(z)

        # z_in_range = torch.logical_and(z >= -1000, z <= 1000)
        # log_det_tanh = torch.where(
        # z_in_range,
        #    -2 * torch.log(torch.cosh(z)),  # Apply if z is in the range [-10, 10]
        #    torch.abs(z) - torch.log(torch.tensor(2.0))  # Apply if z is not in the range
        # ) # if z is large cosh(z) might explode, but log(cosh(z)) = x -log(2) for large z.
        log_det_tanh = -2 * torch.log(torch.cosh(z / 100)) - torch.log(torch.tensor(100))
        z = self.tanh(z / 100)

        z, log_det2 = self.affine2(z)

        total_log_det = log_det_soft + log_det1 + log_det_tanh + log_det2
        return z, total_log_det
        """
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
            
        z, log_det1 = self.affine1(z)

        #log_det_log = -1 * torch.log(torch.abs(z))
        #z = torch.log(z)
        log_det_log = -1/2 * torch.log(torch.abs(z))
        z = torch.sqrt(z)

        z, log_det2 = self.affine2(z)
        
        log_det_elu1 = torch.where(z > 0, torch.zeros_like(z), z)
        z = self.elu1(z)

        z, log_det3 = self.affine3(z)
        
        log_det_elu2 = torch.where(z > 0, torch.zeros_like(z), z)
        z = self.elu2(z)

        z, log_det4 = self.affine4(z)

        #log_det_exp = torch.abs(torch.sum(z))
        #z = torch.exp(z)
        log_det_square = torch.log(torch.abs(2*z))
        z = torch.square(z)

        log_det_soft = torch.log(torch.sigmoid(z)) #sigmoid is derivative of softplus
        z = self.softplus(z) + 0.005 + 10**(-6) # zero threshold is at 0.005, add small error to avoid log(0.005-0.005)= Nan
        

        # Apply the fifth affine transformation
        #z, log_det5 = self.affine5(z)

        # Return the transformed z and the total log-determinant
        #log_det1, log_det2, log_det3, log_det4 = 0, 0, 0, 0
        total_log_det = (
            log_det1 + log_det_elu1 + log_det2 + log_det_elu2 + log_det3 +
            log_det_log + log_det4 + log_det_soft + log_det_square
        )

        # Set z to zero where zero_mask is True
        z = z * (1-zero_mask)
        return z, total_log_det
        """

    def invert_flow(self, y):
        # return y
        y = self.affine2.inverse(y)
        y = self.invert_tanh(y)
        y = self.affine1.inverse(y)
        y = self.softplus(y)

        return y
        """
        # Inverse of softplus
        y = self.invert_softplus(y)

        # Inverse of affine4
        y = self.affine4.inverse(y)

        # Inverse ELU2
        y = self.invert_elu(y)
        
        # Inverse of affine3
        y = self.affine3.inverse(y)

        # Inverse ELU1
        y = self.invert_elu(y)

        # Inverse of affine2
        y = self.affine2.inverse(y)

        # Inverse log transform
        #y = torch.exp(y)
        y = torch.square(y)

        # Inverse of affine1
        y = self.affine1.inverse(y)

        return y
        """

    def invert_tanh(self, y):
        # Inverse of tanh transformation with clamping to ensure numerical stability
        # Clamp values to ensure they are in the valid range of arctanh
        y = torch.clamp(y, min=-0.999, max=0.999)  # Clamp to avoid infinities and numerical issues
        return 0.5 * torch.log((1 + y) / (1 - y))

    def invert_elu(self, y):
        # Inverse of ELU transformation
        finfo = torch.finfo(y.dtype)

        return torch.where(y > 0, y, torch.log((y / self.elu1.alpha + 1).clamp(min=finfo.tiny)))

    def invert_softplus(self, y):
        # Inverse of softplus transformation with clamping to ensure numerical stability
        finfo = torch.finfo(y.dtype)
        y = torch.where(y > 10, y - 0.005, torch.log(torch.expm1(y - 0.005).clamp(min=finfo.tiny)))
        return y


class ResNet(nn.Module):
    def __init__(self, pre_trained=True, out_dim=512):
        super(ResNet, self).__init__()

        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=pre_trained)

        # Modify the first convolutional layer to accept a single input channel (grayscale, for example)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Modify to match your input channels
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        # Replace the fully connected layer with a new one for the specific hidden_dim
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features*6, out_dim*80)
        self.resnet.fc = nn.Linear(320, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # Forward pass through ResNet including the fully connected layer
        B, _, time, hidden_in = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = torch.flatten(x, 1)
        # Calculate the padding required to make it divisible by 80
        pad_size = (80 - (x.shape[1] % 80)) % 80  # Padding required to get to a multiple of 80
        x = F.pad(x, (0, pad_size))  # Pad at the end

        # Reshape into (8, 80, -1) after padding
        x = x.view(B, 80, -1)

        # Pass through the fully connected layer
        x = self.resnet.fc(x)  # linear layerv across channel dim
        x = x.reshape(B, time, self.out_dim)

        return x


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
        return x + self.encoding[: x.size(0), :]


class Response_Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        mice_dim,
        hidden_gru,
        output_dim,
        n_neurons,
        n_hidden_layers=3,
        norm="batch",
        non_linearity=True,
        use_cnn=False,
        cnn_kernel_sizes=None,
        cnn_hidden_channels=None,
        use_resnet=False,
        pre_trained=False,
        residual=False,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mice_dim = mice_dim
        self.use_cnn = use_cnn  # Whether to use CNN or GRU
        self.use_resnet = use_resnet
        self.residual = residual
        self.output_dim = output_dim

        # Default CNN configurations
        self.cnn_kernel_sizes = cnn_kernel_sizes if cnn_kernel_sizes else [11, 5, 5]
        self.cnn_hidden_channels = cnn_hidden_channels if cnn_hidden_channels else [32, 32, 20]

        # Linear layer for the input dimension
        self.linear = nn.ModuleDict(
            (datakey, nn.Linear(n_neurons[datakey], hidden_dim - mice_dim)) for datakey in n_neurons
        )

        if non_linearity:
            self.elu = nn.ELU()
        else:
            self.elu = None
        if norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim - mice_dim)
            self.norm_flex = False
        elif norm == "layer":
            self.norm = nn.LayerNorm([hidden_dim - mice_dim, 80])
            self.norm_flex = False
        elif norm == "layer_flex":
            self.norm = nn.LayerNorm([hidden_dim - mice_dim])
            self.norm_flex = True
        else:
            self.norm = None

        # if behavior_in_encoder:
        # self.behavior = True
        # self.mlp_behavior = nn.ModuleDict((datakey,MLP(behavior_in_encoder["input_size"],behavior_in_encoder["layer_sizes"])) for datakey in n_neurons)
        # output_dim_behavior = behavior_in_encoder["layer_sizes"][-1]
        # else:
        # output_dim_behavior = 0
        # self.behavior = False

        # hidden_dim += output_dim_behavior
        self.GRU = nn.GRU(hidden_dim, hidden_gru, num_layers=n_hidden_layers, dropout=0.1, batch_first=True)

        if use_cnn:
            layers = []
            input_channels = hidden_dim  # Input to CNN comes with 'hidden_dim' channels
            for i in range(len(self.cnn_hidden_channels)):
                layers.append(
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=self.cnn_hidden_channels[i],
                        kernel_size=self.cnn_kernel_sizes[i],
                        padding=self.cnn_kernel_sizes[i] // 2,  # Padding to preserve the sequence length
                    )
                )
                if norm:
                    layers.append(nn.BatchNorm1d(self.cnn_hidden_channels[i]))
                    # layers.append(nn.LayerNorm(self.cnn_hidden_channels[i]))

                if non_linearity:
                    layers.append(nn.ELU())

                # Add residual connection if enabled and not the first layer
                if self.residual and input_channels == self.cnn_hidden_channels[i]:
                    layers.append(nn.Identity())  # Adds a skip connection

                input_channels = self.cnn_hidden_channels[i]

            self.cnn_layers = nn.Sequential(*layers)

        if use_resnet:
            # Load pre-trained ResNet and modify first conv layer for 1D
            self.resnet = ResNet(out_dim=output_dim)
        self.final_linear = nn.Linear(hidden_gru, output_dim)

    def forward(self, x, datakey, mice, behavior=None):
        # B, n_neurons, in_features = x.size()
        # x_reshaped = x.permute(0, 2, 1)  # (Batch, Time, neurons)

        # Apply individual linear layer per mouse
        x = self.linear[datakey](x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, Hidden, Time) output

        # Apply normalization if applicable
        if self.norm:
            if self.norm_flex:
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        # Apply ELU non-linearity
        if self.elu:
            x = self.elu(x).permute(0, 2, 1)  # change to (B, Time, Hidden) for GRU/CNN input
        else:
            x = x.permute(0, 2, 1)

        if self.use_cnn:
            # Apply CNN layers with optional residual connections
            x = x.permute(0, 2, 1)  # permute to (Batch,hidden,time)
            for i, layer in enumerate(self.cnn_layers):
                if isinstance(layer, nn.Conv1d):
                    residual_input = x
                if isinstance(layer, nn.LayerNorm):
                    x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    x = layer(x)
                # Apply residual connection
                if self.residual and isinstance(layer, nn.Conv1d) and residual_input.shape == x.shape:
                    x = x + residual_input
            x = x.permute(0, 2, 1)  # permute back to (Batch,time,hidden)

        elif self.use_resnet:
            x = self.resnet(x.unsqueeze(1))  # Change to (B,1,Time,Hidden)

        else:
            # Apply GRU layers
            # if self.behavior:
            # behavior = self.mlp_behavior[datakey](behavior)
            # x = torch.cat((x,behavior), dim=2)

            x = self.GRU(x)[0]  # (B, Time, Hidden)

        # Apply final linear layer
        if not self.use_resnet:
            x = self.final_linear(x)

        return x


class Response_Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers=2,
        use_cnn=False,
        kernel_sizes=None,
        hidden_channels=None,
    ):
        super(Response_Decoder, self).__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            # Define CNN layers similar to encoder if use_cnn is True
            self.cnn_layers = []
            input_channels = input_dim
            self.kernel_sizes = kernel_sizes if kernel_sizes else [5, 11]
            self.hidden_channels = hidden_channels if hidden_channels else [input_dim, input_dim]

            for i in range(len(self.hidden_channels)):
                self.cnn_layers.append(
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=self.hidden_channels[i],
                        kernel_size=self.kernel_sizes[i],
                        padding=(self.kernel_sizes[i] - 1) // 2,  # Padding to keep dimension same
                    )
                )
                if i < len(self.hidden_channels) - 1:  # don't normalize/ use non-linearity in last layer
                    self.cnn_layers.append(nn.BatchNorm1d(self.hidden_channels[i]))
                    self.cnn_layers.append(nn.ELU())
                input_channels = self.hidden_channels[i]

            self.cnn = nn.Sequential(*self.cnn_layers)
        else:
            # Define GRU layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.1)

    def forward(self, samples):
        if self.use_cnn:
            samples = self.cnn(samples.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            samples, _ = self.gru(samples)

        return samples


class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, att_heads, n_neurons, n_hidden_layers=3, max_len=80, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = OrderedDict((datakey, nn.Linear(n_neurons[datakey], hidden_dim)) for datakey in n_neurons)
        self.pos_enc = PositionalEncoding(d_model=hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=att_heads, dim_feedforward=4 * hidden_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_hidden_layers)

    def forward(self, x, datakey):
        # reshape input
        B, n_neurons, in_features = x.size()
        x_reshaped = x.permute(0, 2, 1)
        """
        if n_neurons < self.input_dim:
            padding_size = self.input_dim - n_neurons
            # Apply padding on the last dimension
            pad_layer = nn.ConstantPad3d((0, 0, 0, padding_size, 0, 0), 0)  # (padding_left, padding_right)
            x = pad_layer(x)
        """
        # Apply the MLP and positional encoding
        x = self.linear[datakey](x.permute(0, 2, 1))
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
            x = self.readout(
                x,
                data_key=data_key,
                sample=kwargs["sample"],
                shift=shift,
                batch_size=batch_size,
                time_points=time_points,
            )
        else:
            x = self.readout(x, data_key=data_key, shift=shift, batch_size=batch_size, time_points=time_points)

        # keep batch dimension if only one image was passed
        params = []
        for param in x:
            params.append(param[None, ...] if len(param.shape) == 1 else param)
        x = torch.stack(params)

        if self.modulator:
            x = self.modulator[data_key](x, behavior=behavior)

        readout_out_idx = 0
        if "logloc" in dir(self):
            logloc = getattr(self, "logloc")[data_key].repeat(batch_size * time_points, 1)
        else:
            logloc = x[readout_out_idx]
            readout_out_idx += 1

        if "q" in dir(self):
            q = getattr(self, "q")[data_key].repeat(batch_size * time_points, 1)
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
        moment_matching=None,
        mle_fitting=None,
        no_mixture_mean=False,
        latent=False,
        encoder=None,
        decoder=None,
        norm_layer="batch",
        non_linearity=True,
        dropout=False,
        dropout_prob=0.5,
        future_prediction=False,
        flow=False,
        position_features=None,
        behavior_in_encoder=None,
    ):
        # out_predicts: if True outputs mean predictions, otherwise outputs parameters of ZIG distribution
        # moment_matching is a dict that contains mean and variances numpy arrays of all neurons, the keys are the mice
        # If it is not None, it is used to calculate k via moment matching
        # mle fitting computes k via MLE of Gamma distribution solved for k, it contains k values as dict for each mice
        # no_mixture_mean if true the models predicts either the mean of the uniform (if q<0.5) or the mean of the gamma distr. if q>0.5
        # if false it predicts the mean of the mixture (the ZIG) distribution
        # latent is bool decides if latent sapce is added or not to model
        # encoder/decoder is a dict containing inforamtion about Encoder/Decoder for latent space
        # if norma_layer = "batch" the Encoder will use batch normailzation at beginning otherwise it uses layernorm
        # non_linearity: if True nonlinearity is applied after linear layer and normalization at beginning of Encoder
        # dropout: applies dropout with dropout_prob probabilitiy at beginning of Encoder if True
        # future_prediction: If true the Encoder predicts q(z_(t+1)|y_t) instead of q(z_t|y_t). Thus, it predicts the future given state for given enruonal responses at time t
        # position_features: If it's a dict, the feature vectors w_q, w_theta which map the latent space to response parameters q, theta are computed with a MLP
        #                    the dict contains information about the MLP which take neurons brain position as input
        # behavior_in_encoder: If it's a dict, contains dict of MLP which processes behavior to add it to responses input in encoder
        super().__init__(
            core, readout, zero_thresholds, loc_image_dependent, q_image_dependent, offset, shifter, modulator
        )
        self.sigma = nn.Parameter(torch.ones(1))
        self.moment_matching = moment_matching
        self.mle_fitting = mle_fitting
        self.no_mixture_mean = no_mixture_mean
        self.latent = latent
        self.future = future_prediction
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.position_features = position_features
        self.behavior_in_encoder = behavior_in_encoder

        if latent:
            n_neurons = OrderedDict((data_key, ro.outdims) for data_key, ro in self.readout.items())
            # self.strong_decoder = nn.ModuleDict((datakey, nn.GRU(2*n_neurons[datakey],n_neurons[datakey], dropout=0.1, batch_first=True)) for datakey in n_neurons)
            # define individual parameters that are fed into the latent GRU for each mouse
            self.mice = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(80, encoder["mice_dim"])) for data_key, _ in self.readout.items()}
            )
            if flow:
                self.flow_base = flow
                self.flow = nn.ModuleDict((datakey, FlowLayer(n_neurons[datakey])) for datakey in n_neurons)
                self.psi = {data_key: torch.ones(n_neurons[data_key]) for data_key, _ in self.readout.items()}
            else:
                self.flow = None

            if behavior_in_encoder:
                self.mlp_behavior = nn.ModuleDict(
                    (datakey, MLP(behavior_in_encoder["input_size"], behavior_in_encoder["layer_sizes"]))
                    for datakey in n_neurons
                )
                self.output_dim_behavior = behavior_in_encoder["layer_sizes"][-1]
            else:
                self.output_dim_behavior = 0
            """
            if "transformer" in encoder:
                self.encoder = Transformer_Encoder(encoder["input_dim"], encoder["hidden_dim"],encoder["att_heads"], 
                                                    n_neurons,n_hidden_layers=encoder["hidden_layers"], use_cnn = encoder_dict["use_cnn"] )
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
            """
            self.encoder = Response_Encoder(
                encoder["input_dim"],
                encoder["hidden_dim"],
                encoder["mice_dim"],
                encoder["hidden_gru"],
                encoder["output_dim"],
                n_neurons,
                n_hidden_layers=encoder["hidden_layers"],
                norm=norm_layer,
                non_linearity=non_linearity,
                use_cnn=encoder["use_cnn"],
                cnn_kernel_sizes=encoder["kernel_size"],
                cnn_hidden_channels=encoder["channel_size"],
                use_resnet=encoder["use_resnet"],
                pre_trained=encoder["pretrained"],
                residual=encoder["residual"],
                behavior_in_encoder=behavior_in_encoder,
            )
            self.samples = encoder["n_samples"]

            if decoder:
                # self.decoder = nn.GRU(encoder["output_dim"],decoder["hidden_dim"],num_layers=decoder["hidden_layers"], dropout=0.1, batch_first=True)
                self.decoder = Response_Decoder(
                    input_dim=encoder["output_dim"] + self.output_dim_behavior,
                    hidden_dim=decoder["hidden_dim"],
                    n_layers=decoder["hidden_layers"],
                    use_cnn=decoder["use_cnn"],  # if Behavior is added to the latent, adapt the latent dim
                    kernel_sizes=decoder["kernel_size"],
                    hidden_channels=decoder["channel_size"],
                )

                out_dim = decoder["hidden_dim"]
            else:
                self.decoder = None
                out_dim = encoder["output_dim"]

            if position_features:
                # self.mlp_q = MLP(position_features["input_size"],position_features["layer_sizes"])
                self.mlp_q = nn.ModuleDict(
                    (datakey, MLP(position_features["input_size"], position_features["layer_sizes"]))
                    for datakey in n_neurons
                )
                # self.mlp_theta = MLP(position_features["input_size"],position_features["layer_sizes"])
                self.mlp_theta = nn.ModuleDict(
                    (datakey, MLP(position_features["input_size"], position_features["layer_sizes"]))
                    for datakey in n_neurons
                )
                self.latent_feature = {}

            else:
                self.latent_feature = nn.ParameterDict(
                    {
                        data_key + suffix: nn.Parameter(torch.randn(out_dim, ro.outdims))
                        # data_key+suffix: nn.Parameter(torch.zeros(out_dim, 1))
                        for data_key, ro in self.readout.items()
                        for suffix in ["_q", "_theta"]
                    }
                )  # means has shape (batch,time,hidden_dim)
        else:  # if no latent
            self.flow = False

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
            k = nn.functional.elu(logk) + 1 + self.offset
        return k

    def forward(
        self,
        x,
        data_key,
        behavior_core=None,
        pupil_center_core=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        out_predicts=True,
        train=False,
        sample_prior=False,
        neuron_mask=None,
        positions=None,
        **kwargs
    ):
        # latent_params is a tuple containing tensor with means, variance factor and number of samples for mc sampling
        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")

        if (
            "behavior" in kwargs.keys()
        ):  # rename behavior, pupil_center keys, which are for latent input, in kwargs to avoid conflicts with core input
            kwargs["behavior_encoder"] = kwargs.pop("behavior")
            kwargs["pupil_center_encoder"] = kwargs.pop("pupil_center")

        x, q, logloc, readout_out_idx, batch_size, time_points = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior_core,
            pupil_center=pupil_center_core,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )

        if "logtheta" in dir(self):
            logtheta = getattr(self, "logtheta")[data_key].repeat(batch_size * time_points, 1)
        else:
            logtheta = x[readout_out_idx]
            readout_out_idx += 1

        if "logk" in dir(self):
            logk = getattr(self, "logk")[data_key].repeat(batch_size * time_points, 1)
        else:
            logk = (
                x[readout_out_idx].mean(dim=0).repeat(batch_size * time_points, 1)
            )  # average k over time, it should be constant for each neuron (not time dependent as theta and q)
            readout_out_idx += 1

        if self.latent:
            total_time = kwargs["responses"].shape[2]  # total number of time points before core
            responses = kwargs["responses"].clone()

            if self.behavior_in_encoder:
                behavior = kwargs["behavior_encoder"].clone().permute(0, 2, 1)  # shape it to (B, time, behav)
                if self.behavior_in_encoder["input_size"] == 4:  # if pupil_center should be included in encoder as well
                    pupil_center = kwargs["pupil_center_encoder"].clone().permute(0, 2, 1)
                    behavior = torch.cat((behavior, pupil_center), dim=2)
                behavior = self.mlp_behavior[data_key](
                    behavior
                )  # apply MLP to scale dim from 2/4 to bigger dim like 10
                behavior = behavior[:, -time_points:]  # chop of the first time points which are not needed
            else:
                behavior = None

            B, number_neurons, total_time = responses.shape
            if self.dropout:
                if train:
                    if self.dropout == "entire_trial":
                        mask = torch.rand(B, number_neurons, 1, device=x.device) <= self.dropout_prob
                        mask = mask.expand(-1, -1, total_time)
                    else:
                        mask = torch.rand(B, number_neurons, total_time, device=x.device) <= self.dropout_prob

                    responses.masked_fill_(mask, 0)
                    responses = (
                        1 / (1 - self.dropout_prob) * responses
                    )  # after dropout non zero values are multiplied with 1/(1-p), p=0.5 here
                    mask = mask[:, :, -time_points:].permute(0, 2, 1)  # bring it in suitable form for output
                else:
                    if neuron_mask:
                        responses[:, neuron_mask, :] = 0
                        responses[:, : number_neurons // 4, :] = 0  # first quarte is always masked for evaluation
                    else:
                        responses[:, : number_neurons // 2, :] = 0

                means = self.encoder(responses, data_key, self.mice[data_key][0:total_time, :], behavior=behavior)
            else:
                means = self.encoder(
                    kwargs["responses"], data_key, self.mice[data_key][0:total_time, :], behavior=behavior
                )

            sigma_squared = self.sigma**2
            # chop of the first time steps since they are also choped of by the core
            means = means[:, -time_points:, :]
            B, time, hidden = means.shape

            if (not out_predicts) or sample_prior:
                # Sample epsilon from a standard normal distribution
                if sample_prior:
                    samples = torch.randn(B, time, hidden, self.samples).to(means.device)
                    # samples = torch.zeros(B,time,hidden,self.samples).to(means.device)

                else:
                    means = means.unsqueeze(-1)
                    epsilon = torch.randn(B, time, hidden, self.samples).to(means.device)
                    samples = (
                        means + torch.sqrt(sigma_squared) * epsilon
                    )  # Broadcasting sigma across all dimensions and samples

                if self.behavior_in_encoder:
                    samples = torch.cat((samples, behavior.unsqueeze(-1).repeat(1, 1, 1, self.samples)), dim=2)

                if self.decoder:
                    samples = samples.permute(0, 3, 1, 2)
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # after that it has form (B*n_samples,time,hidden), every sample goes independetly through decoder
                    samples = self.decoder(samples)  # [0]
                    samples = samples.reshape(
                        B, self.samples, samples.shape[1], samples.shape[2]
                    )  # (B,n_samples,time,hidden) after reshape
                    samples = samples.permute(0, 2, 3, 1)  # (B,time,hidden,n_samples) after permute
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # reshape samples to (B*time,hidden,n_samples)
                else:
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # reshape samples to (B*time,hidden,n_samples)

                # (B*time,hidden,samples) (hidden,neurons) -> (B*time,neurons,samples)
                if self.position_features:
                    latent_feature_theta = self.mlp_theta[data_key](positions).permute(
                        1, 0
                    )  # shape to (latent_dim, n_neurons)
                    latent_feature_q = self.mlp_q[data_key](positions).permute(1, 0)

                    latent_repr_theta = torch.einsum(
                        "bhi,hj->bji", samples, latent_feature_theta
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bhi,hj->bji", samples, latent_feature_q
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)

                else:
                    # theta_feature_norms = torch.norm(self.latent_feature[data_key+"_theta"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_theta"] = self.latent_feature[data_key+"_theta"] / theta_feature_norms

                    # q_feature_norms = torch.norm(self.latent_feature[data_key+"_q"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_q"] = self.latent_feature[data_key+"_q"] / q_feature_norms

                    latent_repr_theta = torch.einsum(
                        "bhi,hj->bji", samples, self.latent_feature[data_key + "_theta"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bhi,hj->bji", samples, self.latent_feature[data_key + "_q"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)

                if self.future:  # shift the latents, such that they predict the proceeding time step
                    # Reshape from (B*time, n_neurons, n_samples) to (B, time, n_neurons, n_samples)
                    latent_repr_theta = latent_repr_theta.view(B, time, number_neurons, self.samples)
                    latent_repr_q = latent_repr_q.view(B, time, number_neurons, self.samples)

                    zeros = torch.zeros(B, 1, number_neurons, self.samples, device=latent_repr_theta.device)
                    # Concatenate zeros at the beginning and remove the last time step to shift everything by one and shape back to (B, time, n_neurons, n_samples)
                    latent_repr_theta = torch.cat([zeros, latent_repr_theta[:, :-1]], dim=1)
                    latent_repr_q = torch.cat([zeros, latent_repr_q[:, :-1]], dim=1)
                    latent_repr_theta = latent_repr_theta.view(B * time, number_neurons, self.samples)
                    latent_repr_q = latent_repr_q.view(B * time, number_neurons, self.samples)

                q = (
                    q.unsqueeze(-1) + latent_repr_q
                )  # shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations
                logtheta = logtheta.unsqueeze(-1) + latent_repr_theta

                # logtheta = logtheta.unsqueeze(-1).repeat(1, 1, 1, self.samples).permute(0,3,1,2).reshape(B*self.samples,time,number_neurons)
                # latent_repr_theta = latent_repr_theta.reshape(B, time, number_neurons, self.samples).permute(0,3,1,2).reshape(B*self.samples,time,number_neurons)
                # logtheta = torch.cat((logtheta, latent_repr_theta), dim=2)
                # logtheta = self.strong_decoder[data_key](logtheta)[0]
                # logtheta = logtheta.reshape(B,self.samples,time,number_neurons).permute(0,2,3,1).reshape(B*time,number_neurons,self.samples)

                # q = q.unsqueeze(-1).repeat(1, 1, 1, self.samples).permute(0,3,1,2).reshape(B*self.samples,time,number_neurons)
                # latent_repr_q = latent_repr_q.reshape(B, time, number_neurons, self.samples).permute(0,3,1,2).reshape(B*self.samples,time,number_neurons)
                # q = torch.cat((q, latent_repr_q), dim=2)
                # q = self.strong_decoder[data_key](q)[0]
                # q = q.reshape(B,self.samples,time,number_neurons).permute(0,2,3,1).reshape(B*time,number_neurons,self.samples)

            else:
                if self.behavior_in_encoder:
                    means = torch.cat((means, behavior), dim=2)

                if self.decoder:
                    means = self.decoder(means)  # [0]
                    means = means.reshape(-1, means.shape[2])
                else:
                    means = means.reshape(-1, means.shape[2])
                # (B*time,hidden) (hidden,neurons) -> (B*time,neurons)
                if self.position_features:
                    latent_feature_theta = self.mlp_theta[data_key](positions).permute(
                        1, 0
                    )  # shape to (latent_dim, n_neurons)
                    latent_feature_q = self.mlp_q[data_key](positions).permute(1, 0)

                    latent_repr_theta = torch.einsum(
                        "bh,hj->bj", means, latent_feature_theta
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bh,hj->bj", means, latent_feature_q
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)

                else:
                    # theta_feature_norms = torch.norm(self.latent_feature[data_key+"_theta"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_theta"] = self.latent_feature[data_key+"_theta"] / theta_feature_norms

                    # q_feature_norms = torch.norm(self.latent_feature[data_key+"_q"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_q"] = self.latent_feature[data_key+"_q"] / q_feature_norms

                    latent_repr_theta = torch.einsum(
                        "bh,hj->bj", means, self.latent_feature[data_key + "_theta"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bh,hj->bj", means, self.latent_feature[data_key + "_q"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)

                if self.future:  # shift the latents, such that they predict the proceeding time step
                    # Reshape from (B*time, n_neurons, n_samples) to (B, time, n_neurons, n_samples)
                    latent_repr_theta = latent_repr_theta.view(B, time, number_neurons)
                    latent_repr_q = latent_repr_q.view(B, time, number_neurons)

                    zeros = torch.zeros(B, 1, number_neurons, device=latent_repr_theta.device)
                    # Concatenate zeros at the beginning and remove the last time step to shift everything by one and shape back to (B, time, n_neurons, n_samples)
                    latent_repr_theta = torch.cat([zeros, latent_repr_theta[:, :-1]], dim=1)
                    latent_repr_q = torch.cat([zeros, latent_repr_q[:, :-1]], dim=1)
                    latent_repr_theta = latent_repr_theta.view(B * time, number_neurons)
                    latent_repr_q = latent_repr_q.view(B * time, number_neurons)

                logtheta = (
                    logtheta + latent_repr_theta
                )  # shape logtheta and q to (B*time,n_neurons) and add latent representations
                q = q + latent_repr_q  # shape logtheta and q to (B*time,n_neurons) and add latent representations

                # logtheta = logtheta.view(B, time, number_neurons)
                # latent_repr_theta = latent_repr_theta.view(B, time, number_neurons)
                # logtheta = torch.cat((logtheta, latent_repr_theta), dim=2)
                # logtheta = self.strong_decoder[data_key](logtheta)[0]
                # logtheta = logtheta.reshape(B*time,number_neurons)

                # q = q.view(B, time, number_neurons)
                # latent_repr_q = latent_repr_q.view(B, time, number_neurons)
                # q = torch.cat((q, latent_repr_q), dim=2)
                # q = self.strong_decoder[data_key](q)[0]
                # q = q.reshape(B*time,number_neurons)

        if out_predicts or sample_prior:
            theta = self.theta_nl(logtheta)
            loc = self.loc_nl(logloc)
            if self.moment_matching:
                mean = self.moment_matching[data_key + "_mean"]
                variance = self.moment_matching[data_key + "_variance"]
                k = (mean - loc[0, :]) ** 2 / variance
                k = k.repeat(batch_size * time_points, 1)
            elif self.mle_fitting:
                k = self.mle_fitting[data_key + "fitted_k"]
                k = k.repeat(batch_size * time_points, 1)
                # fix loc
                loc = torch.tensor(0.005).repeat(batch_size * time_points, 1).to(theta.device)
            else:
                k = self.k_nl(logk)

            q = self.q_nl(q)
            if sample_prior:
                q = q.mean(dim=2)
                theta = theta.mean(dim=2)
                x = fitted_zig_mean(theta, k, loc, q, no_mixture_mean=self.no_mixture_mean)
            else:
                x = fitted_zig_mean(theta, k, loc, q, no_mixture_mean=self.no_mixture_mean)
            return x.reshape(((batch_size, time_points) + x.size()[1:]))
        else:
            theta = self.theta_nl(logtheta).reshape(((batch_size, time_points) + logtheta.size()[1:]))
            loc = self.loc_nl(logloc).reshape(((batch_size, time_points) + x.size()[2:]))

            if self.moment_matching:
                mean = self.moment_matching[data_key + "_mean"]
                variance = self.moment_matching[data_key + "_variance"]

                # fix loc
                loc = torch.tensor(0.005).repeat(batch_size, time_points, x.shape[2]).to(theta.device)

                k = (mean - loc[0, 0, :]) ** 2 / variance
                k = k.repeat(batch_size * time_points, 1).reshape(((batch_size, time_points) + x.size()[2:]))

            elif self.mle_fitting:
                k = self.mle_fitting[data_key + "fitted_k"]
                k = k.repeat(batch_size * time_points, 1).reshape(((batch_size, time_points) + x.size()[2:]))
                # fix loc
                loc = torch.tensor(0.005).repeat(batch_size, time_points, x.shape[2]).to(k.device)
            else:
                k = self.k_nl(logk).reshape(((batch_size, time_points) + x.size()[2:]))

            q = self.q_nl(q).reshape(((batch_size, time_points) + q.size()[1:]))
            if self.latent:
                if train:
                    return (theta, k, loc, q, means, sigma_squared, self.latent_feature, self.samples, mask)
                else:
                    return (theta, k, loc, q, means, sigma_squared, self.latent_feature, self.samples)
            else:
                return (theta, k, loc, q)

    def predict_mean(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_mean(theta, k, loc, q)

    def predict_variance(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_variance(theta, k, loc, q)

    def forward_prior(
        self,
        x,
        data_key,
        behavior_core=None,
        pupil_center_core=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        out_predicts=True,
        repeats=10,
        n_samples=100,
        positions=None,
        **kwargs
    ):
        if (
            "behavior" in kwargs.keys()
        ):  # rename behavior, pupil_center keys, which are for latent input, in kwargs to avoid conflicts with core input
            kwargs["behavior_encoder"] = kwargs.pop("behavior")
            kwargs["pupil_center_encoder"] = kwargs.pop("pupil_center")

        x, q_zig, logloc, readout_out_idx, batch_size, time_points = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior_core,
            pupil_center=pupil_center_core,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )

        if "logtheta" in dir(self):
            logtheta_zig = getattr(self, "logtheta")[data_key].repeat(batch_size * time_points, 1)
        else:
            logtheta_zig = x[readout_out_idx]
            readout_out_idx += 1

        if "logk" in dir(self):
            logk = getattr(self, "logk")[data_key].repeat(batch_size * time_points, 1)
        else:
            logk = (
                x[readout_out_idx].mean(dim=0).repeat(batch_size * time_points, 1)
            )  # average k over time, it should be constant for each neuron (not time dependent as theta and q)
            readout_out_idx += 1

        responses = kwargs["responses"].clone()
        B, n_neurons, total_time = responses.shape

        k = self.mle_fitting[data_key + "fitted_k"]
        k = k.repeat(batch_size * time_points, 1).reshape(((batch_size, time_points) + x.size()[2:]))
        # fix loc
        loc = torch.tensor(0.005).repeat(batch_size, time_points, x.shape[2]).to(k.device)

        q_all = torch.zeros(B * time_points, n_neurons, device=responses.device)
        theta_all = torch.zeros(B * time_points, n_neurons, device=responses.device)
        self.samples = n_samples
        for _ in range(repeats):
            if self.latent:
                samples = torch.randn(B, time_points, self.encoder.output_dim, self.samples).to(responses.device)

                if self.behavior_in_encoder:
                    behavior = kwargs["behavior_encoder"].clone().permute(0, 2, 1)  # shape it to (B, time, behav)
                    if (
                        self.behavior_in_encoder["input_size"] == 4
                    ):  # if pupil_center should be included in encoder as well
                        pupil_center = kwargs["pupil_center_encoder"].clone().permute(0, 2, 1)
                        behavior = torch.cat((behavior, pupil_center), dim=2)
                    behavior = self.mlp_behavior[data_key](
                        behavior
                    )  # apply MLP to scale dim from 2/4 to bigger dim like 10
                    behavior = behavior[:, -time_points:]  # chop of the first time points which are not needed
                    samples = torch.cat((samples, behavior.unsqueeze(-1).repeat(1, 1, 1, self.samples)), dim=2)
                else:
                    behavior = None

                if self.decoder:
                    samples = samples.permute(0, 3, 1, 2)
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # after that it has form (B*n_samples,time,hidden), every sample goes independetly through decoder
                    samples = self.decoder(samples)  # [0]
                    samples = samples.reshape(
                        B, self.samples, samples.shape[1], samples.shape[2]
                    )  # (B,n_samples,time,hidden) after reshape
                    samples = samples.permute(0, 2, 3, 1)  # (B,time,hidden,n_samples) after permute
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # reshape samples to (B*time,hidden,n_samples)
                    # samples = self.decoder(samples)
                else:
                    samples = samples.reshape(
                        -1, samples.shape[2], samples.shape[3]
                    )  # reshape samples to (B*time,hidden,n_samples)

                # (B*time,hidden,samples) (hidden,neurons) -> (B*time,neurons,samples)
                if self.position_features:
                    latent_feature_theta = self.mlp_theta[data_key](positions).permute(
                        1, 0
                    )  # shape to (latent_dim, n_neurons)
                    latent_feature_q = self.mlp_q[data_key](positions).permute(1, 0)

                    latent_repr_theta = torch.einsum(
                        "bhi,hj->bji", samples, latent_feature_theta
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bhi,hj->bji", samples, latent_feature_q
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                else:
                    # theta_feature_norms = torch.norm(self.latent_feature[data_key+"_theta"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_theta"] = self.latent_feature[data_key+"_theta"] / theta_feature_norms

                    # q_feature_norms = torch.norm(self.latent_feature[data_key+"_q"], dim=0, keepdim=True)
                    # self.latent_feature[data_key+"_q"] = self.latent_feature[data_key+"_q"] / q_feature_norms

                    latent_repr_theta = torch.einsum(
                        "bhi,hj->bji", samples, self.latent_feature[data_key + "_theta"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)
                    latent_repr_q = torch.einsum(
                        "bhi,hj->bji", samples, self.latent_feature[data_key + "_q"]
                    )  # this computes w_z*z and has shape (B*time,n_neurons,samples)

                if self.future:  # shift the latents, such that they predict the proceeding time step
                    # Reshape from (B*time, n_neurons, n_samples) to (B, time, n_neurons, n_samples)
                    latent_repr_theta = latent_repr_theta.view(B, time_points, n_neurons, self.samples)
                    latent_repr_q = latent_repr_q.view(B, time_points, n_neurons, self.samples)

                    zeros = torch.zeros(B, 1, n_neurons, self.samples, device=latent_repr_theta.device)
                    # Concatenate zeros at the beginning and remove the last time step to shift everything by one and shape back to (B, time, n_neurons, n_samples)
                    latent_repr_theta = torch.cat([zeros, latent_repr_theta[:, :-1]], dim=1)
                    latent_repr_q = torch.cat([zeros, latent_repr_q[:, :-1]], dim=1)
                    latent_repr_theta = latent_repr_theta.view(B * time_points, n_neurons, self.samples)
                    latent_repr_q = latent_repr_q.view(B * time_points, n_neurons, self.samples)

                logtheta = (
                    logtheta_zig.unsqueeze(-1) + latent_repr_theta
                )  # shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations
                q = (
                    q_zig.unsqueeze(-1) + latent_repr_q
                )  # shape logtheta and q to (B*time,n_neurons,n_samples) and add latent representations

                theta = self.theta_nl(logtheta)
                q = self.q_nl(q)

            if out_predicts:
                q_all = q_all + q.sum(dim=2)
                theta_all = theta_all + theta.sum(dim=2)
            else:
                if self.latent:
                    q = q.reshape(((batch_size, time_points) + q.size()[1:]))
                    theta = theta.reshape(((batch_size, time_points) + theta.size()[1:]))
                    return (theta, k, loc, q, self.samples)
                else:
                    theta_zig = self.theta_nl(logtheta_zig)
                    q_zig = self.q_nl(q_zig)
                    return (theta_zig, k, loc, q_zig)

        if out_predicts:
            q = 1 / (repeats * self.samples) * q_all
            theta = 1 / (repeats * self.samples) * theta_all
            x = fitted_zig_mean(theta, k, loc, q, no_mixture_mean=self.no_mixture_mean)
            return x.reshape(((batch_size, time_points) + x.size()[2:]))


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

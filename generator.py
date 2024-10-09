import torch
from attention import SpatialTransformer

class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, batch_size, first_kernel_size, num_heads=1, context_dim=256):
        super(Generator, self).__init__()

        # Hidden layers
        self.length = len(num_filters)
        self.hidden_layer = torch.nn.Sequential()
        layer = 0

        for i in range(len(num_filters)):
            model = torch.nn.Sequential()
            for j in range(len(num_filters[i])):
                if (i == 0) and (j == 0):
                    deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i][j], kernel_size=first_kernel_size, stride=1, padding=0)
                elif (j == 0):
                    deconv = torch.nn.ConvTranspose2d(num_filters[i-1][-1], num_filters[i][j], kernel_size=4, stride=2, padding=1)
                else:
                    deconv = torch.nn.ConvTranspose2d(num_filters[i][j-1], num_filters[i][j], kernel_size=4, stride=2, padding=1)

                deconv_name = 'deconv' + str(layer + 1)
                layer = layer + 1
                model.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(layer + 1)
                layer = layer + 1
                model.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i][j]))

                # Activation
                act_name = 'act' + str(layer + 1)
                layer = layer + 1
                model.add_module(act_name, torch.nn.ReLU())
            model_name = 'model' + str(i + 1)
            self.hidden_layer.add_module(model_name, model)
            if i < (len(num_filters) - 1):
                cross_attention = SpatialTransformer(num_filters[i][-1], num_heads, num_filters[i][-1] // num_heads, depth=1, context_dim=context_dim)
                cross_attention_name = 'cross attention' + str(i + 1)
                self.hidden_layer.add_module(cross_attention_name, cross_attention)
    
        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i][j], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

        # Residual layer
        self.residual_layer = torch.nn.Sequential()
        res = torch.nn.Conv2d(in_channels=batch_size,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.residual_layer.add_module('res', res)
        torch.nn.init.normal(res.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(res.bias, 0.0)

    def forward(self, x, cond):
        text_cond = cond.unsqueeze(1)
        for i in range(self.length - 1):
            x = self.hidden_layer[2*i](x)
            x = self.hidden_layer[2*i + 1](x, text_cond)
        x = self.hidden_layer[2 * (self.length - 1)](x)
        out = self.output_layer(x)
        out_ = out.permute(1,0,2,3)
        adv_out = self.residual_layer(out_)
        adv_out_ = adv_out.permute(1,0,2,3)
        return adv_out_

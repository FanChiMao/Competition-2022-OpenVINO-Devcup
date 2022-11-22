import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Type
from typing import Optional, Mapping
from librosa import stft, istft
from music_source_separation.umx_openvino.filtering import wiener

class UMX_openvino:
    
    def __init__(self, device, model_path):
        super(UMX_openvino, self).__init__()
        self.device = device
        self.model_path = model_path
        self.build_model()
    
    def build_model(self):
        core = Core()
        self.model = core.read_model(self.model_path)
        
        ppp = PrePostProcessor(self.model)
        
        for i in range(len(self.model.inputs)):
            ppp.input(i).tensor().set_element_type(Type.f32)
        for i in range(len(self.model.outputs)):
            ppp.output(i).tensor().set_element_type(Type.f32)

        self.model = ppp.build()
        
        devices = self.device.replace('HETERO:', '').split(',')
        device_str = f'HETERO:{",".join(devices)}' if 'HETERO' in self.device else devices[0]
        plugin_config = {}
        compiled_model = core.compile_model(self.model, device_str, plugin_config)
        
        self.infer_request = compiled_model.create_infer_request()
    
    def inference(self, input):
        
        frames_to_infer = {}
        for state in self.infer_request.query_state():
            state.reset()
        
        for _input in self.infer_request.model_inputs:
            #frames_to_infer[_input.any_name] = input.reshape(_input.tensor.shape)
            frames_to_infer[_input.any_name] = input
        #self.models.reshape(input.shape)
        frame_results = self.infer_request.infer(frames_to_infer)
        results = frame_results[self.infer_request.model_outputs[0]]
        return results
    
class Separator_openvino:
    def __init__(
        self,
        target_models: Mapping[str, UMX_openvino],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator_openvino, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len
        
        self.target_models = target_models
        self.nb_targets = len(self.target_models)
        self.sample_rate = sample_rate
        
    def inference(self, audio):
        """Performing the separation on audio input
        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform
        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        #mix_stft = self.stft(audio)
        #X = self.complexnorm(mix_stft)
        
        mix_stft = stft(audio, n_fft=4096, hop_length=1024, center=True, pad_mode="reflect")
        
        X = np.sqrt(mix_stft.real**2 + mix_stft.imag**2)
        # initializing spectrograms variable
        #spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)
        spectrograms = np.zeros(X.shape + (nb_sources,))
        
        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current models to get the source spectrogram
            target_spectrogram = target_module.inference(X)
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.transpose(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.transpose(0, 3, 2, 1)
        
        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = np.zeros(mix_stft.shape + (2, nb_sources,), dtype="complex_")
        
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = np.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1
                targets_stft[sample, cur_frame]  = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter
                )
        
        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = np.ascontiguousarray(targets_stft.transpose(0, 5, 3, 2, 1, 4))
        targets_stft = targets_stft[...,0] + targets_stft[...,1] * 1j
        # inverse STFT
        estimates = istft(targets_stft, hop_length=1024, win_length=4096, n_fft=4096, length=audio.shape[2])
        return estimates

    def to_dict(self, estimates, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary
        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)
        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
        
if __name__ == "__main__":
    import os
    
    import torch
    from openunmix import utils
    
    bass_umx = UMX_openvino("CPU", os.path.join("../models", "bass.onnx"))
    drums_umx = UMX_openvino("CPU", os.path.join("../models", "drums.onnx"))
    other_umx = UMX_openvino("CPU", os.path.join("../models", "other.onnx"))
    vocals_umx = UMX_openvino("CPU", os.path.join("../models", "vocals.onnx"))
    
    separator1 = Separator_openvino({"bass": bass_umx,
                                    "drums": drums_umx,
                                    "other": other_umx,
                                    "vocals": vocals_umx}, niter=1)
    
    input = 2*torch.rand([1,2,44100*3])-1
    input = utils.preprocess(input, 44100, separator1.sample_rate)
    input_ = input.detach().cpu().numpy()
    estimates1 = separator1.inference(input_)
    estimates1 = separator1.to_dict(estimates1, aggregate_dict=None)
    
    source = "vocals"
    
    #print(estimates1[source].shape)
    
    from openunmix import utils
    
    separator2 = utils.load_separator(
        model_str_or_path="umxl",
        niter=1
    )
    estimates2 = separator2(input)
    estimates2 = separator2.to_dict(estimates2, aggregate_dict=None)
    
    #print(estimates2[source].shape)
    #print(np.allclose(estimates1[source].detach().cpu().numpy(),estimates2[source].detach().cpu().numpy(), rtol=0, atol=1e-6)) # atol=1e-7 -> False
    print(np.allclose(estimates1[source],estimates2[source].detach().cpu().numpy(), rtol=0, atol=1e-6)) # atol=1e-7 -> False
    
    import matplotlib.pyplot as plt
    
    #fig, ax1 = plt.subplots()
    #ax1.plot(estimates2[source][0,0,:].cpu().detach().numpy(), color="green")
    #ax2 = ax1.twinx()
    #ax2.plot(estimates1[source][0,0,:].cpu().detach().numpy(), color="red", linestyle="--")
    #plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.plot(estimates2[source][0,0,:].cpu().detach().numpy(), color="green")
    ax2 = ax1.twinx()
    ax2.plot(estimates1[source][0,0,:], color="red", linestyle="--")
    plt.show()
    
    """
    from openunmix import umxl_spec
    from openunmix import utils
    input = 2*torch.rand([1,2,2049,44])-1
    
    models = umxl_spec(targets=None, device="cpu", pretrained="True")
    
    source = "vocals"
    
    umx_model = models[source]
    output1 = umx_model(input)
    
    umx_openvino = UMX_openvino("CPU", os.path.join("models", "{}.onnx".format(source)))
    output2 = torch.from_numpy(umx_openvino.inference(input))
    
    print(output1.shape, output1.dtype)
    print(output2.shape, output1.dtype)
    print(np.allclose(output1.detach().cpu().numpy(),output2.detach().cpu().numpy(), rtol=0, atol=1e-5)) # atol=1e-7 -> False
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots()
    ax1.plot(output1[0,0,:,2].cpu().detach().numpy(), color="green")
    ax2 = ax1.twinx()
    ax2.plot(output2[0,0,:,2].cpu().detach().numpy(), color="red", linestyle="--")
    plt.show()
    """
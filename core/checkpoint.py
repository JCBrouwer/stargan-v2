"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print("Saving checkpoint into %s..." % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        torch.save(outdict, fname)

    def load(self, step, restore_D=True):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + " does not exist!"
        print("Loading checkpoint from %s..." % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device("cpu"))
        for name, module in self.module_dict.items():

            # for checkpoints with fewer domains, duplicate domain specific modules to correct number
            if name == "mapping_network" or name == "style_encoder":
                domain_mappers_resume = [k for k in module_dict[name].keys() if "unshared" in k]
                domain_mappers = [k for k in module.state_dict().keys() if "unshared" in k]
                if len(domain_mappers_resume) < len(domain_mappers):
                    while len(domain_mappers_resume) < len(domain_mappers):
                        domain_mappers_resume += domain_mappers_resume
                    for our_key, resume_key in zip(domain_mappers, domain_mappers_resume):
                        module_dict[name][our_key] = module_dict[name][resume_key]
            if name == "discriminator" and not restore_D:
                continue  # skip restoring discriminator

            module.load_state_dict(module_dict[name])

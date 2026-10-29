import unittest
from pathlib import Path
import numpy as np
import torch
from ruamel.yaml import YAML
from fm4npp.models.mamba_config import build_backbone
from dev_scripts.transformer_pretraining_20261029.report import attention_training_flops

HERE=Path(__file__).resolve().parent


class CampaignTests(unittest.TestCase):
    def test_recipes_preserve_controls_and_initial_states(self):
        y=YAML(typ='safe')
        recipes=y.load((HERE/'recipe.yaml').read_text())
        baseline=y.load((HERE.parent/'backbone_pretraining_comparison/recipe.yaml').read_text())['mamba1_chinchilla_w512_d12_k1']
        allowed={'backbone_version','backbone_sequence_encoding','backbone_rope_theta','num_heads_backbone','backbone_mlp_ratio','save_version'}
        states=[]
        for p in recipes.values():
            for key,value in baseline.items():
                if key not in allowed: self.assertEqual(p[key],value,key)
            torch.manual_seed(42); m=build_backbone(p)
            self.assertEqual(sum(t.numel() for t in m.parameters()),19339651)
            states.append(m.state_dict())
        self.assertEqual(states[0].keys(),states[1].keys())
        for key in states[0]: torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)

    def test_attention_cost_uses_squared_padded_length(self):
        values=attention_training_flops(np.array([64*10,64*20]))
        self.assertEqual(values[1],4*values[0])
        self.assertEqual(values[0],3*4*12*64*10*10*512)


if __name__=='__main__': unittest.main()

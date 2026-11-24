"""feature_mix: how adapters combine the frozen backbone's layer outputs."""

import tempfile
import unittest
from pathlib import Path

import torch

from fm4npp.models.mamba_config import adapter_options, architecture_metadata, validate_checkpoint_architecture
from fm4npp.models.rmsnorm import RMSNorm
from train.downstream.feature_mix import LastLayerNorm, LayerStandardizer, attach_backbone_norm, mix_layers, valid_tokens
from train.downstream.model import AttentionHead, MambaHead, MambaTrackRegressionHead
from train.downstream.trackinghead import MambaAttentionHead

HEADS = (MambaHead, MambaTrackRegressionHead, AttentionHead, MambaAttentionHead)


def head(cls, **mix):
    options = dict(input_dim=32, embed_dim=16, num_layers=1, num_embedder_layers=1, num_feature_layers=3,
                   embed_method="pos_only", dropout=0., num_output_dim=3,
                   **adapter_options(dict(adapter_version="transformer", adapter_num_heads=4,
                                          adapter_sequence_encoding="none", **mix)))
    if cls is MambaAttentionHead:
        options.update(num_output_dim=16, num_prototypes=3, ffn_dim=32, num_self_attn_layers=1)
    if cls is AttentionHead:
        options.update(num_heads=4)
    return cls(**options)


def massive_features(B=4, N=10, L=3, D=32, seed=0):
    """Layer 2 carries a constant offset 1e4 times its signal, like layers 7-12 of the adopted M4."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(L, B, N, D, generator=g)
    f[2] = f[2] + 1e4 * torch.linspace(-1, 1, D)
    return f


class FeatureMixTest(unittest.TestCase):
    def test_raw_is_the_unchanged_default(self):
        self.assertNotIn("feature_mix", adapter_options({}))
        self.assertNotIn("feature_mix", adapter_options({"feature_mix": "raw"}))
        f, w = torch.randn(3, 2, 5, 8), torch.randn(3)
        old = torch.einsum("bsnd,n->bsd", f.permute(1, 2, 0, 3), torch.softmax(w, 0))
        torch.testing.assert_close(mix_layers(f, w), old, rtol=0, atol=0)
        for cls in HEADS:
            model = head(cls)
            self.assertIsNone(model.feature_mixer)
            self.assertFalse(any("feature_mixer" in k for k in model.state_dict()))
            self.assertNotIn("feature_mix", architecture_metadata(model))
        with self.assertRaisesRegex(ValueError, "feature_mix"):
            adapter_options({"feature_mix": "layernorm"})

    def test_last_is_the_backbone_final_norm_of_the_last_layer(self):
        f, w = massive_features(), torch.randn(3)
        for norm in (RMSNorm(32), torch.nn.LayerNorm(32)):
            with torch.no_grad():
                norm.weight.uniform_(0.5, 2)
                if isinstance(norm, torch.nn.LayerNorm):
                    norm.bias.uniform_(-1, 1)
            mixer = LastLayerNorm(32)
            with self.assertRaisesRegex(RuntimeError, "attach_backbone_norm"):
                mix_layers(f, w, mixer)
            mixer.attach(norm)
            with torch.no_grad():
                torch.testing.assert_close(mix_layers(f, w, mixer), norm(f[-1]))
        model = head(MambaHead, feature_mix="last")
        self.assertEqual(architecture_metadata(model)["feature_mix"], "last")
        backbone = torch.nn.Module(); backbone.norm = RMSNorm(32)
        with torch.no_grad():
            backbone.norm.weight.fill_(3.)
        attach_backbone_norm(model, backbone)
        self.assertEqual(model.feature_mixer.weight.unique().item(), 3.)
        attach_backbone_norm(head(MambaHead), backbone)        # no effect on the other mixes

    def test_standardize_uses_valid_clusters_then_freezes(self):
        s = LayerStandardizer(3, 32, stat_batches=2).train()
        f = massive_features()
        valid = torch.ones(4, 10, dtype=torch.bool)
        valid[:, 7:] = False
        f[:, valid.logical_not()] = 1e6                           # padding must not enter the statistics
        out = s(f, valid)
        self.assertLess(out[:, valid].mean((1,)).abs().max().item(), 1e-3)
        self.assertLess((out[:, valid].std((1,)) - 1).abs().max().item(), 0.05)
        s(massive_features(seed=1), valid)
        frozen = s.mean.clone(), s.std.clone(), int(s.batches)
        s(massive_features(seed=2) * 3, valid)                    # past stat_batches: no update
        self.assertEqual(int(s.batches), 2)
        torch.testing.assert_close(s.mean, frozen[0]); torch.testing.assert_close(s.std, frozen[1])
        s.eval()
        with self.assertRaisesRegex(RuntimeError, "no statistics"):
            LayerStandardizer(3, 32).eval()(f, valid)

    def test_constant_channels_are_centred_not_amplified(self):
        s = LayerStandardizer(1, 4, stat_batches=1).train()
        f = torch.zeros(1, 1, 50, 4)
        f[..., 0] = torch.randn(50)
        f[..., 1] = 7.0                                           # constant channel
        out = s(f)
        self.assertEqual(out[..., 1].abs().max().item(), 0.)
        self.assertLess(abs(out[..., 0].std().item() - 1), 0.05)

    def test_bfloat16_keeps_variation_under_a_large_offset(self):
        s = LayerStandardizer(3, 32, stat_batches=1).train()
        f = massive_features()
        s(f)
        out = s.eval()(f.bfloat16())
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertGreater(out[2].float().std().item(), 0.5)    # bf16 subtraction would leave ~0 or 64-steps

    def test_heads_train_save_and_reload(self):
        torch.manual_seed(0)
        backbone = torch.nn.Module(); backbone.norm = RMSNorm(32)
        for cls, mix in [(c, m) for c in HEADS for m in ("standardize", "last")]:
            model = head(cls, feature_mix=mix, feature_stat_batches=1)
            attach_backbone_norm(model, backbone)
            self.assertEqual(architecture_metadata(model)["feature_mix"], mix)
            x = torch.rand(2, 10, 3)
            x[1, 8:] = -100
            mask = valid_tokens(x, None)
            out = model.train()(x, massive_features(B=2), padding_mask=mask, pretrain=True)
            loss = sum(v.float().square().mean() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "head.pth"
                torch.save({"model_state_dict": model.state_dict(), "adapter_architecture": architecture_metadata(model)}, path)
                saved = torch.load(path, weights_only=False)
            again = head(cls, feature_mix=mix)
            again.load_state_dict(saved["model_state_dict"], strict=True)
            for name, buf in model.feature_mixer.named_buffers():
                torch.testing.assert_close(getattr(again.feature_mixer, name), buf)
            validate_checkpoint_architecture(saved, again, "adapter")
            with self.assertRaises(ValueError):                   # a raw head must not load these weights
                validate_checkpoint_architecture(saved, head(cls), "adapter")


if __name__ == "__main__":
    unittest.main()

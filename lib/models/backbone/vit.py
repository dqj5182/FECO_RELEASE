import timm
import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(
        self,
        model_name='vit_huge_patch14_224',
        pretrained=True,
        return_cls=False,
        style_stage=6,
        out_indices=(2, 5, 8, 11),  # typical DPT stages for 12-layer ViT
    ):
        super().__init__()
        self.return_cls = return_cls
        self.style_stage = style_stage
        self.out_indices = out_indices

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size

    def forward(self, x, return_intermediate=False):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        intermediate_outputs = []
        style_feat = None

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.out_indices:
                patch_tokens = x[:, 1:]  # remove cls
                H = W = int(patch_tokens.shape[1] ** 0.5)
                feat = patch_tokens.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
                intermediate_outputs.append(feat)
            if i + 1 == self.style_stage and return_intermediate:
                style_feat = x

        x = self.vit.norm(x)
        patch_tokens = x[:, 1:]
        H = W = int(patch_tokens.shape[1] ** 0.5)
        final_feat = patch_tokens.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        if return_intermediate:
            result = {
                'intermediate_feats': intermediate_outputs,
                'final_feat': final_feat
            }
            if style_feat is not None:
                style_tokens = style_feat[:, 1:]
                Hs = Ws = int(style_tokens.shape[1] ** 0.5)
                style_feat = style_tokens.view(B, Hs, Ws, self.embed_dim).permute(0, 3, 1, 2)
                result['style_stage_feat'] = style_feat
            return result
        else:
            return final_feat
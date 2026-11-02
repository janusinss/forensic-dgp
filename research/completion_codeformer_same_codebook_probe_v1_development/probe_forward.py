"""Owned diagnostic forward copied from pinned CodeFormer.forward.

Only selected code indices may be replaced by an explicitly supplied ORACLE
array from the same inpainting prior. Observed encoder/context/fusion features
are always computed from x. Dependencies are injected; no scientific imports.
The copied function is never assigned to or patched into the production class.
"""

def select_code_indices(predicted,oracle_indices,*,codebook_size):
    if oracle_indices is None:return predicted
    if oracle_indices.dtype!=predicted.dtype or oracle_indices.device!=predicted.device or oracle_indices.shape!=predicted.shape:
        raise ValueError('ORACLE index dtype/device/shape differs')
    if oracle_indices.min().item()<0 or oracle_indices.max().item()>=codebook_size:
        raise ValueError('ORACLE index outside SAME512-codebook range')
    return oracle_indices.detach().clone()

def probe_forward(self, x, w=0, detach_16=True, code_only=False, adain=False, *, F, torch, adaptive_instance_normalization, oracle_indices=None):
    enc_feat_dict = {}
    out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
    for i, block in enumerate(self.encoder.blocks):
        x = block(x)
        if i in out_list:
            enc_feat_dict[str(x.shape[-1])] = x.clone()
    lq_feat = x
    pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
    feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
    query_emb = feat_emb
    for layer in self.ft_layers:
        query_emb = layer(query_emb, query_pos=pos_emb)
    logits = self.idx_pred_layer(query_emb)
    logits = logits.permute(1, 0, 2)
    if code_only:
        return (logits, lq_feat)
    soft_one_hot = F.softmax(logits, dim=2)
    _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
    predicted_indices = top_idx.clone()
    top_idx = select_code_indices(top_idx, oracle_indices, codebook_size=512)
    quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
    if detach_16:
        quant_feat = quant_feat.detach()
    if adain:
        quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
    x = quant_feat
    fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
    for i, block in enumerate(self.generator.blocks):
        x = block(x)
        if i in fuse_list:
            f_size = str(x.shape[-1])
            if w > 0:
                x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
    out = x
    return (out, logits, lq_feat, predicted_indices, top_idx.clone())

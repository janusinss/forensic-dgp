"""Explicit biased identity backward for actual saved-PNG forward values.

No framework is imported. A guarded caller supplies Torch and the real durable
PNG decoded tensor; this helper cannot authenticate that reference by itself.
"""


def build_png_surrogate(torch):
    """Build a custom Function class without constructing a neural model.

    apply(delivered, decoded_png_tensor) returns a detached clone of the actual
    supplied PNG values, exactly. The backward is an identity estimator for the
    delivered input only; it is not the true derivative of a hard PNG exporter.
    """
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    def tensor(value, *, connected):
        need(isinstance(value, torch.Tensor) and value.shape == (1, 3, 256, 256)
             and value.dtype == torch.float32 and torch.isfinite(value).all().item()
             and value.min().item() >= 0 and value.max().item() <= 1,
             'Finite FP32 RGB256 tensor in[0,1] required')
        need(bool(value.requires_grad) == connected,
             'Delivered RAW requires gradient; decoded PNG reference must be detached')

    class PNGDecodedReferenceIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, delivered, decoded_png_tensor):
            tensor(delivered, connected=True)
            tensor(decoded_png_tensor, connected=False)
            need(delivered.device == decoded_png_tensor.device,
                 'Delivered RAW and actual saved-PNG reference must share a device')
            return decoded_png_tensor.detach().clone()

        @staticmethod
        def backward(ctx, grad_output):
            need(isinstance(grad_output, torch.Tensor)
                 and grad_output.shape == (1, 3, 256, 256)
                 and grad_output.dtype == torch.float32
                 and torch.isfinite(grad_output).all().item(),
                 'Finite FP32 RGB256 upstream surrogate gradient required')
            return grad_output, None

    return PNGDecodedReferenceIdentity

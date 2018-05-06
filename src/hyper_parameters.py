from tensor2tensor.models.transformer import transformer_base_single_gpu
from tensor2tensor.utils import registry


@registry.register_hparams
def transformer_base_small_gpu():
    """HParams for transformer base model for single GPU with low memory."""
    hparams = transformer_base_single_gpu()
    hparams.batch_size = 512
    hparams.learning_rate_warmup_steps = 16000
    return hparams

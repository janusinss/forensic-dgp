# Reconstruction-path review after V42

V40, V41 and V42 retain the same added 17,952-parameter spatial decoder and
frozen original DGP. Their objectives or optimizer handling differ, but all
three stop at 50 updates without qualifying structure. V42 also fails compound
appearance preservation in both sources, in raw and PNG outputs. Its failure
is independently audited and remains binding.

The invalid assumption is that nonzero gradients, changed decoder weights and
direct correction supervision necessarily produce enough useful facial structure
under this frozen-base recipe. V42 changes all 57 learned tensors and slightly
reduces a fixed-cohort correction loss, while giving only 0.00692364% delivered
structure improvement. This does not prove a unique architecture, optimizer or
loss cause, or that all possible spatial decoders are incapable.

## What the next review must distinguish

The user requested improving the current trained DGP and investigating additional
epochs. V42 trains a new decoder while keeping the entire original DGP frozen;
it does not extend the original model's training. More epochs of these two
different trainable paths are different experiments. V42 ends before one complete
epoch, so it provides no epoch-1/2/5 comparison and no proof of undertraining.

The current model already contains an FPN/MobileNet feature path, four pyramid
heads and an RGB reconstruction path. A serialized metadata check identifies
14 original reconstruction convolution tensors with 609,219 parameters, compared
with the added decoder's 17,952. This is a parameter count, not a capacity test
or proof that increasing trainable parameters will improve preservation. The
encoder and normalization buffers are outside that 14-tensor count.

The added decoder receives the observed image and baseline plus detached,
upsampled lateral0 and smooth2 maps. All 102 channels are compressed to width
16 before a two-scale path. The original feature representations cannot adapt
in V42. Whether this bottleneck, loss alignment or finite preservation coverage
dominates the weak learning remains unproved.

The local app's DGP adapter explicitly uses `torch.inference_mode()` in its
forward method. Merely toggling `requires_grad` cannot turn that inference API
into a valid original-model learner. Any proposed learner needs a separate,
reviewed VM-only forward with initial inference parity, preserved normalization
and explicit trainable tensor ownership. Shared encoder module aliases require
unique parameter enumeration. No such learner is implemented in this review.

## Primary research checked

The official NAFNet architecture has a multiscale encoder/decoder, skip features,
learnable residual scales and a final image residual. Our added decoder is a
distinct small adaptation and does not inherit NAFNet's experimental performance.
This supports inspecting the entire reconstruction path, not treating a few
similarly named blocks as a validated restoration model.
[Official NAFNet implementation](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py).

MPRNet's released losses use robust pixel error and a Laplacian construction.
Its broader architecture also exchanges features across restoration stages.
V42's correction-loss adaptation alone is not a reproduction or evidence that
its training recipe should work on CCTV faces.
[Official losses](https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py),
[original paper](https://arxiv.org/abs/2102.02808).

DeblurGAN-v2's original work uses an FPN generator for motion deblurring with
declared backbone choices. Its deblurring results do not establish preservation
or native CCTV face quality for our fine-tuned checkpoint.
[Original project](https://github.com/VITA-Group/DeblurGANv2),
[original paper](https://arxiv.org/abs/1908.03826).

No weights from these external restoration models are downloaded or substituted
for our trained DGP in this review. The research informs possible designs;
the next direction is an inference, not an experimentally established remedy.

## Recommended discussion direction

Review a separate copy of the current DGP's original reconstruction and feature
path before another added-decoder pilot. The diagnostic question is pending:
review original encoder/decoder training, or investigate the tiny added-decoder
corrections further. No answer or approval is inferred from elapsed time.

The review needs to satisfy five conditions before a new manual VM packet:

1. Reconcile the original-path failures already retained from V6/V9 and V33–V38;
   do not rename an unchanged rejected decoder recipe or assume more epochs fix it.
2. Establish initial parity and a valid trainable path from the retained current
   checkpoint, with normalization and parameter ownership explicitly checked.
3. Distinguish reconstruction learning from preservation interference using a
   finite, predeclared diagnostic; keep clean targets outside model inputs.
4. Retain the existing raw/PNG structure, appearance, source/profile and brightness
   requirements and whole-face visual review. A larger model is not a waiver.
5. Declare updates, exposure, timing, VRAM, storage, export and portable state
   requirements before transfer. All actual learning remains manual on the L4.

Audited HQ references and approved source replay remain the training candidates.
Native CCTV development review remains unpaired. Reserved final identities stay
outside tuning. No new acquisition, native/final pixel exposure, app change or
completion-model experiment occurs here. All visible facial features and seven
covering families remain in the full goal.

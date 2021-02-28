
# deep vector quantization

Experiments with VQVAEs, i.e. autoencoders with categorical latent variable bottlenecks, which are then easy to subsequently plug into existing infrastructure for modeling sequences of discrete variables (GPT and friends). In a semi-rough state with magic number in the code, still in process of cleaning up...

DeepMind's [VQVAE](https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb)

Should train out of the box as `cd dvq; python vqvae.py --gpus 1 --vq_flavor vqvae`.

I am able to get what I think are expected results on CIFAR-10 using VQVAE (judging by reconstruction loss achieved). However I had to resort to a data-driven intialization scheme with k-means (which is with current implementation not multi-gpu compatible), and which the sonnet repo does not use, potentially due to more careful model initialization treatment. When I do not use data-driven init the training exhibits catastrophic index collapse.

Jang et al. [Gumbel Softmax](https://arxiv.org/abs/1611.01144) version, also the version used in [DALL-E](https://openai.com/blog/dall-e/) allegedly, though we have not seen the details yet.

Should train out of the box as `cd dvq; python vqvae.py --gpus 1 --vq_flavor gumbel`.

Trains and converges to slightly higher reconstruction loss, but tuning the scale of the kl divergence loss and the temperature decay rate and the version of gumbel (soft/hard) has so far proved a little bit finicky. Also the whole thing trains much slower. Requires a bit more thorough hyperparameter search than a few one-off guesses.

[OpenAI's DALL-E](https://openai.com/blog/dall-e/) VQVAE model version can also in principle be trained using `--enc_dec_flavor openai --vq_flavor gumbel`, but the re-implementation is not yet complete, e.g. we still use MSE is still used as a loss, the (hardcoded) hyperparameter decays are wrong, we're only using CIFAR-10 so far, etc etc. However, the different encoder/decoder architecture trains and gives comparable results to the (simpler) DeepMind version on untuned 1-GPU trial runs on stride /4 VQVAEs. Situation is developing...

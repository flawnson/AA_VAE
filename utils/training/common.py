import torch


def label_smoothing(inputs, epsilon):
    k = inputs.size()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / k)


def calculate_gradient_stats(parameters):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_grad = max(p.grad.data.max() for p in parameters)
    min_grad = min(p.grad.data.min() for p in parameters)
    return max_grad, min_grad


def kl_loss_function(mu, logvar):
    """

    :param mu: Mean of the embedding.
    :param logvar: variance of the embedding.
    :return: KL Loss of the embeddings

     Calculates the representation loss of the embedding.
     see Appendix B from VAE paper:
     Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
     https://arxiv.org/abs/1312.6114
     0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    """
    kld: torch.Tensor = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def reconstruction_accuracy(predicted, actual, mask):
    """

    :param predicted: The result returned by the model
    :param actual: The comparison data
    :param mask: The mask that differentiates the actual data points from padding.
    :return: The accuracy of reconstruction

    Computes average sequence identity between input and output sequences
    """
    output_sequences = torch.masked_select(actual, mask)
    input_sequences = torch.masked_select(predicted.argmax(axis=1), mask)

    return (((input_sequences == output_sequences).sum()) / float(len(input_sequences))).item()
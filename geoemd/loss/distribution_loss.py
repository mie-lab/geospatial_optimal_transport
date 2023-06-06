import torch
from torch.nn import MSELoss, CrossEntropyLoss


class DistributionMSE:
    def __init__(self) -> None:
        self.standard_mse = MSELoss()

    def __call__(self, a_in, b_in):
        # normalize a and b
        a = a_in.softmax(dim=-1)
        b = b_in.softmax(dim=-1)

        # apply standard MSE
        return self.standard_mse(a, b)


class StepwiseCrossentropy:
    def __init__(self) -> None:
        self.standard_loss = CrossEntropyLoss()

    def __call__(self, inputs, targets_raw):
        # convert targets into probabilities
        targets = targets_raw.softmax(dim=-1)
        # apply stepwise if predicting several steps --> not necessary!
        # if inputs.dim() > 2:
        #     steps_ahead = inputs.size()[1]
        #     result = torch.empty(steps_ahead)
        #     for i in range(steps_ahead):
        #         result[i] = self.standard_loss(inputs[:, i], targets[:, i])
        #     print(torch.mean(result))
        # Somehow it's magically taking the correct cross entropy (with the
        # swapped axes) as a metric when adding it to the collection
        if inputs.dim() > 2:
            return self.standard_loss(
                torch.swapaxes(inputs, 1, 2), torch.swapaxes(targets, 1, 2)
            )
        else:
            return self.standard_loss(inputs, targets)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# let's assume a single state for simplicity
state = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
# first action is correct
ground_truth = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=False) 
mse_loss =nn.MSELoss()
num_iterations = 40
USE_DESCENT = False # True for gradient descent, False for gradient ascent

# create simple neural network
input_dim = state.size()[0]
hidden_dim = 20
output_dim = ground_truth.size()[0]
layers = [
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
]
model = nn.Sequential(*layers)
#optimizer = optim.SGD(model.parameters(), lr=0.05)
optimizer = optim.Adam(model.parameters(), lr=0.01)
print(model)

# train neural network for 1 state and 4 actions, where
# the goal is to increase the probability of first action
for i in range(0, num_iterations):
    prediction = model(state) # forward pass
    output = torch.distributions.Categorical(logits=prediction)
    a = output.sample()
    if USE_DESCENT:
        loss = mse_loss(output.probs, ground_truth) # positive loss for gradient descent
        signal = "error"
    else:
        correct_action = torch.argmax(ground_truth).item()
        reward = 1.0 if a.item() == correct_action else 0.0  # simple reward function
        loss = -output.log_prob(a) * reward  # negative log-probability (maximise the action probability)
        signal = "reward"

    signal_value = loss.detach().numpy()
    print("i=%s, a=%s, pred=%s, probs=%s, %s=%s" % \
	        (i, a, prediction.detach(), output.probs.detach(), signal, signal_value))

    optimizer.zero_grad() # reset gradients
    loss.backward() # compute gradients
    #print("state.grad="+str(state.grad))
    #print("ground_truth.grad="+str(ground_truth.grad))
    optimizer.step() # update parameters
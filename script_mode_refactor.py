import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import argparse

def one_hot_and_tensify(position):
    one_hot = [0 if x != position else 1 for x in range(9)]
    return torch.tensor(one_hot, dtype=torch.int64, device=device)

def make_reward_tensor(reward):
    #one_hot = [0 if x != position else reward for x in range(9)]
    return torch.tensor(reward, dtype = torch.int64, device=device)

class MemoryDataset(Dataset):
    def __init__(self, raw_memory):
        self.raw_memory = raw_memory

    def __len__(self):
        return len(self.raw_memory)

    def __getitem__(self, idx):
        entry = self.raw_memory[idx]
        state = torch.tensor(list(entry[0]), dtype=torch.float32)
        action = one_hot_and_tensify(entry[1])
        next_state = torch.tensor(list(entry[2]), dtype=torch.float32)
        reward = make_reward_tensor(entry[3])
        return state, action, next_state, reward


def train():

    # import data and format (train & test)
    '''
    with open('drive/My Drive/Projects/TTT/data/memory_0.pkl', 'rb') as file:
        raw_memory = pkl.load(file)
    '''
    raw_train_memory = get_train_data(args.train)
    raw_test_memory = get_test_data(args.test)

    # argparse arguments :
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    GAMMA = args.gamma
    epochs = args.epochs
    model_dir = args.model_dir


    # logger

    # training dataloader
    dataset = MemoryDataset(raw_train_memory)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # get model()
    policy_net = get_model().to(device)
    target_net = get_model().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # criterion
    criterion = nn.SmoothL1Loss()

    # optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)


    epochs = 5
    for epoch in range(epochs) :

        for state_batch, action_batch, next_state_batch, reward_batch in dataloader:

            ## update 2nd model?

            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            reward_batch = reward_batch.to(device)

            # Compute a mask of non-final states
            # (a final state's next state would've been after the simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=device, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in next_state_batch if s is not None], dim = 0)

            actions_taken = action_batch.argmax(dim=1, keepdim=True)


            # y_hat
            state_action_values = policy_net(state_batch).gather(1, actions_taken)


            ''' Next states values calculated '''
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            state_action_values = state_action_values.squeeze(1)

            # Compute Huber loss
            loss = criterion(state_action_values, expected_state_action_values)

            #print(f'Loss is {loss}')

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()


    # Evaluate on test set 
    # need to wrap loss calculation before this for readability
    '''
    with torch.no_grad():
        y = model(x_test.float()).flatten()
        mse = ((y - y_test) ** 2).sum() / y_test.shape[0]
    print("\nTest MSE:", mse.numpy())


    '''


    torch.save(model.state_dict(), model_dir + "/model.pth")

    #PyTorch requires that the inference script must
    # be in the .tar.gz model file and Step Functions SDK doesn't do this.
    inference_code_path = model_dir + "/code/"

    if not os.path.exists(inference_code_path):
        os.mkdir(inference_code_path)
        logger.info("Created a folder at {}!".format(inference_code_path))

    shutil.copy("train_deploy_pytorch_without_dependencies.py", inference_code_path)
    shutil.copy("pytorch_model_def.py", inference_code_path)
    logger.info("Saving models files to {}".format(inference_code_path))

if __name__ == "__main__":

    args, _ = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train()


'''
[ml.m6i.xlarge, ml.trn1.32xlarge, ml.p2.xlarge, 
ml.m5.4xlarge, ml.m4.16xlarge, ml.m6i.12xlarge, 
ml.p5.48xlarge, ml.m6i.24xlarge, ml.p4d.24xlarge, 
ml.g5.2xlarge, ml.c5n.xlarge, ml.p3.16xlarge, ml.m5.large, 
ml.m6i.16xlarge, ml.p2.16xlarge, ml.g5.4xlarge, ml.c4.2xlarge, 
ml.c5.2xlarge, ml.c6i.32xlarge, ml.c4.4xlarge, ml.c6i.xlarge, 
ml.g5.8xlarge, ml.c5.4xlarge, ml.c6i.12xlarge, ml.c5n.18xlarge, 
ml.g4dn.xlarge, ml.c6i.24xlarge, ml.g4dn.12xlarge, ml.c4.8xlarge, 
ml.g4dn.2xlarge, ml.c6i.2xlarge, ml.c6i.16xlarge, ml.c5.9xlarge, 
ml.g4dn.4xlarge, ml.c6i.4xlarge, ml.c5.xlarge, ml.g4dn.16xlarge, 
ml.c4.xlarge, ml.trn1n.32xlarge, ml.g4dn.8xlarge, ml.c6i.8xlarge, 
ml.g5.xlarge, ml.c5n.2xlarge, ml.g5.12xlarge, ml.g5.24xlarge, 
ml.c5n.4xlarge, ml.trn1.2xlarge, ml.c5.18xlarge, ml.p3dn.24xlarge, 
ml.m6i.2xlarge, ml.g5.48xlarge, ml.g5.16xlarge, ml.p3.2xlarge, 
ml.m6i.4xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.c5n.9xlarge, 
ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, 
ml.m6i.8xlarge, ml.m6i.large, ml.p2.8xlarge, ml.m5.2xlarge, 
ml.m6i.32xlarge, ml.p4de.24xlarge, ml.p3.8xlarge, ml.m4.4xlarge]
'''
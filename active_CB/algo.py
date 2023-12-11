import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler
import torch.nn as nn
import torch.optim as optim
from ucimlrepo import fetch_ucirepo 
import argparse

class NeuralNetwork(nn.Module):
    def __init__(self, context_dim, num_arms, action_dim, hidden_size):
        super(NeuralNetwork, self).__init__()
        input_dim = context_dim + num_arms*action_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, context, action):
        combined = torch.cat((context, action), dim=1)
        hidden1 = torch.relu(self.fc1(combined))
        reward = torch.sigmoid(self.fc2(hidden1))
        return torch.reshape(reward,(-1,))

class NeuralNetwork2(nn.Module):
    def __init__(self, context_dim, num_arms, action_dim, hidden_size):
        super(NeuralNetwork, self).__init__()
        input_dim = context_dim + num_arms*action_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, context, action):
        combined = torch.cat((context, action), dim=1)
        hidden1 = torch.relu(self.fc1(combined))
        hidden2 = torch.relu(self.fc2(hidden1))
        reward = torch.sigmoid(self.fc3(hidden2))
        return torch.reshape(reward,(-1,))

def compute_alphas(r_t, x, a, Z, state_dim, num_states, action_dim, num_arms, set_A, dataset, t, beta, X, y, idx, args):
    arms = set()
    losses = []
    slack = 1e-4
    if args.dataset == 'iris':
        r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
    else:
        r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)

    r_ = lambda x, a : r(x, ohe(a, num_arms))
    optimizer = optim.Adam(r.parameters(), lr=0.05)
    a_stars = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = (Z[:t] * torch.square(dataset[:t] - r_(x[:t], a[:t]))).sum()
        if loss <= beta:
            a_star = -1
            max_reward = -1
            for arm in set_A:
                reward = r_t(x[t].unsqueeze(0), ohe(arm.unsqueeze(0), len(set_A)))
                if reward > max_reward:
                    max_reward = reward
                    a_star = arm
            arms.add(a_star.item())
            a_stars.append(a_star.item())
        loss.backward()
        optimizer.step()

    for arm_a in set_A:
        if arm_a.item() in a_stars:
            continue
        alpha1 = torch.tensor(1., requires_grad=True)
        alpha2 = torch.tensor(1., requires_grad=True)
        alpha3 = torch.tensor(1., requires_grad=True)
        optimizer_alphas = optim.Adam([alpha1, alpha2, alpha3], lr=0.02)
       
        if args.dataset == 'iris':
            r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
        else:
            r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)

        r_ = lambda x, a : r(x, ohe(a, num_arms))
        optimizer = optim.Adam(r.parameters(), lr=0.05)
    
        set_B = [b for b in set_A if b != arm_a]
        tensor_a = arm_a.unsqueeze(0)
        tensor_b1 = set_B[0].unsqueeze(0)
        tensor_b2 = set_B[1].unsqueeze(0)
       
        if args.dataset == 'iris':
            K = 10
            J = 60
        else:
            K = 15
            J = 70

        for k in range(K):
            for j in range(J):
                term1 = (Z[:t] * torch.square(dataset[:t] - r_(x[:t], a[:t]))).sum() - beta
                term2 = r_(x[t].unsqueeze(0), tensor_a) - r_(x[t].unsqueeze(0), tensor_b1) + slack
                term3 = r_(x[t].unsqueeze(0), tensor_a) - r_(x[t].unsqueeze(0), tensor_b2) + slack
                if term2 > 0 and term3 > 0 and term1 <= 0:
                    arms.add(arm_a)
                loss = alpha1 * term1 - alpha2 * term2 - alpha3 * term3 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            term1 = (Z[:t] * torch.square(dataset[:t] - r_(x[:t], a[:t]))).sum() - beta
            term2 = r_(x[t].unsqueeze(0), tensor_a) - r_(x[t].unsqueeze(0), tensor_b1)
            term3 = r_(x[t].unsqueeze(0), tensor_a) - r_(x[t].unsqueeze(0), tensor_b2)

            loss_alphas = -(alpha1 * term1 - alpha2 * term2 - alpha3 * term3)
            optimizer_alphas.zero_grad()
            loss_alphas.backward()
            optimizer_alphas.step()
            
            alpha1.data = torch.clip(alpha1.data, min=0)
            alpha2.data = torch.clip(alpha2.data, min=0)
            alpha3.data = torch.clip(alpha3.data, min=0)
            if k == 9:
                diff = []
                for ctx in np.arange(num_states):
                    for arm in set_A:
                        x_ = X[ctx].unsqueeze(0)
                        a_ = arm.unsqueeze(0)
                        target = 1. if arm == y[ctx] else 0
                        diff.append(np.absolute(((r_(x_, a_) - target)).item()))
                ret = np.mean(diff)
   
    diff = []
    for ctx in np.arange(num_states):
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = arm.unsqueeze(0)
            target = 1. if arm == y[ctx] else 0
            diff.append(np.absolute(((r_(x_, a_) - target)).item()))
    ret = np.mean(diff)
    if len(arms) == 0:
        arms.add(np.random.choice(set_A))
    return torch.tensor(list(arms)), ret

def compute_sup(X, A, Z, state_dim, num_states, action_dim, num_arms, dataset, set_A_t, t, beta, args):
    tensor_x = X[t].unsqueeze(0)
    diff = []
    losses = []
    for arm_a in set_A_t:
        loss_list = []
        alpha1 = torch.tensor(1., requires_grad=True)
        alpha2 = torch.tensor(1., requires_grad=True)
        optimizer_alphas = optim.SGD([alpha1, alpha2], lr=0.01)
        
        if args.dataset == 'iris':
            r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
            r_ = NeuralNetwork(state_dim, num_arms, action_dim, 64)
        else:
            r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
            r_ = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
        
        optimizer = optim.SGD(r.parameters(), lr=0.03)
        optimizer_ = optim.SGD(r_.parameters(), lr=0.03)

        tensor_a = ohe(arm_a.unsqueeze(0), num_arms)
        for k in range(10):
            for j in range(60):
                obj = torch.square(r(tensor_x, tensor_a) - r_(tensor_x, tensor_a))
                constr1 = (Z[:t] * torch.square(r(X[:t], ohe(A[:t], num_arms)) - dataset[:t])).sum() - beta
                constr2 = (Z[:t] * torch.square(r_(X[:t], ohe(A[:t], num_arms)) - dataset[:t])).sum() - beta
            
                if constr1 <= 0 and constr2 <= 0:
                    diff.append(r(tensor_x, tensor_a) - r_(tensor_x, tensor_a))
                loss = -obj + alpha1 * constr1 + alpha2 * constr2
                optimizer.zero_grad()
                optimizer_.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_.step()
            obj = torch.square(r(tensor_x, tensor_a) - r_(tensor_x, tensor_a))
            constr1 = (Z[:t] * torch.square(r(X[:t], ohe(A[:t], num_arms)) - dataset[:t])).sum() - beta
            constr2 = (Z[:t] * torch.square(r_(X[:t], ohe(A[:t], num_arms)) - dataset[:t])).sum() - beta

            optimizer_alphas.zero_grad()
            loss_alphas = obj - alpha1 * constr1 - alpha2 * constr2

            loss_alphas.backward()
            optimizer_alphas.step()
            
            alpha1.data = torch.clip(alpha1.data, min=0)
            alpha2.data = torch.clip(alpha2.data, min=0)
            
    sup = max(diff) if diff != [] else 0
    return sup

def compute_p(r_t, A, A_t, gamma, x):
    x = x.unsqueeze(0)
    p = np.zeros(len(A_t))
    a_star_t = -1
    max_reward = -1
    for a in A_t:
        reward = r_t(x, ohe(a.unsqueeze(0), len(A)))
        if reward > max_reward:
            max_reward = reward
            a_star_t = a
            
    for i in range(len(A_t)):
        a = A_t[i]
        if a.item() != a_star_t.item():
            p[i] = 1/(len(A_t) + gamma * (max_reward - r_t(x, ohe(a.unsqueeze(0), len(A)))))
    i_star = (A_t == a_star_t.item()).nonzero(as_tuple=True)[0]
    p[i_star] = 1 - p.sum()
    p = p/p.sum()
    return p

def ohe(state, total_states):
    return torch.eye(total_states)[state.to(torch.int)]

def model_reward(beta, state_dim, num_states, action_dim, num_arms, T, X, y, args):    
    run = wandb.init()
    x = torch.zeros((T, state_dim))
    a = torch.zeros(T)
    Z = torch.zeros(T)
    w = torch.zeros(T)
    targets = torch.zeros(T)
    dataset = torch.zeros(T)
    set_A = torch.arange(num_arms)
    losses = []
    returns = []
    diff_list = []
    err_list = []
    candidate_size = []
    include_optimal = []

    if args.dataset == 'iris':
        r_t = NeuralNetwork(state_dim, num_arms, action_dim, 64)
    else:
        r_t = NeuralNetwork2(state_dim, num_arms, action_dim, 64)

    optimizer = optim.SGD(r_t.parameters(), 0.05) 

    #Iteration 0
    idx = np.random.randint(num_states)
    x[0] = X[idx]
    x_0 = x[0].unsqueeze(0)
    set_A_t = set_A

    best_arm = y[idx]
    is_in = 1
    include_optimal.append(is_in)
    candidate_size.append(len(set_A_t))
    p = [1/len(set_A_t)] * len(set_A_t)
    a[0] = np.random.choice(set_A_t, p=p)
    Z[0] = 1
    at = int(a[0])
    a_ohe = ohe(a[0].unsqueeze(0), num_arms)

    targets[0] = 1 if at == best_arm else 0 
    returns.append(targets[0].item())

    for i in range(20):        
        pref_loss = torch.square(r_t(x_0, a_ohe) - targets[0]).sum()
        optimizer.zero_grad()
        pref_loss.backward()
        optimizer.step()
    losses.append(pref_loss.item())
    
    diff = []
    for ctx in np.arange(num_states):
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = ohe(arm.unsqueeze(0), num_arms)
            target = 1. if arm == y[ctx] else 0
            diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
    diff_list.append(np.mean(diff))

    for t in range(1, T):
        idx = np.random.randint(num_states)
        x[t] = X[idx]
        best_arm = y[idx]

        set_A_t, err = compute_alphas(r_t, x, a, Z, state_dim, num_states, action_dim, num_arms, set_A, dataset, t, beta, X, y, idx, args)
        candidate_size.append(len(set_A_t))
        is_in = 1 if best_arm in set_A_t else 0

        include_optimal.append(is_in)
        err_list.append(err)

        Z[t] = 1 if len(set_A_t) > 1 else 0

        if Z[t] == 1:
            lmbda = 1 if torch.sum(Z[:t] * w[:t]) >= np.sqrt(num_arms*T/beta) else 0
            if lmbda == 0:
                w[t] = compute_sup(x, a, Z, state_dim, num_states, action_dim, num_arms, dataset, set_A_t, t, beta, args)
                p = [1/len(set_A_t)] * len(set_A_t)
                a[t] = np.random.choice(set_A_t, p=p)

            else:
                gamma = np.sqrt(num_arms*T/beta)
                p = compute_p(r_t, set_A, set_A_t, gamma, x[t])
                a[t] = np.random.choice(set_A_t, p=p)

            a_ohe = ohe(a[:t+1], num_arms)
            at = int(a[t])
            targets[t] = 1. if at == best_arm else 0.

            for i in range(50): 
                pref_loss = torch.square(r_t(x[:t+1], a_ohe) - targets[:t+1]).sum()
                optimizer.zero_grad()
                pref_loss.backward()
                optimizer.step()
            dataset[t] = r_t(x[:t+1], a_ohe)[-1].item()
            losses.append(pref_loss.item())
            returns.append(targets[t].item())
        else:
            a[t] = set_A_t[0]
            at = int(a[t])
            targets[t] = 1. if at == best_arm else 0.
            returns.append(targets[t].item())
            a_ohe = ohe(a[:t+1], num_arms)
            dataset[t] = r_t(x[:t+1], a_ohe)[-1].item()
        diff = []
        for ctx in np.arange(num_states):
            for arm in set_A:
                x_ = X[ctx].unsqueeze(0)
                a_ = ohe(arm.unsqueeze(0), num_arms)
                target = 1. if arm == y[ctx] else 0
                diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
        diff_list.append(np.mean(diff))
        
    preds = [[],[]]
    for ctx in np.arange(num_states):
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = ohe(arm.unsqueeze(0), num_arms)
            target = 1. if arm == y[ctx] else 0
            preds[0].append(r_t(x_, a_))
            preds[1].append(target)
            diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
    return returns, Z, x, losses, diff_list, err_list, candidate_size, include_optimal, preds

## Preference
def compute_alphas_p(r_t, x, a, b, Z, state_dim, num_states, action_dim, num_arms, set_A, dataset, t, beta, X, y, args):
    arms = set()
    losses = []
    slack = 1e-4
    if args.dataset == 'iris':
        r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
    else:
        r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
        
    f = lambda x, a, b: r(x, ohe(a, num_arms)) - r(x, ohe(b, num_arms))
    f_t = lambda x, a, b: r_t(x, ohe(a, num_arms)) - r_t(x, ohe(b, num_arms))
    optimizer = optim.Adam(r.parameters(), lr=0.05)
    a_stars = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = (Z[:t] * torch.square(dataset[:t] - f(x[:t], a[:t], b[:t]))).sum()
        if loss <= beta:
            a_star = -1
            max_reward = -1
            for arm_a in set_A:
                cnt = 0
                for arm_b in set_A:
                    diff = f_t(x[t].unsqueeze(0), arm_a.unsqueeze(0), arm_b.unsqueeze(0))
                    if diff >= 0:
                        cnt += 1
                if cnt == num_arms:
                    arms.add(arm_a.item())
                    a_stars.append(arm_a.item())
        loss.backward()
        optimizer.step()

    for arm_a in set_A:
        if arm_a.item() in a_stars:
            continue

        alphas = []
        for _ in range(num_arms):
            alphas.append(torch.tensor(1., requires_grad=True))
        optimizer_alphas = optim.Adam(alphas, lr=0.02)

        if args.dataset == 'iris':
            r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
        else:
            r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
       
        f = lambda x, a, b : r(x, ohe(a, num_arms)) - r(x, ohe(b, num_arms))
        optimizer = optim.Adam(r.parameters(), lr=0.05)
    
        tensor_bs = [b.unsqueeze(0) for b in set_A if b != arm_a]
        tensor_a = arm_a.unsqueeze(0)

        for k in range(10):
            for j in range(60):
                func_space_constr = (Z[:t] * torch.square(dataset[:t] - f(x[:t], a[:t], b[:t]))).sum() - beta
                terms = [func_space_constr]
                satisfied = [(func_space_constr <= 0).item()]
                for l in range(num_arms-1):
                    constr = f(x[t].unsqueeze(0), tensor_a, tensor_bs[l]) + slack
                    terms.append(constr)
                    satisfied.append((constr > 0).item())
                if np.all(satisfied):
                    arms.add(arm_a)
                loss = (alphas[0] * terms[0]).flatten()
                for m in range(1, len(alphas)):
                    loss -= (alphas[m] * terms[m])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            func_space_constr = (Z[:t] * torch.square(dataset[:t] - f(x[:t], a[:t], b[:t]))).sum() - beta
            terms = [func_space_constr]
            for l in range(num_arms-1):
                constr = f(x[t].unsqueeze(0), tensor_a, tensor_bs[l]) + slack
                terms.append(constr)

            loss_alphas = (-alphas[0] * terms[0]).flatten()
            for m in range(1, len(alphas)):
                loss_alphas += (alphas[m] * terms[m])

            optimizer_alphas.zero_grad()
            loss_alphas.backward()
            optimizer_alphas.step()
            
            for n in range(len(alphas)):
                alphas[n].data = torch.clip(alphas[n].data, min=0)

    diff = []
    for _ in range(100):
        ctx = np.random.randint(num_states)
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = arm.unsqueeze(0)
            target = 1. if arm == y[ctx] else 0
            diff.append(np.absolute(((r(x_, ohe(a_, num_arms)) - target)).item()))
    ret = np.mean(diff)
    if len(arms) == 0:
        arms.add(np.random.choice(set_A))
    return torch.tensor(list(arms)), ret

def compute_sup_p(X, A, B, Z, state_dim, num_states, action_dim, num_arms, dataset, set_A_t, t, beta, args):
    tensor_x = X[t].unsqueeze(0)
    diff = []
    losses = []
    for arm_a in set_A_t:
        for arm_b in set_A_t:
            loss_list = []
            alpha1 = torch.tensor(1., requires_grad=True)
            alpha2 = torch.tensor(1., requires_grad=True)
            optimizer_alphas = optim.Adam([alpha1, alpha2], lr=0.01)
            
            if args.dataset == 'iris':
                r = NeuralNetwork(state_dim, num_arms, action_dim, 64)
                r_ = NeuralNetwork(state_dim, num_arms, action_dim, 64)
            else:
                r = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
                r_ = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
           
            f = lambda x, a, b: r(x, ohe(a, num_arms)) - r(x, ohe(b, num_arms))
            f_ = lambda x, a, b: r_(x, ohe(a, num_arms)) - r_(x, ohe(b, num_arms))

            optimizer = optim.Adam(r.parameters(), lr=0.03)
            optimizer_ = optim.Adam(r_.parameters(), lr=0.03)
            
            for k in range(10):
                for j in range(60):
                    obj = torch.square(f(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)) - f_(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)))
                    constr1 = (Z[:t] * torch.square(f(X[:t], A[:t], B[:t]) - dataset[:t])).sum() - beta
                    constr2 = (Z[:t] * torch.square(f_(X[:t], A[:t], B[:t]) - dataset[:t])).sum() - beta
                
                    if constr1 <= 0 and constr2 <= 0:
                        diff.append(f(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)) - f_(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)))
                    loss = -obj + alpha1 * constr1 + alpha2 * constr2
                    optimizer.zero_grad()
                    optimizer_.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_.step()
                obj = torch.square(f(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)) - f_(tensor_x, arm_a.unsqueeze(0), arm_b.unsqueeze(0)))
                constr1 = (Z[:t] * torch.square(f(X[:t], A[:t], B[:t]) - dataset[:t])).sum() - beta
                constr2 = (Z[:t] * torch.square(f_(X[:t], A[:t], B[:t]) - dataset[:t])).sum() - beta

                optimizer_alphas.zero_grad()
                loss_alphas = obj - alpha1 * constr1 - alpha2 * constr2

                loss_alphas.backward()
                optimizer_alphas.step()
                
                alpha1.data = torch.clip(alpha1.data, min=0)
                alpha2.data = torch.clip(alpha2.data, min=0)
                
    sup = max(diff) if diff != [] else 0
    return sup

def compute_p(r_t, A, A_t, gamma, x):
    x = x.unsqueeze(0)
    p = np.zeros(len(A_t))
    a_star_t = -1
    max_reward = -1
    for a in A_t:
        reward = r_t(x, ohe(a.unsqueeze(0), len(A)))
        if reward > max_reward:
            max_reward = reward
            a_star_t = a
            
    for i in range(len(A_t)):
        a = A_t[i]
        if a.item() != a_star_t.item():
            p[i] = 1/(len(A_t) + gamma * (max_reward - r_t(x, ohe(a.unsqueeze(0), len(A)))))
    i_star = (A_t == a_star_t.item()).nonzero(as_tuple=True)[0]
    p[i_star] = 1 - p.sum()
    p = p/p.sum()
    return p

def model_preference(beta, state_dim, num_states, action_dim, num_arms, T, X, y, args):    
    run = wandb.init()
    x = torch.zeros((T, state_dim))
    a = torch.zeros(T)
    b = torch.zeros(T)
    Z = torch.zeros(T)
    w = torch.zeros(T)
    targets_a = torch.zeros(T)
    targets_b = torch.zeros(T)
    dataset = torch.zeros(T)
    set_A = torch.arange(num_arms)
    losses = []
    returns = []
    diff_list = []
    err_list = []
    candidate_size = []
    include_optimal = []

    if args.dataset == 'iris':
        r_t = NeuralNetwork(state_dim, num_arms, action_dim, 64)
        optimizer = optim.SGD(r_t.parameters(), 0.05)
    else:
        r_t = NeuralNetwork2(state_dim, num_arms, action_dim, 64)
        optimizer = optim.Adam(r_t.parameters(), 0.03)

    #Iteration 0
    idx = np.random.randint(num_states)
    x[0] = X[idx]
    x_0 = x[0].unsqueeze(0)
    set_A_t = set_A

    best_arm = y[idx]
    is_in = 1
    include_optimal.append(is_in)
    candidate_size.append(len(set_A_t))
    p = [1/len(set_A_t)] * len(set_A_t)
    a[0] = np.random.choice(set_A_t, p=p)
    b[0] = np.random.choice(set_A_t, p=p)
    Z[0] = 1
    at = int(a[0])
    bt = int(b[0])

    a_ohe = ohe(a[0].unsqueeze(0), num_arms)
    b_ohe = ohe(b[0].unsqueeze(0), num_arms)

    targets_a[0] = 1 if at == best_arm else 0 
    targets_b[0] = 1 if bt == best_arm else 0 

    returns.append((targets_a[0]-targets_b[0]).item())

    for i in range(20):        
        pref_loss = torch.square(f_t(x_0, a_ohe, b_ohe) - (targets_a[0]-targets_b[0])).sum()
        optimizer.zero_grad()
        pref_loss.backward()
        optimizer.step()
    losses.append(pref_loss.item())
    
    diff = []
    for ctx in np.arange(num_states):
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = ohe(arm.unsqueeze(0), num_arms)
            target = 1. if arm == y[ctx] else 0
            diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
    diff_list.append(np.mean(diff))

    for t in range(1, T):
        idx = np.random.randint(num_states)
        x[t] = X[idx]
        best_arm = y[idx]

        set_A_t, err = compute_alphas_p(r_t, x, a, b, Z, state_dim, num_states, action_dim, num_arms, set_A, dataset, t, beta, X, y, idx, args)
        candidate_size.append(len(set_A_t))
        is_in = 1 if best_arm in set_A_t else 0

        include_optimal.append(is_in)
        err_list.append(err)

        if args.query == 'passive':
            Z[t] = 1
        else:
            Z[t] = 1 if len(set_A_t) > 1 else 0

        if Z[t] == 1:
            lmbda = 1 if torch.sum(Z[:t] * w[:t]) >= np.sqrt(num_arms*T/beta) else 0
            if lmbda == 1 or args.query == 'passive':
                gamma = np.sqrt(num_arms*T/beta)
                p = compute_p(r_t, set_A, set_A_t, gamma, x[t])
                a[t] = np.random.choice(set_A_t, p=p)
                b[t] = np.random.choice(set_A_t, p=p)
            else:
                w[t] = compute_sup_p(x, a, b, Z, state_dim, num_states, action_dim, num_arms, dataset, set_A_t, t, beta, args)
                p = [1/len(set_A_t)] * len(set_A_t)
                a[t] = np.random.choice(set_A_t, p=p)
                b[t] = np.random.choice(set_A_t, p=p)

            a_ohe = ohe(a[:t+1], num_arms)
            b_ohe = ohe(b[:t+1], num_arms)
            at = int(a[t])
            bt = int(b[t])  

            targets_a[t] = 1. if at == best_arm else 0.
            targets_b[t] = 1. if bt == best_arm else 0.

            for i in range(70): 
                pref_loss = torch.square(f_t(x[:t+1], a_ohe, b_ohe) - (targets_a[:t+1]-targets_b[:t+1])).sum()
                optimizer.zero_grad()
                pref_loss.backward()
                optimizer.step()
            dataset[t] = f_t(x[:t+1], a_ohe, b_ohe)[-1].item()
            losses.append(pref_loss.item())
            returns.append((targets_a[t]-targets_b[t]).item())

        else:
            a[t] = set_A_t[0]
            b[t] = set_A_t[0]
            at = int(a[t])
            bt = int(b[t])
            targets_a[t] = 1. if at == best_arm else 0.
            targets_b[t] = 1. if bt == best_arm else 0.
            returns.append((targets_a[t]-targets_b[t]).item())
            a_ohe = ohe(a[:t+1], num_arms)
            b_ohe = ohe(b[:t+1], num_arms)
            dataset[t] = f_t(x[:t+1], a_ohe, b_ohe)[-1].item()

        diff = []
        for _ in range(100):
            ctx = np.random.randint(num_states)
            for arm in set_A:
                x_ = X[ctx].unsqueeze(0)
                a_ = ohe(arm.unsqueeze(0), num_arms)
                target = 1. if arm == y[ctx] else 0
                diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
        diff_list.append(np.mean(diff))
        
    preds = [[],[]]
    for _ in range(100):
        ctx = np.random.randint(num_states)
        for arm in set_A:
            x_ = X[ctx].unsqueeze(0)
            a_ = ohe(arm.unsqueeze(0), num_arms)
            target = 1. if arm == y[ctx] else 0
            preds[0].append(r_t(x_, a_))
            preds[1].append(target)
            diff.append(np.absolute(((r_t(x_, a_) - target)).item()))
    return returns, Z, x, losses, diff_list, err_list, candidate_size, include_optimal, preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris', help='Name of the dataset (iris/car)')
    parser.add_argument('--query', type=str, default='active', help='Query type (active/passive)')
    parser.add_argument('--model', type=str, default='preference', help='Model type (reward/preference)')

    torch.manual_seed(233)
    np.random.seed(233) 
    args = parser.parse_args()
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    min_max_scaler = MinMaxScaler()
    encoder = LabelEncoder()

    if args.dataset == 'iris':
        iris = fetch_ucirepo(id=53) 
        X = iris.data.features 
        y = iris.data.targets 
        X_copy = X.copy()
        X_copy.iloc[:, [0, 1, 2, 3]] = min_max_scaler.fit_transform(X_copy.iloc[:, [0, 1, 2, 3]])
        X_tensor = torch.tensor(X_copy.values)
        X_tensor = X_tensor.to(torch.float32)
        encoded_y = encoder.fit_transform(y.values.reshape(-1,))
        A_tensor = torch.tensor(encoded_y)
        num_states = 150
        state_dim = 4
        num_arms = 3
    elif args.dataset == 'car':
        car_eval = fetch_ucirepo(id=19) 
        X = car_eval.data.features 
        y = car_eval.data.targets   
        X_copy = X.copy()
        for column in X_copy.columns:
            X_copy[column] = encoder.fit_transform(X_copy[column])
        X_copy.iloc[:, [0, 1, 2, 3, 4, 5]] = min_max_scaler.fit_transform(X_copy.iloc[:, [0, 1, 2, 3, 4, 5]])
        encoded_y = encoder.fit_transform(y.values.reshape(-1,))
        X_tensor = torch.tensor(X_copy.values)
        X_tensor = X_tensor.to(torch.float32)
        A_tensor = torch.tensor(encoded_y)
        num_states = 1728
        state_dim = 6
        num_arms = 4
    elif args.dataset == 'knowledge':
        knowledge = pd.read_csv('data/knowledge.csv')
        X = knowledge.iloc[:, :-1]
        y = knowledge.iloc[:, -1]
        X_tensor = torch.tensor(X.values)
        X_tensor = X_tensor.to(torch.float32)
        encoded_y = encoder.fit_transform(y.values.reshape(-1,))
        A_tensor = torch.tensor(encoded_y)
        num_states = 403
        state_dim = 5
        num_arms = 5

    T = 1000
    beta = 10
    action_dim = 1

    if args.model == 'reward':
        returns, Z, x, losses, diff_list, err_list, candidate_size, include_optimal, preds = model_reward(beta, state_dim, num_states, action_dim, num_arms, T, X_tensor, A_tensor, args) 

    else:
        returns, Z, x, losses, diff_list, err_list, candidate_size, include_optimal, preds = model_preference(beta, state_dim, num_states, action_dim, num_arms, T, X_tensor, A_tensor, args) 

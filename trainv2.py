"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    

    #tensorboard紀錄
    writer = SummaryWriter(opt.log_path)

    #創建環境跟神經網路DeepQNetwork()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    target_model = DeepQNetwork()

    optimizer = torch.optim.Adam(target_model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    
    for param in model.parameters():
        param.requires_grad = False
    
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0


    max_line= 100
    max_steps = 20000
    step = 0
    T0=True
    
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
        #換到評估模式 停止dropout and batchnormalization
        model.eval()

        with torch.no_grad():#停止autograd用以加速GPU
            predictions = model(next_states)[:, 0]#選出最佳下一步

        #換到訓練模式 啟用dropout and batchnormalization
        model.train()


        if random_action:#隨機選擇或選最佳
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        #把下一個狀態及現在的動作存起來
        next_state = next_states[index, :]
        
        action = next_actions[index]
        
        #算出reward and done
        reward, done = env.step(action, render=False)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        
        #把這round的state, reward, next_state, done記錄到memory裡
        replay_memory.append([state, reward, next_state, done])
        if(step>=max_steps):
            done=True
        if(T0):
            torch.save(model, "{}/tetris_0".format(opt.saved_path))
            print("save")
            T0=False
            
        if done:
            #結束時紀錄分數、消除行數
            step = 0
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if final_cleared_lines >= max_line:
                print("saved")
                max_line = final_cleared_lines
                torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            step += 1
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1
        #train

        #從mem隨機採取長度為replay_memory或opt.batch_size的資料
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)#從batch解壓縮
        state_batch = torch.stack(tuple(state for state in state_batch))#dim=0,把state疊成[state1, state2, state3.....]from state_batch
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])#從numpy轉成tensor 共同相同mem
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))#dim=0,把state疊成[state1, state2, state3.....]from next_state_batch

        if torch.cuda.is_available():#資料設為使用GPU運算
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()


        q_values = target_model(state_batch)
        target_model.eval()
        with torch.no_grad():
            next_prediction_batch = target_model(next_state_batch)

        target_model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        
        

        optimizer.zero_grad()#清空gradient
        
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()


        if epoch % 4 == 0:
            update_parameters(target_model, model)

        #輸出回合數及儲存model
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}, loss: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines, 
            loss))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Averge Score', final_score/epoch, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)
        writer.add_scalar('Train/Loss', loss, epoch - 1)

        if(epoch%300==0):
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
        if 0xFF == ord('q') or final_cleared_lines>=20000:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
            break


    torch.save(model, "{}/tetris".format(opt.saved_path))
    


if __name__ == "__main__":
    opt = get_args()
    train(opt)

# main.py
import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from tqdm import tqdm
from Model import MUTANT
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search


class ExpConfig:
    dataset = "SWaT"
    val = 0.35
    max_train_size = None
    train_start = 0
    max_test_size = None
    test_start = 0
    input_dim = get_data_dim(dataset)
    batch_size = 120
    out_dim = 5
    window_length = 20
    hidden_size = 100
    latent_size = 100
    N = 256
    lora_rank = 4
    lora_alpha = 8
    lora_layers = ['fc1', 'gc1']


def main():
    config = ExpConfig()
    save_path = 'model.pt'


    # 数据加载
    (train_data, _), (test_data, test_label) = get_data(config.dataset,
                                                        config.max_train_size,
                                                        config.max_test_size,
                                                        train_start=config.train_start,
                                                        test_start=config.test_start)

    # 数据分割后调整标签长度
    n = int(test_data.shape[0] * config.val)
    test_data, val_data = test_data[:-n], test_data[-n:]
    test_label, val_label = test_label[:-n], test_label[-n:]

    # 窗口处理函数
    def window_data(data):
        return data[np.arange(config.window_length)[None, :] +
                    np.arange(data.shape[0] - config.window_length)[:, None]]

    # 应用窗口处理并调整标签
    train_data = window_data(train_data)
    val_data = window_data(val_data)  # 形状变为 (n - w+1, w, f)
    val_label = val_label[config.window_length - 1:]  # 关键调整

    test_data = window_data(test_data)
    test_label = test_label[config.window_length - 1:]

    # 数据加载器
    w_size = config.input_dim * config.out_dim
    train_loader = get_loader(train_data, config.batch_size,
                              config.window_length, config.input_dim, True)
    val_loader = get_loader(val_data, config.batch_size,
                            config.window_length, config.input_dim, True)
    test_loader = get_loader(test_data, config.batch_size,
                             config.window_length, config.input_dim, False)

    #插入检查代码
    # 在数据预处理后添加
    print("Processed Data Shapes:")
    print(f"val_data: {val_data.shape}, val_label: {val_label.shape}")
    print(f"test_data: {test_data.shape}, test_label: {test_label.shape}")



    # 模型初始化
    model = MUTANT(config.input_dim, w_size, config.hidden_size,
                   config.latent_size, config.batch_size,
                   config.window_length, config.out_dim)

    # 加载检查点
    start_epoch, best_f1 = 0, -1
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, best f1 {checkpoint['best_f1']:.4f}")
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']

    # 优化器设置
    trainable_params = [p for n, p in model.named_parameters()
                        if any(l in n for l in config.lora_layers)]
    optimizer = optim.Adam(trainable_params, lr=0.01)

    # 训练循环
    for epoch in range(start_epoch, 10):
        model.train()
        pbar = tqdm(train_loader)
        for i, inputs in enumerate(pbar):
            optimizer.zero_grad()
            loss = model(inputs)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if i % config.N == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

        # 验证
        # 评估部分修改
        val_score = model.is_anomaly(val_loader)
        print(f"[Debug] val_score len: {len(val_score)}, val_label len: {len(val_label)}")
        t, th = bf_search(val_score, val_label, 700)


        # 保存最佳模型
        if t[0] > best_f1:
            best_f1 = t[0]
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_f1': best_f1,
                'lora_config': {
                    'rank': config.lora_rank,
                    'alpha': config.lora_alpha,
                    'layers': config.lora_layers
                }
            }, save_path)
            print(f"New best model saved at epoch {epoch} with f1 {best_f1:.4f}")



    # 最终测试
    model.load_state_dict(torch.load(save_path)['model_state'])
    test_score = model.is_anomaly(test_loader,
                                  len(test_loader),
                                  len(test_loader.dataset) % config.batch_size)
    t, th = bf_search(test_score, test_label[-len(test_score):], 700)

    # 保存结果
    results = pd.DataFrame({
        'Dataset': [config.dataset],
        'Threshold': [th],
        'TP': [t[3]], 'FP': [t[5]], 'FN': [t[6]],
        'Precision': [t[1]], 'Recall': [t[2]], 'F-Score': [t[0]]
    })

    try:
        existing = pd.read_csv('results.csv')
        results = pd.concat([existing, results])
    except FileNotFoundError:
        pass

    results.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
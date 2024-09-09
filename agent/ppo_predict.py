import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))

import torch
import pandas as pd
from actor import PtrNet1
from env_separatedPBS import PanelBlockShop
import csv
import time

def load_data(file_path):
    df = pd.read_excel(file_path)
    numeric_data = df.iloc[:, 3:13].values  # 크레인배재부터 론지수정까지의 데이터
    return torch.FloatTensor(numeric_data)

def load_model(model_path, params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PtrNet1(params).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict_actor'])
    model.eval()
    return model, device

def predict(model, data, device):
    with torch.no_grad():
        if len(data.shape) == 2:
            data = data.unsqueeze(0)

        data_normalized = data / data.amax(dim=(1, 2), keepdim=True).expand(data.shape[0], data.shape[1], data.shape[2])
        pred_seq, _, _ = model(data_normalized, device)
    return pred_seq.squeeze(0)

def evaluate_model(model_path, data, params):
    model, device = load_model(model_path, params)
    env = PanelBlockShop(num_process=params['num_of_process'], num_p1=params['num_p1'], num_of_blocks=data.shape[0])
    
    makespans = []
    for _ in range(1000):
        pred_seq = predict(model, data, device)
        makespan = env.calculate_makespan(data, pred_seq)
        makespans.append(makespan.item())
    
    max_makespan = max(makespans)
    min_makespan = min(makespans)
    avg_makespan = sum(makespans) / len(makespans)
    
    return max_makespan, min_makespan, avg_makespan

if __name__ == '__main__':
    num_of_process = 10  # 총 공정 수
    num_p1 = 3  # 두 번째 갈래로 분기되는 공정 수
    num_of_blocks = 1060  # 새로운 데이터의 블록 수

    params = {
        "num_of_process": num_of_process,
        "num_of_blocks": num_of_blocks,
        "n_embedding": 1024,
        "n_hidden": 512,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": 10,
        "T": 1.0,
        "decode_type": "sampling",
        "n_glimpse": 1,
        "n_process": 3,
        "num_p1": num_p1,
        # 아래 파라미터들은 예측 시에는 사용되지 않지만, 모델 구조의 일관성을 위해 포함
        "batch_size": 32,
        "epsilon": 0.1,
        "optimizer": "Adam",
        "lr": 1e-5,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "lr_decay_step": 2000,
        "load_model": True
    }

    data_path = r"C:\Users\User\Desktop\PBSgit\environment\판넬_데이터.xlsx"
    data = load_data(data_path)
    
    model_dir = r"C:\Users\User\Desktop\PBSgit\agent\result\model\ppo"
    steps = range(1000, 4000, 1000)
    results = []

    for step in steps:
        model_path = os.path.join(model_dir, f"0831_17_05_step{step}_act.pt")
        
        # 시간 측정 시작
        start_time = time.time()
        
        max_makespan, min_makespan, avg_makespan = evaluate_model(model_path, data, params)
        
        # 시간 측정 종료
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        print(f"Step {step}: {elapsed_minutes:.2f} minutes")
        
        results.append({
            "step": step,
            "max_makespan": max_makespan,
            "min_makespan": min_makespan,
            "avg_makespan": avg_makespan,
            "elapsed_time": elapsed_minutes  # 시간을 분 단위로 저장
        })

    # CSV 파일로 저장
    csv_file = r"C:\Users\User\Desktop\PBSgit\agent\result\model\ppo\makespan_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["step", "max_makespan", "min_makespan", "avg_makespan", "elapsed_time"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_file}")
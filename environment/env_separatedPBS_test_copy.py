from env_separatedPBS import PanelBlockShop
import numpy as np
import pandas as pd
import torch

# 테스트 파라미터 설정
num_process = 10  # 총 공정 수
num_p1 = 3  # 두 번째 갈래로 분기되는 공정 수
num_of_blocks = 50  # 각 배치의 블록 수
batch_size = 32  # 배치 크기

# PanelBlockShop 인스턴스 생성
pbs = PanelBlockShop(num_process=num_process, num_p1=num_p1, num_of_blocks=num_of_blocks)

# 데이터 생성 테스트 시작
print("데이터 생성 테스트 시작...")
try:
    generated_data = pbs.generate_data(batch_size=batch_size)
    print("데이터 생성 완료.")
    print(f"생성된 데이터의 shape: {generated_data.shape}")
    
    # 생성된 데이터의 일부를 출력해보기
    print("첫 번째 배치의 첫 번째 블록 데이터:")
    print(generated_data[0, 0, :])  # 첫 번째 배치의 첫 번째 블록에 대한 데이터 출력
    
    # 생성된 데이터를 pandas DataFrame으로 변환하여 저장
    df = pd.DataFrame(generated_data.reshape(-1, num_process), 
                      columns=[f'Process_{i+1}' for i in range(num_process)])
    
    # 각 블록의 타입 정보 추가 (이 부분은 실제 타입 정보를 사용하도록 수정해야 할 수 있습니다)
    # 각 블록의 타입 정보 추가
    df['타입'] = pbs.selected_types

    
    # 생성된 데이터를 엑셀 파일로 저장
    output_file = "generated_data_test.xlsx"
    df.to_excel(output_file, index=False)
    print(f"생성된 데이터가 '{output_file}' 파일에 저장되었습니다.")

    print(f"Total blocks: {batch_size * num_of_blocks}")
    print(f"Selected types length: {len(pbs.selected_types)}")
    print(f"First few selected types: {pbs.selected_types[:10]}")
    print(f"DataFrame shape: {df.shape}")
    
    if pbs.all_data is not None:
        print(f"All data keys: {list(pbs.all_data.keys())}")
        first_type = list(pbs.all_data.keys())[0]
        print(f"Sample data shape: {pbs.all_data[first_type]['0'].shape}")
    else:
        print("Warning: all_data is None. Make sure generate_data was called.")

    # Makespan 계산
    print("\nMakespan 계산 시작...")
    
    # 각 배치에 대한 시퀀스 생성 (0부터 num_of_blocks-1까지의 순서)
    sequences = [list(range(num_of_blocks)) for _ in range(batch_size)]
    
    # 각 배치에 대해 makespan 계산
    makespan_batch = []
    for i in range(batch_size):
        makespan = pbs.calculate_makespan(torch.FloatTensor(generated_data[i]), sequences[i])
        makespan_batch.append(makespan)
    
    makespan_batch = torch.stack(makespan_batch)
    
    print("Makespan 계산 완료.")
    print(f"Makespan 결과 shape: {makespan_batch.shape}")
    print("각 배치의 Makespan:")
    print(makespan_batch)
    
    print(f"\n평균 Makespan: {makespan_batch.mean().item():.2f}")
    print(f"최소 Makespan: {makespan_batch.min().item():.2f}")
    print(f"최대 Makespan: {makespan_batch.max().item():.2f}")

    # 특정 배치(예: 첫 번째 배치)에 대한 makespan 계산
    batch_index = 0
    specific_makespan = pbs.calculate_makespan(torch.FloatTensor(generated_data[batch_index]), sequences[batch_index])
    print(f"\n배치 {batch_index}의 Makespan: {specific_makespan.item():.2f}")

    # 특정 배치의 블록 타입 출력
    batch_types = pbs.selected_types[batch_index * num_of_blocks : (batch_index + 1) * num_of_blocks]
    print(f"배치 {batch_index}의 블록 타입:")
    print(batch_types)

except Exception as e:
    print(f"테스트 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

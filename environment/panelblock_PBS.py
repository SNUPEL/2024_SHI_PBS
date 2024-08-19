import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import pickle

class DataGenerator:
    def __init__(self, data_file=None, num_of_blocks=50, size=1, default_type_params=None):
        self.data_file = data_file
        self.num_of_blocks = num_of_blocks
        self.size = size
        self.type_params = self.initialize_type_params()
        self.correlation_matrices = {}
        self.type_counts = None
        self.stats_by_type = None

        # 데이터 초기화 (로드 또는 생성)
        self.initialize_data()

    def initialize_type_params(self):
        return {
            'TP_1': {
                'shape': [0.1052, 0.2015, 0.0534, 0.0534, 0.3823, 0.1911, 0.4037, 0.2665, 0.0115],
                'scale': [8.62, 17.30, 25.96, 25.96, 26.21, 22.89, 41.24, 44.31, 45.06]
            },
            'TP_10': {
                'shape': [0.3009, 0.3808, 0.4490, 0.4490, 0.2812, 0.1816, 0.3431, 0.3737, 0.2464],
                'scale': [12.26, 25.66, 37.27, 37.27, 34.42, 28.76, 52.56, 47.27, 37.03]
            },
            'TP_11': {
                'shape': [0.2005, 0.3489, 0.3052, 0.3052, 0.3085, 0.2105, 0.2257, 0.3383, 0.1660],
                'scale': [10.70, 21.91, 30.93, 30.93, 31.77, 26.38, 41.18, 39.72, 37.51]
            },
            'TP_12': {
                'shape': [0.1289, 0.1738, 0.2958, 0.2958, 0.1231, 0.0617, 0.0942, 0.0048, 0.0071],
                'scale': [17.26, 34.46, 54.12, 54.12, 38.70, 30.94, 74.07, 51.40, 34.70]
            },
            'TP_13': {
                'shape': [0.0, 0.0, 0.1003, 0.1003, 0.0, 0.0, 0.0405, 0.0, 0.0],
                'scale': [13.0, 25.0, 49.75, 49.75, 65.0, 30.0, 43.21, 33.7, 39.7]
            },
            'TP_14': {
                'shape': [0.1551, 0.1582, 0.0325, 0.0325, 0.00000001, 0.0510, 0.00000001, 0.00000001, 0.00000001],
                'scale': [12.85, 27.75, 47.10, 47.10, 45.0, 41.20, 41.5, 35.6, 42.9]
            },
            'TP_15': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [8.0, 15.0, 15.0, 15.0, 20.0, 20.0, 31.0, 32.9, 38.4]
            },
            'TP_16': {
                'shape': [0.2289, 0.1356, 0.3717, 0.3717, 0.1052, 0.00000001, 0.0504, 0.0862, 0.1291],
                'scale': [9.41, 16.51, 19.51, 19.51, 21.54, 20.0, 32.13, 30.95, 35.05]
            },
            'TP_17': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [8.0, 15.0, 15.0, 15.0, 25.0, 20.0, 41.5, 32.9, 38.4]
            },
            'TP_18': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [8.0, 15.0, 15.0, 15.0, 25.0, 20.0, 41.5, 32.9, 38.4]
            },
            'TP_19': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [8.0, 12.0, 20.0, 20.0, 20.0, 20.0, 38.0, 27.3, 29.0]
            },
            'TP_2': {
                'shape': [0.3875, 0.4995, 0.5201, 0.5201, 0.1688, 0.2867, 0.2913, 0.3558, 0.2203],
                'scale': [14.75, 30.78, 45.53, 45.53, 42.41, 31.22, 74.16, 60.45, 35.76]
            },
            'TP_20': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1134, 0.0, 0.0],
                'scale': [15.0, 30.0, 37.0, 37.0, 35.0, 30.0, 30.80, 35.9, 43.4]
            },
            'TP_21': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [8.0, 12.0, 20.0, 20.0, 25.0, 20.0, 34.5, 27.8, 29.9]
            },
            'TP_22': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [12.0, 30.0, 40.0, 40.0, 35.0, 30.0, 62.5, 45.1, 29.4]
            },
            'TP_23': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [11.0, 25.0, 35.0, 35.0, 35.0, 30.0, 62.5, 45.1, 29.4]
            },
            'TP_24': {
                'shape': [0.0966, 0.3169, 0.3456, 0.3456, 0.1437, 0.0966, 0.1533, 0.1192, 0.1919],
                'scale': [8.46, 13.49, 24.38, 24.38, 24.75, 23.64, 40.18, 28.35, 30.52]
            },
            'TP_3': {
                'shape': [0.0, 0.0, 0.1682, 0.1682, 0.0, 0.2027, 0.0, 0.0, 0.0],
                'scale': [13.0, 35.0, 88.74, 88.74, 45.0, 36.74, 157.0, 88.0, 32.7]
            },
            'TP_4': {
                'shape': [0.0, 0.0, 0.0104, 0.0104, 0.0, 0.0555, 0.0, 0.0, 0.0],
                'scale': [20.0, 50.0, 90.66, 90.66, 45.0, 41.60, 132.5, 88.0, 32.7]
            },
            'TP_5': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [15.0, 35.0, 98.0, 98.0, 45.0, 30.0, 87.0, 70.5, 33.8]
            },
            'TP_6': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [18.0, 45.0, 100.0, 100.0, 45.0, 45.0, 73.0, 50.0, 33.6]
            },
            'TP_7': {
                'shape': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'scale': [12.0, 30.0, 47.0, 47.0, 35.0, 30.0, 76.5, 63.3, 44.7]
            },
            'TP_8': {
                'shape': [0.2031, 0.2541, 0.3591, 0.3591, 0.00000001, 0.00000001, 0.00000001, 0.00000001, 0.00000001],
                'scale': [15.01, 41.89, 64.78, 64.78, 45.0, 30.0, 90.5, 69.7, 33.4]
            },
            'TP_9': {
                'shape': [0.1679, 0.1110, 0.0363, 0.0363, 0.1375, 0.0859, 0.00000001, 0.00000001, 0.00000001],
                'scale': [12.49, 25.84, 26.32, 26.32, 29.72, 26.57, 38.0, 35.6, 42.9]
            }
        }

    def initialize_data(self):
        # 파일 경로 설정
        type_params_file = 'type_params.pkl'
        correlation_matrices_file = 'correlation_matrices.pkl'
        type_counts_file = 'type_counts.pkl'
        stats_by_type_file = 'stats_by_type.pkl'

        # pkl 파일 존재 여부 확인
        if os.path.exists(type_params_file) and os.path.exists(correlation_matrices_file) and os.path.exists(type_counts_file) and os.path.exists(stats_by_type_file):
            # 파일이 존재하는 경우, 로드
            with open(type_params_file, 'rb') as f:
                self.type_params = pickle.load(f)
            with open(correlation_matrices_file, 'rb') as f:
                self.correlation_matrices = pickle.load(f)
            with open(type_counts_file, 'rb') as f:
                self.type_counts = pickle.load(f)
            with open(stats_by_type_file, 'rb') as f:
                self.stats_by_type = pickle.load(f)
        else:
            # 필요한 파일이 없는 경우, 엑셀 파일을 읽어서 계산 후 저장
            if self.data_file is not None:
                self.calculate_and_store_data()
            else:
                raise FileNotFoundError("Required data files not found. Please ensure type_params.pkl, correlation_matrices.pkl, type_counts.pkl, and stats_by_type.pkl are available.")

    def calculate_and_store_data(self):
        # 엑셀 데이터 읽기
        df = pd.read_excel(self.data_file)

        # 타입별 데이터 개수 계산
        self.type_counts = df['타입'].value_counts()

        # 숫자형 데이터만 선택 (타입, 호선번호, 블록명 등 비숫자형 데이터 제외)
        numeric_df = df.select_dtypes(include=[float, int])

        # '타입'별 피처별 평균 및 표준편차 계산
        self.stats_by_type = numeric_df.groupby(df['타입']).agg(['mean', 'std', 'count'])

        # 상관계수 행렬이 필요한 타입만 필터링
        df_filtered = df[df['타입'].isin(['TP_10', 'TP_2', 'TP_11', 'TP_8', 'TP_9', 'TP_14', 'TP_12', 'TP_16', 'TP_24'])]

        # 필터링된 타입에 대해서만 상관계수 행렬을 계산하고 shape와 scale도 계산
        self.correlation_matrices, calculated_type_params = self.calculate_and_store_correlations(df_filtered)

        # 전역 type_params를 calculated_type_params로 업데이트
        self.type_params.update(calculated_type_params)

        # 계산된 데이터를 파일로 저장
        with open('type_params.pkl', 'wb') as f:
            pickle.dump(self.type_params, f)
        with open('correlation_matrices.pkl', 'wb') as f:
            pickle.dump(self.correlation_matrices, f)
        with open('type_counts.pkl', 'wb') as f:
            pickle.dump(self.type_counts, f)
        with open('stats_by_type.pkl', 'wb') as f:
            pickle.dump(self.stats_by_type, f)

    def calculate_and_store_correlations(self, df, type_column='타입'):
        correlation_matrices = {}
        calculated_type_params = {}

        for type_name, group in df.groupby(type_column):
            numeric_data = group.select_dtypes(include=[float, int])
            if numeric_data.shape[1] == 10:  # 10개의 피쳐가 모두 있는지 확인
                if type_name in ['TP_10', 'TP_2', 'TP_11', 'TP_8', 'TP_9', 'TP_14', 'TP_12', 'TP_16', 'TP_24']:
                    corr_matrix = numeric_data.corr()
                    correlation_matrices[type_name] = corr_matrix
                
                shapes = []
                scales = []
                for col in numeric_data.columns:
                    shape, loc, scale = stats.lognorm.fit(numeric_data[col], floc=0)
                    shapes.append(shape)
                    scales.append(scale)
                
                calculated_type_params[type_name] = {'shape': shapes, 'scale': scales}

        return correlation_matrices, calculated_type_params

    def generate_data_for_type(self, type_name, type_counts):
        print(f"Generating data for type: {type_name}")
        print(f"Available types: {list(self.type_params.keys())}")

        params = self.type_params[type_name]
        shape = params['shape']
        scale = params['scale']
        correlation_matrix = self.correlation_matrices.get(type_name, None)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        process_time_temp = np.zeros((self.size * self.num_of_blocks, 10))

        if type_name not in ['TP_10', 'TP_2', 'TP_11','TP_8', 'TP_9', 'TP_14', 'TP_12', 'TP_16', 'TP_24']:
            print(f"타입 {type_name}은 상관관계 매트릭스가 없습니다. 평균 기반으로 가우시안 노이즈를 이용해 데이터를 생성합니다.")
            
            # 해당 타입의 평균 데이터 가져오기
            type_stats = self.stats_by_type.loc[type_name]
            type_mean = type_stats[type_stats.index.get_level_values(1) == 'mean']
            type_std = type_stats[type_stats.index.get_level_values(1) == 'std']
            
            for i, (mean, std) in enumerate(zip(type_mean, type_std)):
                feature_mean = max(mean, 0.1)  # 최소값 0.1 설정
                feature_std = max(std, 0.01)  # 최소 표준편차 0.01 설정
                
                # NaN 체크 및 처리
                if np.isnan(feature_std):
                    print(f"Warning: NaN detected for type {type_name}, feature {i}. Using default values.")
                    feature_std = 10
                
                if i in [0, 1, 2]:  # 크레인배재, 주판판계, 전면SAW
                    process_time_temp[:, i] = np.round(feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks))
                elif i in [5, 6]:  # NC마킹, 절단
                    process_time_temp[:, i] = np.round((feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks)) / 5) * 5
                elif i == 7:  # 론지배재
                    process_time_temp[:, i] = np.round((feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks)) * 2) / 2
                else:  # 론지용접, 론지수정
                    process_time_temp[:, i] = np.round(feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks), 1)
                    
            # 전면SAW와 후면SAW 값을 동일하게 설정
            process_time_temp[:, 4] = process_time_temp[:, 2]
            
            # TurnOver 값을 고정값 20으로 설정
            process_time_temp[:, 3] = 20
            
            # 순서 변경: 크레인배재, 주판판계, 전면SAW, TurnOver, 후면SAW, NC마킹, 절단, 론지배재, 론지용접, 론지수정
            process_time_temp = process_time_temp[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        if type_name in ['TP_8', 'TP_9', 'TP_14', 'TP_12', 'TP_16', 'TP_24']:
            print(f"타입 {type_name}에 대해 가능한 상관관계를 고려하여 데이터를 생성합니다.")
            
            type_stats = self.stats_by_type.loc[type_name]
            type_mean = type_stats[type_stats.index.get_level_values(1) == 'mean']
            type_std = type_stats[type_stats.index.get_level_values(1) == 'std']
            
            # 상관관계 매트릭스 확인
            correlation_matrix = self.correlation_matrices.get(type_name)
            
            # Group 1: 크레인배재 - 주판판계 - 전면SAW = 후면SAW
            if type_name in ['TP_8', 'TP_9', 'TP_14', 'TP_12', 'TP_16', 'TP_24'] and correlation_matrix is not None:
                group1_corr = correlation_matrix.iloc[:3, :3].values
                group1_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group1_corr, size=self.size * self.num_of_blocks)
                
                for i in range(3):
                    feature_mean = max(type_mean.iloc[i], 0.1)
                    feature_std = max(type_std.iloc[i], 0.01)
                    data = stats.norm.ppf(stats.norm.cdf(group1_data[:, i]), loc=feature_mean, scale=feature_std)
                    process_time_temp[:, i] = np.round(data)
                
                process_time_temp[:, 4] = process_time_temp[:, 2]  # 후면SAW
            else:
                for i in range(3):
                    feature_mean = max(type_mean.iloc[i], 0.1)
                    feature_std = max(type_std.iloc[i], 0.01)
                    process_time_temp[:, i] = np.round(feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks))
                process_time_temp[:, 4] = process_time_temp[:, 2]  # 후면SAW
            
            # TurnOver는 고정값 20
            process_time_temp[:, 3] = 20
            
            # Group 2: 론지배재 - 론지용접 - 론지수정
            if type_name in ['TP_12', 'TP_16', 'TP_24'] and correlation_matrix is not None:
                group2_corr = correlation_matrix.iloc[7:, 7:].values
                group2_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group2_corr, size=self.size * self.num_of_blocks)
                
                for i in range(3):
                    feature_mean = max(type_mean.iloc[i+7], 0.1)
                    feature_std = max(type_std.iloc[i+7], 0.01)
                    data = stats.norm.ppf(stats.norm.cdf(group2_data[:, i]), loc=feature_mean, scale=feature_std)
                    if i == 0:  # 론지배재
                        process_time_temp[:, i+7] = np.round(data * 2) / 2
                    else:  # 론지용접, 론지수정
                        process_time_temp[:, i+7] = np.round(data, 1)
            else:
                for i in [7, 8, 9]:
                    feature_mean = max(type_mean.iloc[i], 0.1)
                    feature_std = max(type_std.iloc[i], 0.01)
                    if i == 7:  # 론지배재
                        process_time_temp[:, i] = np.round((feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks)) * 2) / 2
                    else:  # 론지용접, 론지수정
                        process_time_temp[:, i] = np.round(feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks), 1)
            
            # NC마킹, 절단
            for i in [5, 6]:
                feature_mean = max(type_mean.iloc[i], 0.1)
                feature_std = max(type_std.iloc[i], 0.01)
                process_time_temp[:, i] = np.round((feature_mean + norm.rvs(loc=0, scale=feature_std, size=self.size * self.num_of_blocks)) / 5) * 5

            # 순서 변경: 크레인배재, 주판판계, 전면SAW, TurnOver, 후면SAW, NC마킹, 절단, 론지배재, 론지용접, 론지수정
            process_time_temp = process_time_temp[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        elif type_name in ['TP_10']:
            base_size = self.size * self.num_of_blocks
            process_time_temp = np.zeros((base_size, 10))
            
            # Group 1: 크레인배재 - 주판판계 - 전면SAW = 후면SAW
            group1_corr = correlation_matrix.iloc[:3, :3].values
            group1_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group1_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group1_data[:, i]), s=shape[i], scale=scale[i])
                if i == 0:  # 크레인배재
                    process_time_temp[:, i] = np.clip(np.round(data), 8, 35)
                elif i == 1:  # 주판판계
                    process_time_temp[:, i] = np.clip(np.round(data), 10, 105)
                else:  # 전면SAW
                    process_time_temp[:, i] = np.clip(np.round(data), 15, 125)
            
            process_time_temp[:, 4] = process_time_temp[:, 2]  # 후면SAW (전면SAW와 동일하게 설정)
            process_time_temp[:, 3] = 20  # TurnOver는 고정값 20

            # Group 2: 론지배재 - 론지용접 - 론지수정
            group2_corr = correlation_matrix.iloc[7:, 7:].values
            group2_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group2_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group2_data[:, i]), s=shape[i+7], scale=scale[i+7])
                if i == 0:  # 론지배재
                    process_time_temp[:, 7] = np.clip(np.round(data * 2) / 2, 24, 129)
                elif i == 1:  # 론지용접
                    process_time_temp[:, 8] = np.clip(np.round(data, 1), 16.7, 126)
                else:  # 론지수정
                    process_time_temp[:, 9] = np.clip(np.round(data, 1), 11.3, 54.9)

            # Independent features: NC마킹, 절단
            for i in [5, 6]:
                data = stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=base_size)
                if i == 5:  # NC마킹
                    process_time_temp[:, i] = np.clip(np.round(data / 5) * 5, 20, 65)
                else:  # 절단
                    process_time_temp[:, i] = np.clip(np.round(data / 5) * 5, 20, 45)

            # 극단값 추가 및 분포 조정
            for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
                if i not in [3, 4]:  # TurnOver와 후면SAW는 제외
                    data = process_time_temp[:, i]
                    min_val, max_val = np.min(data), np.max(data)
                    
                    # 극단값 비율 설정 (샘플 크기에 따라 조정)
                    extreme_ratio = min(0.05, 5 / base_size)
                    extreme_count = max(1, int(base_size * extreme_ratio))
                    
                    # 극단값 생성 및 추가 (상한값을 초과하지 않도록 주의)
                    if i == 0:  # 크레인배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 35), size=extreme_count)
                    elif i == 1:  # 주판판계
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 105), size=extreme_count)
                    elif i in [2, 4]:  # 전면SAW, 후면SAW
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 125), size=extreme_count)
                    elif i == 5:  # NC마킹
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 65), size=extreme_count)
                    elif i == 6:  # 절단
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 45), size=extreme_count)
                    elif i == 7:  # 론지배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 129), size=extreme_count)
                    elif i == 8:  # 론지용접
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 126), size=extreme_count)
                    elif i == 9:  # 론지수정
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 54.9), size=extreme_count)
                    
                    random_indices = np.random.choice(base_size, extreme_count, replace=False)
                    data[random_indices] = extreme_values
                    
                    # 규칙 적용
                    if i in [0, 1, 2, 5, 6]:  # 정수값 필요한 컬럼
                        data = np.round(data)
                    elif i == 7:  # 론지배재 (0.5 단위)
                        data = np.round(data * 2) / 2
                    else:  # 론지용접, 론지수정 (소수점 1자리)
                        data = np.round(data, 1)
                    
                    process_time_temp[:, i] = data

            # 전면SAW와 후면SAW 값을 동일하게 설정
            process_time_temp[:, 4] = process_time_temp[:, 2]

            # 전면SAW, TurnOver, 후면SAW 순서로 변경
            process_time_temp = process_time_temp[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
            
        elif type_name in ['TP_2']:
            base_size = self.size * self.num_of_blocks
            process_time_temp = np.zeros((base_size, 10))
            
            # Group 1: 크레인배재 - 주판판계 - 전면SAW = 후면SAW
            group1_corr = correlation_matrix.iloc[:3, :3].values
            group1_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group1_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group1_data[:, i]), s=shape[i], scale=scale[i])
                if i == 0:  # 크레인배재
                    process_time_temp[:, i] = np.clip(np.round(data), 8, 35)
                elif i == 1:  # 주판판계
                    process_time_temp[:, i] = np.clip(np.round(data), 12, 105)
                else:  # 전면SAW
                    process_time_temp[:, i] = np.clip(np.round(data), 15, 107)
            
            process_time_temp[:, 4] = process_time_temp[:, 2]  # 후면SAW (전면SAW와 동일하게 설정)
            process_time_temp[:, 3] = 20  # TurnOver는 고정값 20

            # Group 2: 론지배재 - 론지용접 - 론지수정
            group2_corr = correlation_matrix.iloc[7:, 7:].values
            group2_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group2_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group2_data[:, i]), s=shape[i+7], scale=scale[i+7])
                if i == 0:  # 론지배재
                    process_time_temp[:, 7] = np.clip(np.round(data * 2) / 2, 34.5, 101)
                elif i == 1:  # 론지용접
                    process_time_temp[:, 8] = np.clip(np.round(data, 1), 23.3, 95.4)
                else:  # 론지수정
                    process_time_temp[:, 9] = np.clip(np.round(data, 1), 17.4, 50.4)

            # Independent features: NC마킹, 절단
            for i in [5, 6]:
                data = stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=base_size)
                if i == 5:  # NC마킹
                    process_time_temp[:, i] = np.clip(np.round(data / 5) * 5, 20, 45)
                else:  # 절단
                    process_time_temp[:, i] = np.maximum(np.round(data / 5) * 5, 20)

            # 극단값 추가 및 분포 조정
            for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
                if i not in [3, 4]:  # TurnOver와 후면SAW는 제외
                    data = process_time_temp[:, i]
                    min_val, max_val = np.min(data), np.max(data)
                    
                    # 극단값 비율 설정 (샘플 크기에 따라 조정)
                    extreme_ratio = min(0.05, 5 / base_size)
                    extreme_count = max(1, int(base_size * extreme_ratio))
                    
                    # 극단값 생성 및 추가 (상한값을 초과하지 않도록 주의)
                    if i == 0:  # 크레인배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 35), size=extreme_count)
                    elif i == 1:  # 주판판계
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 105), size=extreme_count)
                    elif i in [2, 4]:  # 전면SAW, 후면SAW
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 107), size=extreme_count)
                    elif i == 5:  # NC마킹
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 45), size=extreme_count)
                    elif i == 7:  # 론지배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 101), size=extreme_count)
                    elif i == 8:  # 론지용접
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 95.4), size=extreme_count)
                    elif i == 9:  # 론지수정
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 50.4), size=extreme_count)
                    else:  # 절단 (상한 없음)
                        extreme_values = np.random.uniform(max_val, max_val * 1.1, size=extreme_count)
                    
                    random_indices = np.random.choice(base_size, extreme_count, replace=False)
                    data[random_indices] = extreme_values
                    
                    # 규칙 적용
                    if i in [0, 1, 2, 5, 6]:  # 정수값 필요한 컬럼
                        data = np.round(data)
                    elif i == 7:  # 론지배재 (0.5 단위)
                        data = np.round(data * 2) / 2
                    else:  # 론지용접, 론지수정 (소수점 1자리)
                        data = np.round(data, 1)
                    
                    process_time_temp[:, i] = data

            # 전면SAW와 후면SAW 값을 동일하게 설정
            process_time_temp[:, 4] = process_time_temp[:, 2]

            # 전면SAW, TurnOver, 후면SAW 순서로 변경
            process_time_temp = process_time_temp[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        elif type_name in ['TP_11']:
            base_size = self.size * self.num_of_blocks
            process_time_temp = np.zeros((base_size, 10))
            
            # Group 1: 크레인배재 - 주판판계 - 전면SAW = 후면SAW
            group1_corr = correlation_matrix.iloc[:3, :3].values
            group1_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group1_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group1_data[:, i]), s=shape[i], scale=scale[i])
                if i == 0:  # 크레인배재
                    process_time_temp[:, i] = np.clip(np.round(data), 8, 15)
                elif i == 1:  # 주판판계
                    process_time_temp[:, i] = np.clip(np.round(data), 10, 40)
                else:  # 전면SAW
                    process_time_temp[:, i] = np.clip(np.round(data), 20, 65)
            
            process_time_temp[:, 4] = process_time_temp[:, 2]  # 후면SAW (전면SAW와 동일하게 설정)
            process_time_temp[:, 3] = 20  # TurnOver는 고정값 20

            # Group 2: 론지배재 - 론지용접 - 론지수정
            group2_corr = correlation_matrix.iloc[7:, 7:].values
            group2_data = np.random.multivariate_normal(mean=[0, 0, 0], cov=group2_corr, size=base_size)
            
            for i in range(3):
                data = stats.lognorm.ppf(stats.norm.cdf(group2_data[:, i]), s=shape[i+7], scale=scale[i+7])
                if i == 0:  # 론지배재
                    process_time_temp[:, 7] = np.clip(np.round(data * 2) / 2, 27, 59)
                elif i == 1:  # 론지용접
                    process_time_temp[:, 8] = np.clip(np.round(data, 1), 23, 75.5)
                else:  # 론지수정
                    process_time_temp[:, 9] = np.clip(np.round(data, 1), 21.8, 54.9)

            # Independent features: NC마킹, 절단
            for i in [5, 6]:
                data = stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=base_size)
                if i == 5:  # NC마킹
                    process_time_temp[:, i] = np.clip(np.round(data / 5) * 5, 20, 65)
                else:  # 절단
                    process_time_temp[:, i] = np.clip(np.round(data / 5) * 5, 20, 45)

            # 극단값 추가 및 분포 조정
            for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
                if i not in [3, 4]:  # TurnOver와 후면SAW는 제외
                    data = process_time_temp[:, i]
                    min_val, max_val = np.min(data), np.max(data)
                    
                    # 극단값 비율 설정 (샘플 크기에 따라 조정)
                    extreme_ratio = min(0.05, 5 / base_size)
                    extreme_count = max(1, int(base_size * extreme_ratio))
                    
                    # 극단값 생성 및 추가 (상한값을 초과하지 않도록 주의)
                    if i == 0:  # 크레인배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 15), size=extreme_count)
                    elif i == 1:  # 주판판계
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 40), size=extreme_count)
                    elif i in [2, 4]:  # 전면SAW, 후면SAW
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 65), size=extreme_count)
                    elif i == 5:  # NC마킹
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 65), size=extreme_count)
                    elif i == 6:  # 절단
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 45), size=extreme_count)
                    elif i == 7:  # 론지배재
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 59), size=extreme_count)
                    elif i == 8:  # 론지용접
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 75.5), size=extreme_count)
                    elif i == 9:  # 론지수정
                        extreme_values = np.random.uniform(max_val, min(max_val * 1.1, 54.9), size=extreme_count)
                    
                    random_indices = np.random.choice(base_size, extreme_count, replace=False)
                    data[random_indices] = extreme_values
                    
                    # 규칙 적용
                    if i in [0, 1, 2, 5, 6]:  # 정수값 필요한 컬럼
                        data = np.round(data)
                    elif i == 7:  # 론지배재 (0.5 단위)
                        data = np.round(data * 2) / 2
                    else:  # 론지용접, 론지수정 (소수점 1자리)
                        data = np.round(data, 1)
                    
                    process_time_temp[:, i] = data

            # 전면SAW와 후면SAW 값을 동일하게 설정
            process_time_temp[:, 4] = process_time_temp[:, 2]

            # 전면SAW, TurnOver, 후면SAW 순서로 변경
            process_time_temp = process_time_temp[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]


        process_time_temp = process_time_temp.reshape((self.size, self.num_of_blocks, 10))
        process_time = {}

        for i in range(self.size):
            process_time[str(i)] = torch.FloatTensor(process_time_temp[i]).to(device)

        return process_time


    def generate_all_types(self, type_counts):
        all_data = {}
        total_samples = self.num_of_blocks
        total_ex = 1060
        type_ratios = {
            'TP_10': 878 / total_ex,
            'TP_2': 77 / total_ex,
            'TP_11': 48 / total_ex,
            'TP_14': 8 / total_ex,
            'TP_9': 6 / total_ex,
            'TP_12': 5 / total_ex,
            'TP_24': 4 / total_ex,
            'TP_4': 3 / total_ex,
            'TP_8': 3 / total_ex,
            'TP_1': 3 / total_ex,
            'TP_16': 3 / total_ex,
            'TP_22': 2 / total_ex,
            'TP_13': 2 / total_ex,
            'TP_15': 2 / total_ex,
            'TP_20': 2 / total_ex,
            'TP_23': 2 / total_ex,
            'TP_17': 2 / total_ex,
            'TP_18': 2 / total_ex,
            'TP_7': 2 / total_ex,
            'TP_3': 2 / total_ex,
            'TP_19': 1 / total_ex,
            'TP_6': 1 / total_ex,
            'TP_5': 1 / total_ex,
            'TP_21': 1 / total_ex,
        }

        types = list(type_ratios.keys())
        probabilities = list(type_ratios.values())

        # 확률에 따라 타입 선택 (정확히 total_samples 개수만큼)
        selected_types = np.random.choice(types, size=total_samples, p=probabilities)

        # 각 타입별로 데이터 생성
        for type_name in types:
            type_count = np.sum(selected_types == type_name)
            if type_count > 0:
                self.size = type_count
                all_data[type_name] = self.generate_data_for_type(type_name, type_counts)

        return all_data, selected_types


    def save_all_data_to_excel(self, all_data, selected_types, filepath, num_of_process):
        os.makedirs(filepath, exist_ok=True)
        
        # 파일 이름 생성 및 중복 확인
        base_filename = f'PBS_{num_of_process}_{self.num_of_blocks}'
        file_index = 1
        while True:
            filename = f'{base_filename}_{file_index}.xlsx'
            full_path = os.path.join(filepath, filename)
            if not os.path.exists(full_path):
                break
            file_index += 1
        
        writer = pd.ExcelWriter(full_path, engine='openpyxl')
        
        all_df = []
        
        for type_name, data in all_data.items():
            if data:  # 데이터가 있는 경우에만 처리
                data_np = data['0'].cpu().numpy()
                df = pd.DataFrame(data_np, columns=['크레인배재', '주판판계', '전면SAW', 'TurnOver', '후면SAW', 'NC마킹', '절단', '론지배재', '론지용접', '론지수정'])
                df['타입'] = type_name
                all_df.append(df)
        
        final_df = pd.concat(all_df, ignore_index=True)
        
        # selected_types에 따라 데이터 재정렬
        final_df_reordered = pd.DataFrame(columns=final_df.columns)
        for type_name in selected_types:
            type_data = final_df[final_df['타입'] == type_name].sample(n=1)
            final_df_reordered = pd.concat([final_df_reordered, type_data], ignore_index=True)
        
        final_df_reordered.to_excel(writer, sheet_name='All_Data', index=False)
        
        writer.close()

        print(f"파일이 저장되었습니다: {filename}")

        # 타입별 개수 확인
        type_counts = final_df_reordered['타입'].value_counts()
        print("타입별 생성된 데이터 개수:")
        print(type_counts)


# if __name__ == "__main__":
#     # DataGenerator 인스턴스를 생성합니다.
#     data_generator = DataGenerator(num_of_blocks=1060, size=1)

#     # 데이터를 생성합니다.
#     all_data, selected_types = data_generator.generate_all_types(data_generator.type_counts)

    # # 생성된 데이터 구조를 출력해봅니다.
    # print("데이터 생성 결과를 확인합니다.")
    
    # # all_data의 구조 확인
    # for type_name, blocks in all_data.items():
    #     print(f"Type: {type_name}")
    #     for block_index, block_data in blocks.items():
    #         print(f"  Block index: {block_index}")
    #         for process_index, process_time in enumerate(block_data):
    #             print(f"    Process {process_index + 1}: {process_time}")
    
    # # selected_types의 구조 확인
    # print("선택된 타입 확인:")
    # print(selected_types)

    # # 데이터를 엑셀 파일로 저장합니다.
    # output_directory = r'C:\Users\ohj\Desktop\PBS\environment'
    # data_generator.save_all_data_to_excel(all_data, selected_types, output_directory, num_of_process=10)

    # print("데이터 생성 및 저장이 완료되었습니다.")

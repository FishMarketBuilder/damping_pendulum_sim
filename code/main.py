import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from control import matlab, tf

class Physical_SIM:
    '''
    物理の数値解析のためのクラス
    '''
    def __init__(self):
        self.params = {}
        self.dt = 0        
    
    def setup(self, params, dt):
        '''
        パラメータ設定
        
        入力
        params: dict, EOMの物理パラメータ
        dt: float, シミュレーションの時間刻み
        '''
        self.params = params
        self.dt = dt
        
    def EOM(self, vector):
        '''
        減衰振り子の運動方程式
        d^2 theta/dt^2 + (b/m)* d theta/dt + (g/L) * sin(theta) = 0
        
        入力
        vector = [theta, omega]: np.array, 角度と角速度
        self.params: dict, 物理パラメータ
        self.params['b']: float, 減衰係数
        self.params['m']: float, 振り子の質量
        self.params['g']: float, 重力加速度
        self.params['L']: float, 振り子の長さ
        
        出力
        dvector_dt = [dtheta_dt, domega_dt]: np.array, 角度と角速度の時間微分
        '''
        theta = vector[0]
        omega = vector[1]
        b = self.params['b']
        m = self.params['m']
        g = self.params['g']
        L = self.params['L']
        dtheta_dt = omega
        domega_dt = - b/m * omega - g/L * np.sin(theta)
        # domega_dt = - b/m * omega - g/L * theta # 1次近似
        
        dvector_dt = np.array([dtheta_dt, domega_dt])
        
        return dvector_dt
    
    def Runge_Kutta_4th(self, vector):
        '''
        4次のルンゲクッタ法
        
        入力
        vector: np.array, 現時刻での角度、角速度
        self.dt: float, 時間刻み
        
        出力
        vector_next: np.array, 次時刻での角度、角速度
        '''
        k1 = self.EOM(vector)
        k2 = self.EOM(vector + k1*self.dt/2)
        k3 = self.EOM(vector + k2*self.dt/2)
        k4 = self.EOM(vector + k3*self.dt)
        
        vector_next = vector + (k1 + 2*k2 + 2*k3 + k4) * self.dt / 6
        return vector_next
    
    def calc_time_evolution(self, params, vector_init, Nt, dt):
        '''
        時間発展を計算
        
        入力
        params: dict, EOMの物理パラメータ
        vector_init: list, 初期値
        Nt: int, シミュレーションの時間ステップ数
        dt: float, シミュレーションの時間刻み
        
        出力
        list_time: list, 時間のリスト
        list_vector: list, 角度、角速度ベクトルのリスト
        '''
        self.setup(params, dt)
        vector = vector_init.copy()
        
        list_time = []
        list_vector = []
        for it in np.arange(0, Nt, 1):
            list_time.append(it*dt)
            list_vector.append(vector)
            
            vector = self.Runge_Kutta_4th(vector)
            
        return list_time, list_vector
    
    def calc_analytical_solution(self, params, vector_init, list_time):
        '''
        EOMの解析解（sinを3次関数に近似）を出力
        
        入力
        params: dict, EOMの物理パラメータ
        list_time: list, 時間のリスト
        
        出力
        list_theta_analytics: list, 角度の解析解
        '''
        theta_init = vector_init[0]
        omega_init = vector_init[1]
        b = params['b']
        m = params['m']
        g = params['g']
        L = params['L']
        gamma = b / m
        alpha = g / L
        
        # TODO: Aを厳密に求める
        lambda_t0 = np.sqrt(alpha - gamma**2 / 4)
        delta = np.arctan(-gamma/2/lambda_t0)
        A = theta_init / np.cos(delta)
        
        list_theta_analytics = []
        for t in list_time:
            # 3次近似
            lambda_t = np.sqrt(alpha - gamma**2 / 4 - alpha / 8 * A**2 *np.exp(-gamma*t))
            B_t = - 1 / 192 * alpha / (alpha - gamma**2/4) * A**3 #* np.exp(-gamma*t)
            # 1次近似
            # lambda_t = np.sqrt(alpha - gamma**2 / 4)
            # B_t = 0
            theta = np.exp(-gamma/2*t) * (A*np.cos(lambda_t*t + delta) + B_t*np.cos(3*(lambda_t*t + delta)))
            list_theta_analytics.append(theta)
        return list_theta_analytics
    
class Data_Processor:
    '''
    データ処理のためのクラス
    '''
    def __init__(self):
        self.params = {}
        self.vector_init = []
        self.Nt = 0
        self.dt = 0
        self.output_path = ''
        
    ### パラメータ設定メソッド群 Start ###
    def setup_by_hand(self):
        '''
        メソッド内でのパラメータ設定。簡単にパラメータ設定してシミュレーションする際に使用
        '''
        self.params = {
            'b': 0.1, # 減衰係数 [kg/s]
            'm': 0.1, # 振り子の質量 [kg]
            'g': 9.81,# 重力加速度 [m/s^2]
            'L': 1,   # 振り子の長さ [m]
        }
        self.vector_init = [np.pi/4, 0] # 初期値 [theta, omega]
        self.Nt = 1000
        self.dt = 0.01
        self.output_path = 'output.xlsx'
        
        print(self.params)
        
    def setup_by_input(self, params, vector_init, Nt, dt, output_path):
        '''
        入力でのパラメータ設定。数種類のパラメータ設定で順次シミュレーションする際に使用
        '''
        self.params = params.copy()
        self.vector_init = vector_init.copy()
        self.Nt = Nt
        self.dt = dt
        self.output_path = output_path
    ### パラメータ設定メソッド群 End ###
    
    def write_excel(self, path, list_time, list_vector, list_theta_analytics = 0, list_theta_fitting = 0):
        '''
        データとパラメータをExcelに書き込む
        
        Posシート: 角度・角速度の時間発展結果
        Paramsシート: パラメータデータ
        '''
        with pd.ExcelWriter(path) as writer:
            df_vector = pd.DataFrame(list_vector, columns=['theta', 'omega'])
            df_vector.index = list_time
            
            
            if list_theta_analytics != 0:
                df_vector['theta_analytics'] = list_theta_analytics
            
            if list_theta_fitting != 0:
                df_vector['theta_fitting'] = list_theta_fitting
            
            df_vector.to_excel(writer, sheet_name = 'Pos')
            
            df_params = pd.DataFrame(list(self.params.items()), columns=['Parameter', 'Value'])
            df_params.to_excel(writer, sheet_name = 'Params')
            
    def do_sim(self):
        '''
        シミュレーションの実行メソッド
        注意: パラメータ設定メソッド群(setup_*)のいずれかでパラメータ設定後に実行すること
        '''
        list_time, list_vector = Physical_SIM().calc_time_evolution(self.params, self.vector_init, self.Nt, self.dt)
        list_theta_analytics =  Physical_SIM().calc_analytical_solution(self.params, self.vector_init, list_time)
        self.write_excel(self.output_path, list_time, list_vector, list_theta_analytics)
        
        self.list_time = list_time
        self.list_vector = list_vector
        self.list_theta_analytics = list_theta_analytics
        
    def do_fit(self):
        '''
        フィッティング実行メソッド
        注意: do_sim()メソッドを実行した後に実行すること
        '''
        # list_theta_fitting = Fitting_Method().do_fit(self.list_time, self.list_vector)
        
        list_theta_fitting = Fitting_s_space().do_fit(self.list_time, self.list_vector)
        
        self.write_excel(self.output_path, self.list_time, self.list_vector, self.list_theta_analytics, list_theta_fitting)
    
class Fitting_Class:
    def __init__(self):
        pass
    
    def fit_func(self, x, params):
        '''
        フィッティング関数: y = a*x**2 + b*x + c
        
        入力
        x: np.array, xの値のリスト
        params: list, フィッティングパラメータ
        
        出力
        y: float, yの値
        '''
        a = params[0]
        b = params[1]
        c = params[2]
        
        y = a*x**2 + b*x + c
        return y
        
    def least_square(self, list_x, list_y):
        '''
        最小二乗法によるフィッティング
        フィッティング関数: y = a*x**2 + b*x + c
        
        入力
        list_x: list, xのデータリスト
        list_y: list, yのデータリスト
        
        出力
        params = [a, b, c]: list, フィット関数のパラメータ
        '''
        list_x2 = np.array(list_x)**2
        list_x1 = np.array(list_x)
        list_x0 = np.ones(len(list_x))
        list_y  = np.array(list_y)
        
        mat_X = np.matrix([list_x2, list_x1, list_x0])
        mat_Y = np.matrix(list_y)
        params = mat_Y @ mat_X.T @ np.linalg.inv(mat_X @ mat_X.T)
        params = np.array(params).flatten()
        
        return params
    
    def do_fit(self, list_time, list_vector):
        '''
        フィッティングを実行するためのメソッド
        '''
        pass
    
class Fitting_Method(Fitting_Class):
    def __init__(self):
        self.index_wide = 0
        self.mode_separate = 1 # 0: 1/2周期、1: 1/4周期、2: 1/8周期で分割を行う
    
    def do_fit(self, list_time, list_vector):
        '''
        フィッティングを実行するためのメソッド
        波形を切り出して、その部分ごとにフィッティングを行う
        
        入力
        list_time: list, 時間
        list_vector: list, 角度、角速度の時間発展
        
        出力
        list_theta_fitting: list, フィッティングされた角度のリスト
        '''
        list_time = np.array(list_time)
        list_theta = [vector[0] for vector in list_vector] # thetaのみ抽出
        
        index_zero_theta, index_peak_theta = self.detect_half_period(list_vector)
        
        # 初期時刻と衆力時刻を追加
        index_zero_theta.insert(0, 0)
        index_zero_theta.append(len(list_time))
        
        # ピーク時間を追加（1/4周期フィッティング）
        if self.mode_separate > 0:
            index_zero_theta.extend(index_peak_theta)
            index_zero_theta.sort()
        
        # 時間をさらに半分にする（1/8周期フィッティング）
        if self.mode_separate > 1:
            index_zero_theta_half = []
            for it in np.arange(1, len(index_zero_theta), 1):
                index_mid = (index_zero_theta[it-1] + index_zero_theta[it]) // 2
                index_zero_theta_half.append(index_mid)
            index_zero_theta.extend(index_zero_theta_half)
            index_zero_theta.sort()
    
        # print(index_zero_theta)
        
        list_params = []
        list_theta_fitting = []
        for it in np.arange(1, len(index_zero_theta), 1):
            it1 = index_zero_theta[it-1]
            it2 = index_zero_theta[it]
            
            it1_wide = -self.index_wide
            it2_wide = self.index_wide
            if it == 1:
                it1_wide = 0
            if it == len(index_zero_theta) - 1:
                it2_wide = 0
            
            params = self.least_square(
                list_time[int(it1+it1_wide) : int(it2+it2_wide)], 
                list_theta[int(it1+it1_wide) : int(it2+it2_wide)]
            )
            list_params.append(params)
            
            # print(it1, it2)
            # print(list_time[it1:it2])
            list_theta_fitting.extend(self.fit_func(list_time[it1:it2], params))
            
        return list_theta_fitting
        
    def detect_half_period(self, list_vector):
        '''
        半周期を検出するメソッド
        '''
        list_theta = [vector[0] for vector in list_vector]
        list_omega = [vector[1] for vector in list_vector]
        
        index_zero_theta = self.detect_zero_crossing(list_theta)
        index_peak_theta = self.detect_zero_crossing(list_omega) # omega=0 -> thetaのピーク
        
        return index_zero_theta, index_peak_theta
    
    def detect_zero_crossing(self, list_target):
        '''
        ゼロ点を横切った時のインデックスを検出するメソッド
        
        入力
        list_target: list, ゼロ点を検知するターゲット(角度か角速度)
        
        出力
        index_zero_cross: list, ゼロ点を横切った時のlist_targetのインデックス
        '''
        index_zero_cross = []
        for it in np.arange(1, len(list_target), 1):
            if list_target[it-1] * list_target[it] < 0:
                index_zero_cross.append(it)
        return index_zero_cross
    
class Fitting_s_space(Fitting_Class):
    def __init__(self):
        self.index_wide = 0
        self.mode_separate = 1 # 0: 1/2周期、1: 1/4周期、2: 1/8周期で分割を行う
    
    def calc_Laplace(self, list_time, list_sig):
        '''
        ラプラス変換を実行
        
        入力
        list_time: list, 時間
        list_sig: list, ラプラス変換する信号
        
        出力
        list_s: list, s(=i*f)
        list_sig_s: list, ラプラス変換後の信号
        '''
        list_sig_s = []
        dt = list_time[1] - list_time[0]
        list_s = np.arange(0, 1/dt, 0.1)
        for s in list_s:
            laplace = 0
            for it, t in enumerate(list_time):
                exp_s = np.exp(-s*t) * list_sig[it]
                laplace += exp_s
            laplace *= dt
            list_sig_s.append(laplace)
        return list_s, np.array(list_sig_s)
    
    def func_eval(self, wave1, wave2):
        ave = 0
        Nt = min(len(wave1), len(wave2))
        for i in np.arange(0, Nt, 1):
            ave += (wave1[i] - wave2[i])**2
        ave /= Nt
        return np.sqrt(ave)
    
    def func_objective(self, trial):
        '''
        目的関数
        '''
        a = trial.suggest_float('a', 0, 10)
        b = trial.suggest_float('b', 0, 10)
        c = trial.suggest_float('c', 0, 10)
        
        y = a*list_s**2 + b*list_s + c
        ans = func_eval(y, self.list_tgt)
        return ans
    
    def do_fit(self, list_time, list_vector):
        '''
        フィッティングを実行するためのメソッド
        
        入力
        list_time: list, 時間
        list_vector: list, 角度、角速度の時間発展
        
        出力
        list_theta_fitting: list, フィッティングされた角度のリスト
        '''
        list_time = np.array(list_time)
        list_theta = np.array([vector[0] for vector in list_vector]) # thetaのみ抽出
        list_omega = np.array([vector[1] for vector in list_vector]) # omegaのみ抽出
        stroke_width = list_theta[0]
        
        list_s, list_omega_s = self.calc_Laplace(list_time, -list_omega)
        
        # optunaで探索範囲を制限する
#         self.list_s = list_s.copy()
#         self.list_tgt = 1 / list_omega_s
#         self.study = optuna.create_study()
#         self.study.optimize(seld.func_objective, n_trials = 500)
        
#         self.trials = sorted(self.study.best_trials, key=lambda t: t.values)
#         for trial in self.trials:
#             params = trial.params.copy()
#             print(params)
            
#             a = params['a']
#             b = params['b']
#             c = params['c']
        
        a,b,c = self.least_square(list_s, 1/list_omega_s)
        print(a,b,c)
        list_omega_s_fit = self.fit_func(list_s, [a,b,c])
        
        plt.plot(1/list_omega_s)
        plt.plot(list_omega_s_fit, 'r--')
        plt.show()
        
        # システム応答を
        sys = tf([1], [a,b,c])
        (list_theta_fitting, _) = matlab.step(sys, list_time)
        ini = list_theta_fitting[0]
        end = list_theta_fitting[-1]
        list_theta_fitting = list_theta_fitting / (end - ini)
        list_theta_fitting = -stroke_width*list_theta_fitting + stroke_width
        
        plt.plot(list_theta)
        plt.plot(list_theta_fitting)
        plt.show()
        
        return list_theta_fitting
        
    def detect_1st_zero_crossing(self, list_target):
        '''
        初めてゼロ点を横切った時のインデックスを検出するメソッド
        
        入力
        list_target: list, ゼロ点を検知するターゲット(角度か角速度)
        
        出力
        time_zero_cross: int, ゼロ点を横切った時のlist_targetのインデックス
        '''
        index_zero_cross = 0
        for it in np.arange(1, len(list_target), 1):
            if list_target[it-1] * list_target[it] < 0:
                time_zero_cross = it
                break
        return index_zero_cross
    
class test(Fitting_s_space):
    
    def func(self):
        self.gamma = 1
        self.lambda_t = 3.09
        self.delta = 0
        
        list_time = np.arange(0, 100, 0.01)
        list_theta = np.exp(-self.gamma*list_time)*np.sin(self.lambda_t*list_time+self.delta)
        # list_theta = np.exp(-self.gamma*list_time)*np.cos(self.lambda_t*list_time+self.delta)
        
        return list_time, list_theta
    
    def do_test(self):
        list_time, list_theta = self.func()
        list_s, list_theta_s = self.calc_Laplace(list_time, list_theta)
        
        
        list_tgt = 1/list_theta_s
        a,b,c = self.least_square(list_s[:100], list_tgt[:100])
        list_theta_s_fit = self.fit_func(list_s, [a,b,c])
        print(a,b,c)
        
        plt.plot(list_tgt)
        plt.plot(list_theta_s_fit, 'r--')
        plt.show()
        
        df = pd.DataFrame()
        df.index = list_s
        df['tgt'] = list_tgt
        df['fit'] = list_theta_s_fit
        
        
        df.to_excel('test.xlsx')
        
        
        
        
        
    
if __name__=='__main__':
    data_processor = Data_Processor()
    data_processor.setup_by_hand() # パラメータと初期値入力
    # data_processor.do_sim()        # 時間発展の計算
    # data_processor.do_fit()        # フィッティング実行
    
    test().do_test()
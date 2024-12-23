import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt

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
        # フィットモード選択
        self.str_time_space = 'time_space' # 時間区間での切り貼りフィット
        self.str_s_space = 's_space' # 伝達関数推定
        
        self.params = {}
        self.vector_init = []
        self.Nt = 0
        self.dt = 0
        self.path_output_fit = ''
        
        self.list_time = []
        self.list_vector = []
        self.list_theta_analytics = []
        self.list_theta_fitting = []
        
    ### パラメータ設定メソッド群 Start ###
    def setup_by_hand(self, mode_fitting):
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
        self.Nt = 10000
        self.dt = 0.01
        
        self.mode_fitting = mode_fitting
        if self.mode_fitting == self.str_time_space: # 時間区間での切り貼りフィット
            self.path_output_fit = 'output_time_space.xlsx'
        if self.mode_fitting == self.str_s_space: # 伝達関数推定
            self.path_output_fit = 'output_s_space.xlsx'
        
        print('=== Fitting mode: ', self.mode_fitting, ' Start!! ===')
        print('Physical parameter: ', self.params)
        
#     def setup_by_input(self, params, vector_init, Nt, dt, path_output_fit, mode_fitting):
#         '''
#         入力でのパラメータ設定。数種類のパラメータ設定で順次シミュレーションする際に使用
#         '''
#         self.params = params.copy()
#         self.vector_init = vector_init.copy()
#         self.Nt = Nt
#         self.dt = dt
#         self.path_output_fit = path_output_fit
#         self.mode_fitting = mode_fitting
        
#         if self.mode_fitting == self.str_time_space: # 時間区間での切り貼りフィット
#             self.path_output_fit = 'output_time_space.xlsx'
#         if self.mode_fitting == self.str_s_space: # 伝達関数推定
#             self.path_output_fit = 'output_s_space.xlsx'
            
#        print('=== Fitting mode: ', self.mode_fitting, ' Start!! ===')
#        print('Physical parameter: ', self.params)
    ### パラメータ設定メソッド群 End ###
    
    def write_excel(self):
        '''
        データとパラメータをExcelに書き込む
        
        Posシート: 角度・角速度の時間発展結果
        Paramsシート: 物理パラメータデータ

        ※ 次の２つはFitting_s_spaceを実行時のみ生成
        S_spaceシート: 伝達関数推定のs座標でのフィット結果
        Params_S_fitシート: 伝達関数推定のs座標でのフィット関数パラメータ
        
        注意: do_sim()メソッドを実行した後に実行すること
        '''
        with pd.ExcelWriter(self.path_output_fit) as writer:
            df_vector = pd.DataFrame(self.list_vector, columns=['theta', 'omega'])
            df_vector.index = self.list_time
            
            
            if len(self.list_theta_analytics) != 0:
                df_vector['theta_analytics'] = self.list_theta_analytics
            
            if len(self.list_theta_fitting) != 0:
                df_vector['theta_fitting'] = self.list_theta_fitting
            
            df_vector.to_excel(writer, sheet_name = 'Pos')
            
            df_params = pd.DataFrame(list(self.params.items()), columns=['Parameter', 'Value'])
            df_params.to_excel(writer, sheet_name = 'Params_Phys', index = False)
            
            
            if self.mode_fitting == self.str_s_space:
                df_s_log = pd.DataFrame()
                df_s_params = pd.DataFrame()
                for key in Fitting_s_space().dict_fit_log.keys():
                    df_s_log[key] = Fitting_s_space().dict_fit_log[key]
                for key in Fitting_s_space().dict_fit_params.keys():
                    df_s_params[key] = Fitting_s_space().dict_fit_params[key]
                
                df_s_log.to_excel(writer, sheet_name = 'S_space', index = False)
                df_s_params.to_excel(writer, sheet_name = 'Params_S_fit')
            
    def do_sim(self):
        '''
        シミュレーションの実行メソッド
        注意: パラメータ設定メソッド群(setup_*)のいずれかでパラメータ設定後に実行すること
        '''
        list_time, list_vector = Physical_SIM().calc_time_evolution(self.params, self.vector_init, self.Nt, self.dt)
        # list_theta_analytics =  Physical_SIM().calc_analytical_solution(self.params, self.vector_init, list_time)
        
        self.list_time = list_time
        self.list_vector = list_vector
        # self.list_theta_analytics = list_theta_analytics
        
    def do_fit(self):
        '''
        フィッティング実行メソッド
        注意: do_sim()メソッドを実行した後に実行すること
        '''
        if self.mode_fitting == 'time_space':
            list_theta_fitting = Fitting_time_space().do_fit(self.list_time, self.list_vector)
        if self.mode_fitting == 's_space':
            list_theta_fitting = Fitting_s_space().do_fit(self.list_time, self.list_vector)
        
        self.list_theta_fitting = list_theta_fitting
    
class Fitting_Class:
    '''
    フィッティング機能クラスの大元のクラス
    共通して使用するメソッドを格納
    '''
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
        Data_Processorから呼び出すために使用
        '''
        pass
    
class Fitting_time_space(Fitting_Class):
    '''
    フィッティングを実行するためのクラス
    時間領域で波形を切り出して、その部分ごとにフィッティングを行う
    '''
        
    def __init__(self):
        self.index_wide = 0
        self.mode_separate = 2 # 0: 1/2周期、1: 1/4周期、2: 1/8周期で分割を行う
    
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
    '''
    伝達関数をフィットするためのクラス
    '''
    dict_fit_log = {}
    dict_fit_params = {}
    
    def __init__(self):
        self.range_fitting = 3
        self.mode_high_accuracy = 1 # 1: 高精度モードOFF, 1: 高精度モードON

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
                index_zero_cross = it
                break
        return index_zero_cross
    
    def calc_shift(self, list_time, list_tgt, index_shift):
        '''
        初期時刻をシフトさせるメソッド
        
        入力
        list_time: list, 時間
        list_tgt: list, シフト対象のリスト
        index_shift: int, シフトさせるインデックス
        
        出力
        list_time_shift: list, シフト後の時間
        list_tgt_shift: list, シフト後のシフト対象のリスト
        '''
        list_time_shift = list_time[index_shift:] - list_time[index_shift]
        list_tgt_shift = list_tgt[index_shift:]
        return list_time_shift, list_tgt_shift
    
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
    
    def fitting(self, list_time, list_theta):
        '''
        波形に対してラプラス変換を実行し、その逆数に対してフィットを行う
        
        入力
        list_time: list, 時間
        list_theta: list, フィットする波形
        '''
        list_s, list_theta_s = self.calc_Laplace(list_time, list_theta)
        
        list_tgt = 1/list_theta_s
        params = self.least_square(list_s[:self.range_fitting], list_tgt[:self.range_fitting])
        # print(params)
        print('伝達関数の分母のフィット結果 = ', 1,params[1]/params[0],params[2]/params[0])
        
        list_fit = self.fit_func(list_s, params)
        
        # plt.plot(list_tgt)
        # plt.plot(list_fit, 'r--')
        # plt.show()
        
        return params, list_s, list_tgt, list_fit
    
    def calc_response(self, params, list_time, mode, time_zero_cross = 0, stroke_width = 0):
        '''
        フィットした伝達関数のパラメータからインパルス応答orステップ応答を計算
        伝達関数: params[2]/(params[0]*s^2 + params[1]*s + params[2])  ※DCゲインが1になるように調整
        
        入力
        params: list, パラメータ
        list_time: list, 時間
        mode: string, インパルス応答 or STEP応答の選択
        stroke_width: float, ストローク幅。出力のSTEP応答はこれに合わせる
        
        出力
        list_theta_fitting: list, フィット関数
        '''
        # print(mode, time_zero_cross, stroke_width)
        list_time = np.array(list_time)
        a = params[0] * np.sign(params[0])
        b = params[1] * np.sign(params[0])
        c = params[2] * np.sign(params[0])
        
        if mode == 'impulse':
            lambda_t = np.sqrt(c/a - b**2/4/a**2)
            gamma = b/a
            A = 1 /np.sqrt(a*c - b**2 / 4)
            # A *= np.exp(gamma/2*time_zero_cross) # 本来は時間シフト分の振幅の補正が入るはず。これを入れると精度が落ちたため消去。
            # print(A, gamma, lambda_t)
            list_theta_fitting = A*np.exp(-gamma/2*list_time)*np.sin(lambda_t*list_time)
        
        if mode == 'step':
            A = stroke_width
            omega_n = np.sqrt(c/a)
            omega_d = omega_n * np.sqrt(1 - b**2/4/a/c)
            xi = 0.5*b/np.sqrt(a*c)
            delta = np.arctan(np.sqrt(1-xi**2)/xi)
            # print(A, omega_n, omega_d, xi, delta)
            list_theta_fitting = A*np.exp(-xi*omega_n*list_time)/np.sqrt(1-xi**2)*np.sin(omega_d*list_time + delta)
            
        return list_theta_fitting
    
    def update_log(self, params, list_s, list_theta_s, list_theta_s_fit, name):
        '''
        クラス変数dict_fit_params, dict_log_fitにフィット結果を書き込む
        
        入力
        params: list, フィット関数のパラメータ
        list_s: list, s座標
        list_theta_s, list, フィット対象データ
        list_theta_s_fit, list, フィット結果データ
        name, str, 辞書のキーの名前
        '''
        Fitting_s_space.dict_fit_params['params_' + name] = params
        Fitting_s_space.dict_fit_log['list_s_' + name] = list_s
        Fitting_s_space.dict_fit_log['list_theta_s_' + name] = list_theta_s
        Fitting_s_space.dict_fit_log['list_theta_s_fit_' + name] = list_theta_s_fit
        
        
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
        # list_omega = np.array([vector[1] for vector in list_vector]) # omegaのみ抽出
        stroke_width = list_theta[0]
        
        # 初期時刻シフト(exp(-gamma/2*t)*sin(lambda*t)になるようにする)
        index_zero_cross = self.detect_1st_zero_crossing(list_theta)
        print('zero point: time[s]=', list_time[index_zero_cross], 'theta=', list_theta[index_zero_cross])
        list_time_shift, list_theta_shift = self.calc_shift(list_time, list_theta, index_zero_cross)

        # フィットとフィット関数計算
        params, list_s, list_theta_s, list_theta_s_fit = self.fitting(list_time_shift, list_theta_shift)
        list_theta_fitting = self.calc_response(params, list_time, 'step', stroke_width = stroke_width)
        
        self.update_log(params, list_s, list_theta_s, list_theta_s_fit, '1st')
        
        list_theta_fitting_high_accuracy = [0]*len(list_theta_fitting)
        # 高精度化のための追加フィット
        if self.mode_high_accuracy == 1:
            list_tgt = list_theta - list_theta_fitting # エラーの時系列データ
            index_zero_cross = self.detect_1st_zero_crossing(list_tgt)
            list_time_shift, list_tgt_shift = self.calc_shift(list_time, list_tgt, index_zero_cross)
            print('zero point: time[s]=', list_time[index_zero_cross], 'theta=', list_theta[index_zero_cross])
        
            params, list_s, list_theta_s, list_theta_s_fit = self.fitting(list_time_shift, list_tgt_shift)
            list_theta_fitting_high_accuracy = self.calc_response(params, list_time, 'impulse', time_zero_cross = list_time[index_zero_cross])
            
            self.update_log(params, list_s, list_theta_s, list_theta_s_fit, '2nd')
        
        return list_theta_fitting + list_theta_fitting_high_accuracy
        
class test(Fitting_s_space):
    '''
    メイン機能とは関係ないクラス
    
    ラプラス変換の精度を確認するために、
    フィット関数が厳密に計算可能な関数に対してラプラス変換を実施
    '''
    
    def func(self):
        '''
        ラプラス変換する関数を定義
        
        出力
        list_time, list, 時間リスト
        list_theta, list, 角度の時系列データ
        '''
        self.gamma = 1
        self.lambda_t = 3.09
        self.delta = 0
        
        list_time = np.arange(0, 100, 0.01)
        list_theta = np.exp(-self.gamma*list_time)*np.sin(self.lambda_t*list_time+self.delta)
        # list_theta = np.exp(-self.gamma*list_time)*np.cos(self.lambda_t*list_time+self.delta)
        
        return list_time, list_theta
    
    def do_test(self):
        '''
        func()で定義した関数をラプラス変換し、伝達関数の分母を２次関数でフィットする
        '''
        list_time, list_theta = self.func()
        list_s, list_theta_s = self.calc_Laplace(list_time, list_theta)
        
        list_tgt = 1/list_theta_s
        a,b,c = self.least_square(list_s[:100], list_tgt[:100])
        list_theta_s_fit = self.fit_func(list_s, [a,b,c])
        print('フィット結果: ', a,b,c)
        
        # plt.plot(list_tgt)
        # plt.plot(list_theta_s_fit, 'r--')
        # plt.show()
        
        df = pd.DataFrame()
        df.index = list_s
        df['tgt'] = list_tgt
        df['fit'] = list_theta_s_fit
        
        df.to_excel('test.xlsx')
        
        
        
        
        
    
if __name__=='__main__':
    
    # mode_fitting = 'time_space' # 時間区間での切り貼りフィット
    data_processor = Data_Processor()
    data_processor.setup_by_hand('time_space') # パラメータと初期値入力
    data_processor.do_sim()        # 時間発展の計算
    data_processor.do_fit()        # フィッティング実行
    data_processor.write_excel()   # 結果をExcel出力
    
    # mode_fitting = 's_space' # 伝達関数推定
    data_processor.setup_by_hand('s_space') # パラメータと初期値入力
    # data_processor.do_sim()        # 時間発展の計算
    data_processor.do_fit()        # フィッティング実行
    data_processor.write_excel()   # 結果をExcel出力
    
    
    # test().do_test()
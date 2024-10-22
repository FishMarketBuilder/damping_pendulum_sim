import numpy as np
import pandas as pd

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
            'b': 0, # 減衰係数 [kg/s]
            'm': 1, # 振り子の質量 [kg]
            'g': 9.81,# 重力加速度 [m/s^2]
            'L': 1,   # 振り子の長さ [m]
        }
        self.vector_init = [np.pi/4, 0] # 初期値 [theta, omega]
        self.Nt = 1000
        self.dt = 0.01
        self.output_path = 'output.xlsx'
        
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
    
    def write_excel(self, path, list_time, list_vector):
        '''
        データとパラメータをExcelに書き込む
        
        Posシート: 角度・角速度の時間発展結果
        Paramsシート: パラメータデータ
        '''
        with pd.ExcelWriter(path) as writer:
            df_vector = pd.DataFrame(list_vector, columns=['theta', 'omega'])
            df_vector.index = list_time
            df_vector.to_excel(writer, sheet_name = 'Pos')
            
            df_params = pd.DataFrame(list(self.params.items()), columns=['Parameter', 'Value'])
            df_params.to_excel(writer, sheet_name = 'Params')
            
    def do_sim(self):
        '''
        シミュレーションの実行メソッド
        注意: パラメータ設定メソッド群(setup_*)のいずれかでパラメータ設定後に実行すること
        '''
        list_time, list_vector = Physical_SIM().calc_time_evolution(self.params, self.vector_init, self.Nt, self.dt)
        self.write_excel(self.output_path, list_time, list_vector)
    
if __name__=='__main__':
    data_processor = Data_Processor()
    data_processor.setup_by_hand()
    data_processor.do_sim()
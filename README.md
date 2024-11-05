# damping_pendulum_sim

■ディレクトリ説明
.\env
.\env\test.yml: Conda仮想環境

.\doc
.\doc\レポート.pptx: 報告用資料

.\code
.\code\main.py: 開発コード


■実行手段
.\code内で以下のコマンドを実行
python main.py

■フィットモード選択
main.py内のData_Processorクラスの変数で選択可能

方法①：時間区間での切り貼りフィット
self.mode_fitting = 'time_space' 

方法②：伝達関数推定によるフィット
self.mode_fitting = 's_space'
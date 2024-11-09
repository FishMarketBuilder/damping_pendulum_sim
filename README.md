# damping_pendulum_sim

■ディレクトリ説明
.\env
.\env\Conda\env_conda.yml: Condaユーザー向け仮想環境構築用YAML
.\env\Docker\Dockerfile: Dockerユーザー向けDockerfile

.\doc
.\doc\レポート.pptx: 報告用資料

.\code
.\code\main.py: 開発コード

.\data
各種データを格納（レポート.pptx内で使用）


■環境構築＆実行方法
== Condaユーザー向け ==
AnacondaPromptで.\env\Condaに移動し、以下を実行
conda env create -f env_conda.yml # Conda仮想環境を構築
conda activate test # 仮想環境testに入る

.\code内で以下のコマンドを実行
python main.py

== Dockerユーザー向け ==
コマンドプロンプトで.\env\Dockerに移動し、以下を実行
docker compose up -d --build  # Dockerイメージとコンテナを同時作成
docker compose exec python3 bash  # Dockerコンテナに入る

Dockerコンテナ内のディレクトリ.\damping_pendulum_simに移動し、以下を実行
python main.py


■出力
.\codeディレクトリに以下の2つのExcelファイルを出力
output_time_space.xlsx: 時間区間での切り貼りフィット結果
output_s_space.xlsx: 伝達関数推定によるフィット結果

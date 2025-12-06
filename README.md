본 프로젝트는
-reinforce
-PPO
두 가지 알고리즘으로 이루어져 있습니다.
총 3가지 seed를 사용하여 평균을 구합니다.

PPO의 경우 0.1, 0.2 clip에 대해 실험을 진행합니다.


실험 code는 pipeline.py로, 아래 커맨드로 실행 가능합니다.
python pipeline.py --plot
학습률 생성 code는 plot_curves_report.py로, 아래 커맨드로 실행 가능합니다.
python plot_curves_report.py

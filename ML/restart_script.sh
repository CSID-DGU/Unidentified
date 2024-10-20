# Description: Python 스크립트를 무한 루프로 실행하고, 스크립트가 비정상 종료되면 자동으로 재시작하는 스크립트

#!/bin/bash

# 실행할 Python 파일 이름
SCRIPT="model_torch_hyperparam_tuning.py"

# 무한 루프 시작
while true
do
    echo "Starting $SCRIPT..."
    
    # Python 스크립트를 실행
    python $SCRIPT
    
    # 종료 코드를 확인 (0이면 정상 종료, 0이 아니면 오류)
    if [ $? -eq 0 ]; then
        echo "$SCRIPT finished successfully."
        break
    else
        echo "$SCRIPT crashed with exit code $?. Restarting..."
    fi
    
    # 재시작하기 전에 잠시 대기 (5초)
    sleep 5
done

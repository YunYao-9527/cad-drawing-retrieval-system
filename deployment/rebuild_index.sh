#!/bin/bash
# CAD鍥剧焊妫€绱㈢郴缁熼噸寤虹储寮曡剼鏈紙Docker鐗堟湰锛?

set -e

echo "=========================================="
echo "閲嶅缓鍚戦噺鏁版嵁搴撶储寮?
echo "=========================================="

# 妫€鏌ocker
if ! command -v docker &> /dev/null; then
    echo "鉂?閿欒: 鏈壘鍒癉ocker"
    exit 1
fi

# 妫€鏌ュ鍣ㄦ槸鍚﹁繍琛?
if ! docker ps | grep -q "cad-retrieval-system"; then
    echo "鈿狅笍  璀﹀憡: 瀹瑰櫒鏈繍琛岋紝姝ｅ湪鍚姩..."
    ./start.sh
    sleep 10
fi

# 鎵ц閲嶅缓绱㈠紩
echo "姝ｅ湪閲嶅缓绱㈠紩..."
echo "锛堣繖鍙兘闇€瑕佸嚑鍒嗛挓锛屽彇鍐充簬鍥惧簱澶у皬锛?
echo ""

docker compose exec -T cad-retrieval python -c "
from config.config_manager import init_config, get_config
from database.vector_db import init_vector_db, get_vector_db
from monitoring.logger import setup_logger
import sys

try:
    # 鍒濆鍖栭厤缃?
    init_config()
    config = get_config()
    
    # 璁剧疆鏃ュ織
    logger = setup_logger(
        name='rebuild_index',
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file
    )
    
    logger.info('寮€濮嬮噸寤虹储寮?..')
    print('寮€濮嬮噸寤虹储寮?..', flush=True)
    
    # 鍒濆鍖栧悜閲忔暟鎹簱
    init_vector_db()
    vector_db = get_vector_db()
    
    # 閲嶅缓绱㈠紩
    count = vector_db.initialize_database()
    
    logger.info(f'绱㈠紩閲嶅缓瀹屾垚锛屽叡澶勭悊 {count} 涓浘鐗?)
    print(f'鉁?绱㈠紩閲嶅缓瀹屾垚锛屽叡澶勭悊 {count} 涓浘鐗?, flush=True)
    sys.exit(0)
    
except Exception as e:
    logger.error(f'绱㈠紩閲嶅缓澶辫触: {e}', exc_info=True)
    print(f'鉂?绱㈠紩閲嶅缓澶辫触: {e}', flush=True)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "绱㈠紩閲嶅缓鎴愬姛"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "绱㈠紩閲嶅缓澶辫触锛岃鏌ョ湅鏃ュ織"
    echo "=========================================="
    exit 1
fi


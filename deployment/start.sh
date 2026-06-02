#!/bin/bash
# CAD鍥剧焊妫€绱㈢郴缁熷惎鍔ㄨ剼鏈紙Docker鐗堟湰锛?

set -e

echo "=========================================="
echo "鍚姩 CAD鍥剧焊妫€绱㈢郴缁?
echo "=========================================="

# 妫€鏌ocker
if ! command -v docker &> /dev/null; then
    echo "鉂?閿欒: 鏈壘鍒癉ocker锛岃鍏堝畨瑁匘ocker"
    exit 1
fi

# 妫€鏌ocker Compose
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "鉂?閿欒: 鏈壘鍒癉ocker Compose锛岃鍏堝畨瑁匘ocker Compose"
    exit 1
fi

# 妫€鏌ラ暅鍍忔槸鍚﹀瓨鍦?
if ! docker images | grep -q "cad-retrieval-system"; then
    echo "鈿狅笍  璀﹀憡: 鏈壘鍒?cad-retrieval-system 闀滃儚"
    echo "   璇峰厛鍔犺浇闀滃儚: docker load -i cad-retrieval-system.tar"
    exit 1
fi

# 鍒涘缓蹇呰鐨勭洰褰?
echo "鍒涘缓蹇呰鐨勭洰褰?.."
mkdir -p data/gallery
mkdir -p models
mkdir -p qdrant_db
mkdir -p logs
mkdir -p config

# 妫€鏌ocker-compose.yml
if [ ! -f "docker-compose.yml" ]; then
    echo "鉂?閿欒: 鏈壘鍒?docker-compose.yml"
    exit 1
fi

# 鍚姩鏈嶅姟
echo "姝ｅ湪鍚姩鏈嶅姟..."
docker compose up -d

# 绛夊緟鏈嶅姟鍚姩
echo "绛夊緟鏈嶅姟鍚姩锛堟渶澶?0绉掞級..."
for i in {1..30}; do
    if docker compose ps | grep -q "Up"; then
        echo "鉁?鏈嶅姟宸插惎鍔?
        break
    fi
    sleep 1
done

# 鏄剧ず鏈嶅姟鐘舵€?
echo ""
echo "鏈嶅姟鐘舵€?"
docker compose ps

echo ""
echo "=========================================="
echo "鍚姩瀹屾垚锛?
echo "=========================================="
echo ""
echo "鏌ョ湅鏃ュ織: docker compose logs -f cad-retrieval"
echo "妫€鏌ュ仴搴? curl http://localhost:5000/health"
echo "璁块棶鍦板潃: http://localhost:5000"
echo ""


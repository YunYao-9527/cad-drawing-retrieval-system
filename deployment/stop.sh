#!/bin/bash
# CAD鍥剧焊妫€绱㈢郴缁熷仠姝㈣剼鏈紙Docker鐗堟湰锛?

set -e

echo "=========================================="
echo "鍋滄 CAD鍥剧焊妫€绱㈢郴缁?
echo "=========================================="

# 妫€鏌ocker Compose
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "鉂?閿欒: 鏈壘鍒癉ocker Compose"
    exit 1
fi

# 妫€鏌ocker-compose.yml
if [ ! -f "docker-compose.yml" ]; then
    echo "鉂?閿欒: 鏈壘鍒?docker-compose.yml"
    exit 1
fi

# 鍋滄鏈嶅姟
echo "姝ｅ湪鍋滄鏈嶅姟..."
docker compose down

# 妫€鏌ユ槸鍚﹁繕鏈夌浉鍏冲鍣?
if docker ps -a | grep -q "cad-retrieval"; then
    echo "娓呯悊娈嬬暀瀹瑰櫒..."
    docker ps -a | grep "cad-retrieval" | awk '{print $1}' | xargs docker rm -f
fi

echo ""
echo "=========================================="
echo "鏈嶅姟宸插仠姝?
echo "=========================================="
echo ""


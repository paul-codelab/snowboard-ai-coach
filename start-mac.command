#!/bin/bash
# snowcoach 一鍵安裝與啟動（macOS）
# 雙擊執行：第一次會自動安裝（約 5～10 分鐘，需要網路），之後雙擊就直接開啟圖形介面。
# 所有東西都裝在這個資料夾裡（.venv、models），刪掉資料夾就等於解除安裝。
# 加上 --check 參數時只安裝並檢查，不開啟介面（給自動測試用）。

cd "$(dirname "$0")" || exit 1
APP_DIR="$(pwd)"
export SNOWCOACH_MODELS="$APP_DIR/models"
export GLOG_minloglevel=2

fail() {
  echo ""
  echo "❌ $1"
  echo "請把這個視窗的內容截圖或複製下來，詢問提供工具的人。"
  if [ "$CHECK" != "1" ]; then read -r -p "按 Enter 關閉視窗……"; fi
  exit 1
}

CHECK=0
[ "$1" = "--check" ] && CHECK=1

echo "=============================================="
echo "  snowcoach 單板滑雪動作分析"
echo "  ⚠️ 自動分析，非專業教練建議，僅供參考"
echo "=============================================="

# 1. uv：負責下載 Python 與套件的小工具（裝在使用者自己的資料夾，不需要管理員密碼）
UV="$(command -v uv 2>/dev/null)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  echo "第一次使用：安裝必要工具（uv）……"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh || fail "安裝 uv 失敗，請確認網路連線。"
  UV="$HOME/.local/bin/uv"
fi

# 2. Python 與套件：pyproject.toml 有變（例如更新版本）時重新安裝
HASH="$(shasum pyproject.toml | cut -d' ' -f1)"
STAMP=".venv/.snowcoach-installed"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$HASH" ]; then
  echo "安裝 Python 與分析套件（第一次約 5～10 分鐘，請耐心等候）……"
  rm -rf .venv
  "$UV" venv --python 3.11 .venv || fail "建立 Python 環境失敗。"
  "$UV" pip install --python .venv/bin/python -e ".[gui,h264]" || fail "安裝套件失敗，請確認網路連線後再試一次。"
  echo "下載姿勢模型……"
  .venv/bin/snowcoach download-models || fail "下載模型失敗，請確認網路連線後再試一次。"
  echo "$HASH" > "$STAMP"
  echo "✅ 安裝完成！之後雙擊就會直接開啟。"
fi

if [ "$CHECK" = "1" ]; then
  .venv/bin/snowcoach --help > /dev/null || fail "安裝檢查失敗。"
  echo "安裝檢查完成"
  exit 0
fi

echo ""
echo "啟動中……瀏覽器會自動打開分析頁面。"
echo "分析結果也會存在：$APP_DIR/out/gui"
echo "使用完畢後，關閉這個視窗即可結束。"
echo ""
.venv/bin/snowcoach gui || fail "程式發生錯誤。"

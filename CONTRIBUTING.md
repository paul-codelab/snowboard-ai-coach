# 參與貢獻

歡迎回報問題、提出建議或送 Pull Request！自己用 AI 程式助理改進後再分享回來，也非常歡迎。

## 回報問題

到 [Issues](../../issues) 開一則，盡量附上：

- 作業系統（Mac／Windows／Linux）與安裝方式（雙擊安裝或 pip）
- 錯誤訊息的截圖或全文
- 若是分析結果怪怪的：`summary.json` 和報告截圖，以及影片的拍攝方式（自拍桿／跟拍／手機）

⚠️ 請不要把含有他人臉孔、或你沒有權利公開的影片貼到 Issues。

## 送 Pull Request

1. Fork 後開一個新分支修改。
2. 執行測試：
   ```bash
   pip install -e ".[dev,h264,gui]"
   pytest
   ```
3. 修改指標、門檻或建議時，請先讀 [`docs/design.md`](docs/design.md)，並在 PR 說明為什麼這樣改、在哪些影片上驗證過。
4. 程式碼、註解、介面文字以繁體中文為主，風格請和既有程式一致。
5. **不要提交影片**（著作權與隱私）。需要測試資料時，用 `tests/synth.py` 產生合成骨架，或在 PR 中描述你用的影片。

## 貢獻者授權協議（CLA）

本專案採**雙授權**（AGPL-3.0 + 商業授權，見 [COMMERCIAL.md](COMMERCIAL.md)）。為了讓作者能繼續提供商業授權，送出第一個 Pull Request 時，請在 PR 說明中加上這一句：

> 我已閱讀並同意 [CLA.md](CLA.md) 的貢獻者授權協議。

沒有這一句的 PR 無法合併。只回報 Issue 不需要同意 CLA。

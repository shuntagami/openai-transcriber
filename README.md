# OpenAI Transcriber

複数の OpenAI 音声認識モデルを使って音声ファイルを文字起こしするシンプルなツールです。

## インストール

### 前提条件

- Python 3.8 以上
- Poetry

### セットアップ

1. リポジトリをクローン：

```bash
git clone https://github.com/shuntagami/openai-transcriber.git
cd openai-transcriber
```

2. Poetry で依存関係をインストール：

```bash
poetry install
```

3. `.env`ファイルを作成し、OpenAI の API キーを設定：

```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

```bash
# 基本的な使い方
poetry run transcribe 音声ファイル.mp3

# または、Poetryシェル内で実行する場合
poetry shell
transcribe 音声ファイル.mp3
```

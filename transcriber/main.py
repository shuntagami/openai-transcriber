#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def generate_output_filename(output_dir="transcripts"):
    """日時を含む一意のファイル名を生成する"""
    # 出力ディレクトリが存在しない場合は作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 現在の日時を含むファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"transcript_{timestamp}.txt")

def transcribe_with_models(audio_file_path, output_file=None):
    """
    指定された音声ファイルを複数のOpenAIモデルで文字起こしする

    Args:
        audio_file_path (str): 音声ファイルへのパス
        output_file (str, optional): 出力ファイルのパス。指定しない場合は自動生成。

    Returns:
        int: 成功時は0、エラー時は1
    """
    # 環境変数を読み込む
    load_dotenv()

    # OpenAI APIキーを確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
        return 1

    # 出力ファイルのパスを決定
    if output_file is None:
        output_file = generate_output_filename()

    print(f"文字起こし中: {audio_file_path}")
    print(f"出力ファイル: {output_file}")

    # OpenAI クライアントの初期化
    client = OpenAI()

    try:
        # ファイルを開く
        with open(output_file, 'w', encoding='utf-8') as f:
            # ヘッダーを書き込む
            f.write(f"# 文字起こし結果\n")
            f.write(f"# 元ファイル: {os.path.basename(audio_file_path)}\n")
            f.write(f"# 日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# モデル: gpt-4o-transcribe\n\n")
            f.write("\n===== gpt-4o-transcribe =====\n\n")
            f.flush()  # ファイルに即時書き込み

        # 音声ファイルを開いて文字起こし
        with open(audio_file_path, "rb") as audio_file:
            print("Transcribing with gpt-4o-transcribe model...")
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language="ja"
            )

            # 結果をファイルに書き込む
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(transcription.text)
                f.flush()

            # コンソールにも表示
            print("\n===== gpt-4o-transcribe =====\n", transcription.text)
            print(f"\n文字起こし結果をファイル '{output_file}' に保存しました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{audio_file_path}' が見つかりません。")
        return 1
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1

    return 0

def main():
    """
    メイン関数
    """
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="OpenAIモデルによる音声認識")
    parser.add_argument("audio_file", help="文字起こしする音声ファイルのパス")
    parser.add_argument("-o", "--output", help="出力ファイルのパス (指定しない場合は自動生成)")
    parser.add_argument("--output-dir", default="transcripts", help="出力ディレクトリ (デフォルト: transcripts)")

    args = parser.parse_args()

    # 出力ファイルのパスを決定
    output_file = args.output
    if output_file is None and args.output_dir != "transcripts":
        # 出力ディレクトリが指定されている場合、そのディレクトリ内にファイルを生成
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"transcript_{timestamp}.txt")

    return transcribe_with_models(args.audio_file, output_file)

if __name__ == "__main__":
    sys.exit(main())

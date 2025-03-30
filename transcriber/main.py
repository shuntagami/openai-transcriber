#!/usr/bin/env python3
import os
import sys
import argparse
import math
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

def generate_output_filename(output_dir="transcripts"):
    """日時を含む一意のファイル名を生成する"""
    # 出力ディレクトリが存在しない場合は作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 現在の日時を含むファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"transcript_{timestamp}.txt")

def split_audio(audio_file_path, max_duration=1250):
    """
    音声ファイルを指定された最大時間（秒）で分割する

    Args:
        audio_file_path (str): 音声ファイルへのパス
        max_duration (int): 分割する最大時間（秒）

    Returns:
        list: 一時的に作成された分割ファイルのパスのリスト
    """
    print(f"音声ファイルを {max_duration} 秒ごとに分割しています...")

    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(audio_file_path)

    # 分割数を計算
    total_duration = len(audio) / 1000  # ミリ秒から秒に変換
    num_segments = math.ceil(total_duration / max_duration)

    if num_segments <= 1:
        print("音声ファイルは分割の必要がありません。")
        return [audio_file_path]

    # 一時ファイルを保存するディレクトリ
    temp_dir = Path("temp_audio_segments")
    temp_dir.mkdir(exist_ok=True)

    # 分割ファイルのパスリスト
    segment_paths = []

    for i in range(num_segments):
        start_ms = i * max_duration * 1000
        end_ms = min((i + 1) * max_duration * 1000, len(audio))

        segment = audio[start_ms:end_ms]

        # 一時ファイルのパス
        segment_path = temp_dir / f"segment_{i}.mp3"
        segment.export(segment_path, format="mp3")
        segment_paths.append(str(segment_path))

        print(f"セグメント {i+1}/{num_segments} を作成しました: {segment_path}")

    return segment_paths

def transcribe_with_models(audio_file_path, output_file=None, max_duration=1250):
    """
    指定された音声ファイルを複数のOpenAIモデルで文字起こしする
    長い音声ファイルは自動的に分割して処理する

    Args:
        audio_file_path (str): 音声ファイルへのパス
        output_file (str, optional): 出力ファイルのパス。指定しない場合は自動生成。
        max_duration (int): 分割する最大時間（秒）

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
        # 音声ファイルを分割
        segment_paths = split_audio(audio_file_path, max_duration)

        # ファイルを開く
        with open(output_file, 'w', encoding='utf-8') as f:
            # ヘッダーを書き込む
            f.write(f"# 文字起こし結果\n")
            f.write(f"# 元ファイル: {os.path.basename(audio_file_path)}\n")
            f.write(f"# 日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# モデル: gpt-4o-transcribe\n\n")
            f.write("\n===== gpt-4o-transcribe =====\n\n")
            f.flush()  # ファイルに即時書き込み

        full_transcription = ""

        # 各セグメントを文字起こし
        for i, segment_path in enumerate(segment_paths):
            print(f"セグメント {i+1}/{len(segment_paths)} を文字起こし中...")

            # 音声ファイルを開いて文字起こし
            with open(segment_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language="ja"
                )

                # 結果を追加
                full_transcription += transcription.text + "\n\n"

                # 結果をファイルに書き込む
                with open(output_file, 'a', encoding='utf-8') as f:
                    if i > 0:
                        f.write("\n--- セグメント区切り ---\n\n")
                    f.write(transcription.text + "\n")
                    f.flush()

        # 一時ファイルを削除（オリジナルファイルは除く）
        if len(segment_paths) > 1:
            print("一時ファイルを削除しています...")
            for path in segment_paths:
                if path != audio_file_path:
                    os.remove(path)
            os.rmdir("temp_audio_segments")

        # コンソールにも表示
        print("\n===== 文字起こし完了 =====\n")
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
    parser.add_argument("--max-duration", type=int, default=1250, help="分割する最大時間（秒）(デフォルト: 1250)")

    args = parser.parse_args()

    # 出力ファイルのパスを決定
    output_file = args.output
    if output_file is None and args.output_dir != "transcripts":
        # 出力ディレクトリが指定されている場合、そのディレクトリ内にファイルを生成
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"transcript_{timestamp}.txt")

    return transcribe_with_models(args.audio_file, output_file, args.max_duration)

if __name__ == "__main__":
    sys.exit(main())

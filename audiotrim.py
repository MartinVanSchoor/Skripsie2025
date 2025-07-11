from pydub import AudioSegment

audio = AudioSegment.from_mp3("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source1_martin.mp3")

trimmed = audio[2_000:7_000]

trimmed.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source1_martin.wav", format="wav")
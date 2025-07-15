from pydub import AudioSegment

audio = AudioSegment.from_mp3("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source2_werner.mp3")

trimmed = audio[500:10_000]

trimmed.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source2_werner.wav", format="wav")
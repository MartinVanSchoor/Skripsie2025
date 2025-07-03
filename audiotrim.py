from pydub import AudioSegment

audio = AudioSegment.from_mp3("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/Trump_WEF_2018.mp3")

trimmed = audio[0:180_000]

trimmed.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target1_trump.wav", format="wav")
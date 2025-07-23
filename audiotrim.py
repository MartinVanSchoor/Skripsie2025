from pydub import AudioSegment

audio = AudioSegment.from_mp3("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source/Voice 250617_110035.mp3")

trimmed = audio[1_000:31_000]

trimmed.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source/source1_martin.wav", format="wav")
from pydub import AudioSegment

audio = AudioSegment.from_mp3("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source/AUD-20250724-WA0026.mp3")

#trimmed1 = audio[0:180_000]

audio.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source/source3_theo.wav", format="wav")
from pydub import AudioSegment

audio1 = AudioSegment.from_wav("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target1_trump180.wav")
audio2 = AudioSegment.from_wav("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target2_rfk180.wav")
audio3 = AudioSegment.from_wav("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target3_obama180.wav")

trimmed1 = audio1[0:2_500]
trimmed2 = audio2[0:2_500]
trimmed3 = audio3[0:2_500]

trimmed1.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target1_trump2_5.wav", format="wav")
trimmed2.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target2_rfk2_5.wav", format="wav")
trimmed3.export("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target3_obama2_5.wav", format="wav")
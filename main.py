from structs import Parser, Bart
#parser = Parser("whisper")
#parser.eval_dataset("dataset/dataset1", "output/whisper_output_1.jpg")
#parser.eval_dataset("dataset/dataset2", "output/whisper_output_2.jpg")
#parser.eval_dataset("dataset/dataset3", "output/whisper_output_3.jpg")
#parser.eval_dataset("dataset/dataset1", "output/whisper_output_1.jpg")
#parser.eval_dataset("dataset/dataset2", "output/whisper_output_2.jpg")
#parser.eval_dataset("dataset/dataset3", "output/whisper_output_3.jpg")
#transcript = parser.parse_all_console("input.mp4", save_transcript=True, output_dir="output")
#proc_lines = []
#with open("output/transcript.txt", "r") as f:
#    lines = f.readlines()
#    for line in lines:
#        proc_line = parser.post_process(line, save_transcript=False)
#        print(proc_line)
#        proc_lines.append(proc_line)
#
#with open("output/proc_transcript.txt", "w") as f:
#    f.write("\n".join(proc_lines))
bart = Bart("bart")

with open("output/proc_transcript.txt", "r") as f,open("output/bart.txt", "w") as f2:
    raw_text = " ".join(f.readlines())
    summary = bart.summarize(raw_text)
    import pdb;pdb.set_trace()
    f2.write(summary)
from structs import Parser
parser = Parser("whisper")
#parser.eval_dataset("dataset/dataset1", "output/whisper_output_1.jpg")
#parser.eval_dataset("dataset/dataset2", "output/whisper_output_2.jpg")
#parser.eval_dataset("dataset/dataset3", "output/whisper_output_3.jpg")
parser.eval_dataset("dataset/dataset1", "output/whisper_output_1.jpg")
parser.eval_dataset("dataset/dataset2", "output/whisper_output_2.jpg")
parser.eval_dataset("dataset/dataset3", "output/whisper_output_3.jpg")
import pdb;pdb.set_trace()
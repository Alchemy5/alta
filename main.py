from structs import Parser
parser = Parser("input.mp4")
transcript = parser.parse_all_console()
corrected_transcript = parser.post_process(transcript, save_transcript=True, output_dir = "output")
import pdb;pdb.set_trace()
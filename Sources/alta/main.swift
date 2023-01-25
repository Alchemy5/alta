import Foundation
import PythonKit

class Dataset {
    /*
    Wrapper class for use with popular LibriSpeech dataset.
    */
    var dirPath: String
    var filenameTranscriptDict:[String:String] = [:]
    init(dirPath:String){
        self.dirPath = dirPath
        let fm = FileManager.default
        var filenames = [String]()
        do {
            filenames = try fm.contentsOfDirectory(atPath : dirPath)
        }
        catch{
        }
        var transcriptPath = ""
        for filename in filenames {       
            if filename.contains(".trans"){
                transcriptPath = self.dirPath + "/" + filename
                break
            }
        }
        var data = ""
        do{
            data = try String(contentsOfFile:transcriptPath, encoding:.utf8)
        }
        catch{
        }
        let transcriptLines = data.split(separator:"\n")
        for line in transcriptLines{
            let filename = self.dirPath + "/" + line.split(separator:" ")[0] + ".flac"
            let transcriptLine = (line.split(separator:" ")[1...]).joined(separator:" ")
            filenameTranscriptDict[filename] = transcriptLine
        }
    }
}

class Model {
    func parse(inputPath:String)->String{ 
        /*
        Wrapper to parse audio file using Whisper.
        */  
        let sys = Python.import("sys")
        sys.path.append("/home/varun/alta/")
        let swiftUtils = Python.import("swift_utils")
        let audioStr = String(swiftUtils.parseTool(inputPath))!
        return audioStr
    }
}

class Eval {
    func levDis(w1: String, w2: String) -> Int {
        /*
        Edit distance helper function
        */
        let empty = [Int](repeating:0, count: w2.count)
        var last = [Int](0...w2.count)

        for (i, char1) in w1.enumerated() {
            var cur = [i + 1] + empty
            for (j, char2) in w2.enumerated() {
                cur[j + 1] = char1 == char2 ? last[j] : min(last[j], last[j + 1], cur[j]) + 1
            }
            last = cur
        }
        return last.last!
    }

    func eval_dataset(datasetPath:String, outputPath:String)->[Int]{
        /*
        Evaluates model on a LibreSpeech Dataset.
        */
        let dataset = Dataset(dirPath:datasetPath)
        let model = Model()
        var editDistances = [Int]()
        let audioFilenames = dataset.filenameTranscriptDict.keys
        for filename in audioFilenames{
            var modelOutput = model.parse(inputPath:filename).uppercased()
            var model_output_no_punctuation = ""
            for el in modelOutput{
                if !el.isPunctuation{
                    model_output_no_punctuation += String(el)
                } 
            }
            let transcriptLine = String(dataset.filenameTranscriptDict[filename]!)
            let editDistance = levDis(w1:model_output_no_punctuation, w2:transcriptLine)
            print(editDistance)
            editDistances.append(editDistance)
        }
        return editDistances
    }
}
let eval = Eval()
let editDistances = eval.eval_dataset(datasetPath:"dataset/dataset1", outputPath:"output/swiftOutput")
print(editDistances)
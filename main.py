from structs import Parser
parser = Parser("input.mp4")
#transcript = parser.parse_all_console()
#print(transcript)
transcript = ["Hello, hello, are you hearing me? Yeah, I can hear you can hear me. Yeah, I'm hearing you. Great Great. How are you? Pretty good. I'm pretty good. I actually just found out that as a video interview. So yeah, okay It's okay Is this your first time using this platform? No, I'm just before but from two or three months ago. So Hey, I've used I've used other ones, but this is my first time using", "This is my first time using this particular one. Yeah. OK. Are you a range software engineer, or are you seeking for a new opportunity? I'm looking for a new opportunity. Yeah. Yeah. What about you? I'm working recently in Plumburg, London office. Oh, OK. Yeah. OK. Yeah, I'm in Bay Area, California. OK. OK. Cool. Nice to meet you.", "you. So it's a poll that I will start asking you first. Okay. So as you see, do you want some time to read the question first? Yeah, let me see. Okay. So it's a BST successor search. Yeah. Okay. Okay, so in a binary search,", "So in a binary search tree, an inorder successor of a node is defined as the node with the smallest key greater than the key of the input node. Given a node in a binary search tree, you're asked to write a function, find inorder successor, that returns the inorder successor of input node. If input node has a node, it's fine.", "If it has no in-order successor return null. Yes. So what you need to do given a node, you have to get the smallest node which have a smallest value greater than the given one. If it exists, if it doesn't exist, just return null. Okay. In the existing code, you have the struct for the node which you will code in Java. Yeah, I'm going to code in Java. Okay.", "So, you have a class for node that contains the key and lift Android and you have access to the parent itself and the constructor takes the value for the key and for the method which you need to fill which is find in order successors, take the input node and return the node which is the target one and null it if it doesn't exist and you have some logic for the insert", "but you don't need to take care about that. Okay. I'm sorry, can you say that last part again? You don't need to take care about the logic for the insert method. Which is this part. Oh, okay, the insert method. Yeah, it doesn't matter for you. Okay. You will not need something like that. So, all you need is just everything right here, code here, in this part. Or this problem.", "Okay. And then so the input isn't the necessary of the root. It could be any node in the tree. Yeah, it could be any node. Something like the tree example on the list. Okay. Yeah. If I give a new something like nine. Okay. So I'm searching for the node which contains the element", "than nine and the same time is the smallest one so it will be 11 right? Oh okay. So am I expecting when I give you nine as an input you got me as a note which is 11 this one something different if I go if I give you 14 you need to return the root which is 20. Okay. Because it's the first note greater than 14. I see. Okay.", 'So if I do have a nine, I want to return 11. If I have a 12, I want to return 14. Yes. Right. Okay. And I do have access to the parent nodes. Exactly. You have access to the parent nodes. Okay. So just going over a couple examples, say you give me 11.', "11 11 yeah then the in order would be 12 right exactly okay So I'm just kind of think of the different cases that we can have here okay", "So, yeah, let's just say use 12 as an example. It would be 4. So, if we're using 12 and we're going"]
corrected_transcript = parser.post_process(transcript)
print(corrected_transcript == transcript)
import pdb;pdb.set_trace()
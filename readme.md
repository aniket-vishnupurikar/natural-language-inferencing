# Natural Language Inferencing (Textual Entailment)
#### Problem statement and the data: https://nlp.stanford.edu/projects/snli/

## Introduction:

Natural Language processing (NLP) is the area of computer science and artificial intelligence which deals with interactions between computers and languages spoken by humans. Specifically, it focuses on programming computers to make them process, analyse and understand natural languages. Natural language inferencing (NLI) or textual entailment is a sub-field of NLP. NLI is a task of determining whether the given “hypothesis” and “premise” logically follow (entailment) or unfollow (contradiction) or are undetermined (neutral) to each other. Given two sentences the task of NLI is to determine if sentence 2 follows(entails), unfollows(contradicts) or is independent(neutral) of sentence 1. The following examples demonstrate this task:
-------------------------------------------------------------------------------------------------------------------------
|Text (Premise)					  |			Hypothesis				|		Judgement (Label) |
|										
|A soccer game with multiple males playing. |	Some men are playing a sport.			|		entailment		|
|							  
|A black race car starts up in front of a   |	A man is driving down a lonely road.	|		contradiction	|
|crowd of people.					  |								|					|
|							
|An older and a younger man smiling.	  |	Two men are smiling and laughing at the	|		neutral		|
|							  |	cats playing on the floor. 			|					|					
|-------------------------------------------|---------------------------------------------|-----------------------------| 
					Table1: Examples of different tasks in NLI


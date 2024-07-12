

# Author: Sai Vikhyath Kudhroli
# Date: 5 July 2024


import os
import json
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain


def read_csv_data():
    """ Documentation goes here """

    fileName = os.listdir("Documents/RequirementsMatching/Input/")[0] 
    csvData = pd.read_csv("Documents/RequirementsMatching/Input/" + str(fileName))

    return fileName, csvData


def meets_requirements(candidate, feature, value, requirement):
    """ Documentation goes here """

    try:

        document = [Document(page_content="")]

        llm = ChatOllama(model="gemma2")
        chain = load_qa_chain(llm=llm)
        
        prompt = f"""
            Given the candidate's name, candidate's feature, value and a requirement. Return True if the candidate's value for the feature meets the requirement, Otherwise return False.

            Comparison Operators Description:
                1. < is strictly less than
                2. > is strictly greater than
                3. >= is greater than or equal to
                4. <= is less than or equal to
                5. = is equal to

            Candidate Name: {candidate}
            Feature: {feature}
            Value: {value}
            Requirements: {requirement}

            If the value meets the requirement, then JUST return True
            If the value does not meet the requirement, then JUST return False

            JUST RETURN True/False, NOTHING ELSE.

            Output format: True/False

        """

        with get_openai_callback() as callback:
            answer = chain.run(input_documents=document, question=prompt)
            print("=" * 100)
            print(str(candidate) + "'s " + str(feature) + " is " +  str(value))
            print("Requirement: ", requirement)
            print("Requirement met: ", answer)
            print("=" * 100)
            return answer
    
    except Exception as e:
        print("Exception while matching requirements: " + str(e))


def computeCandidateScore(candidateSatisfies: list, requirementWeights: list) -> float:
    """ Documentation goes here """
    
    score = 0

    for idx in range(len(candidateSatisfies)):
        score += (candidateSatisfies[idx] * requirementWeights[idx])
    
    return score


def generate_suggestion():
    """ Documentation goes here """

    fileName, dataframe = read_csv_data()

    requirements = dataframe["Requirements"].tolist()
    features = dataframe["Features"].tolist()
    requirementWeights = dataframe["Requirement Weights"].tolist()

    candidates = {}

    for candidate in dataframe.columns[3:]:
        candidateFeatures = {}
        for i, feature in enumerate(dataframe["Features"]):
            candidateFeatures[feature] = dataframe[candidate][i]
        candidates[candidate] = candidateFeatures

    candidatesMatchingRequirements = {}
    requirementsMatchingCandidates = {}

    for requirement in requirements:
        requirementsMatchingCandidates[requirement] = []

    candidatesScores = {}
    bestCandidates = []
    bestCandidateScore = 0

    for candidate, candidateFeatures in candidates.items():
        
        candidateSatisfies = []
        idx = 0

        for feature, value in candidateFeatures.items():
            if "true" in meets_requirements(candidate, feature, value, requirements[idx]).lower():
                candidateSatisfies.append(1)
                requirementsMatchingCandidates[requirements[idx]].append(candidate)                
            else:
                candidateSatisfies.append(0)
            idx += 1
        
        candidatesMatchingRequirements[candidate] = candidateSatisfies
        print("*" * 150)
        print("Candidate vs Requirements: ", candidatesMatchingRequirements)
        print("*" * 150)
        print("*" * 150)
        print("Requirements vs Candidates: ", requirementsMatchingCandidates)
        print("*" * 150)
        candidateScore = computeCandidateScore(candidateSatisfies, requirementWeights)
        if candidateScore >= bestCandidateScore:
            bestCandidateScore = candidateScore
            bestCandidates.append(candidate)
        candidatesScores[candidate] = candidateScore
    
    print("+" * 150)
    print("Candidates Scores: ", candidatesScores)
    print("+" * 150)

    dataframe["Candidates Satisfying Requirements"] = dataframe["Requirements"].map(requirementsMatchingCandidates)

    addScoresToDataframe = candidatesScores
    addScoresToDataframe["Features"] = "Candidate's Score"
    addScoresToDataframe["Requirements"] = "Max Score"
    addScoresToDataframe["Requirement Weights"] = "-"
    addScoresToDataframe["Candidates Satisfying Requirements"] = "-"

    addScoresDataframe = pd.DataFrame([addScoresToDataframe])

    dataframe = pd.concat([dataframe, addScoresDataframe], ignore_index=True)

    dataframe.to_csv("Documents/RequirementsMatching/Output/Result_" + fileName)

    dataframe = dataframe.astype(str)

    print(dataframe)
    print("Best Candidates: ", bestCandidates)


    return bestCandidates, dataframe
                     


if __name__ == "__main__":
    generate_suggestion()




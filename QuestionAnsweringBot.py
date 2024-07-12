


# Author: Sai Vikhyath Kudhroli
# Date: 25 June 2024


import os
import fitz
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_pdf_text(pdfPath: str) -> str:
    """ Documentation goes here """

    def process_page(pageNumber):
        """ Documentation goes here """
        try:
            pytesseract.pytesseract.tesseract_cmd = r"C:/Users/mohit/AppData/Local/Tesseract-OCR/tesseract.exe"
            page = pdfDocument[pageNumber]
            # Convert the page into an image
            image = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            # Enhancements on the image
            enhancedImage = ImageEnhance.Contrast(Image.frombytes("RGB", (image.width, image.height), image.samples)).enhance(2.0)
            # Perform OCR on the image
            pageText = pytesseract.image_to_string(enhancedImage, lang="eng")
            return pageText
        except Exception as e:
            print("Error processing page " + str(pageNumber) + ": " + str(e))
            return ""

    pdfText = ""

    path = os.path.split(pdfPath)
    head, tail = path[0], path[1]
    fileName, extension = tail.split(".")

    if not os.path.exists("Documents/QA/AlreadyRead/" + str(fileName) + ".txt"):

        pdfDocument = fitz.open("Documents/QA/NewlyUploaded/" + str(fileName) + "." + str(extension))

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_page, range(pdfDocument.page_count))
            for result in results:
                pdfText += result + "\n\n"
        
        with open("Documents/QA/AlreadyRead/" + str(fileName) + ".txt", "w") as file:
            file.write(pdfText)
        
        pdfDocument.close()
    
    else:
        with open("Documents/QA/AlreadyRead/" + str(fileName) + ".txt", "r") as file:
            pdfText = file.read()
    
    return pdfText



def answerUsingLLM(pdfContent: str, query: str):
    """ Documentation goes here """

    def create_chunks(text: str) -> list:
        """ Documentation goes here """
        
        textSplitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=7424, chunk_overlap=128)
        chunks = textSplitter.split_text(text)
        return chunks

    try:
        
        llm = ChatOllama(model="llama3")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        chain = load_qa_chain(llm=llm)
        chunks = create_chunks(pdfContent)

        try:
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            documents = vectorStore.similarity_search(query=query, k=6)
            prompt = f"""
                Given the question, Answer the question based ONLY on the input documents given.
                Instructions:
                    1. The answer to the question may be present in the input documents either directly or indirectly, it is also possible that the answer may not be in the input documents.
                    2. If the answer to the question is directly or indirectly present in the input documents, please output the answer.
                    3. If the answer to the question is not present in the input documents, please output "The information is not found in the document" and also the reason for it.
                    4. Always give clear and detailed answers for the questions.
                    5. If asked questions like who always try to find the names for the question.
            
                Question: {query}
            """

            with get_openai_callback() as callback:
                answer = chain.run(input_documents=documents, question=prompt)
                return answer
        
        except Exception as e:
            print("Vector Store exception: " + str(e))

    except Exception as e:
        print("Exception in vector search: " + str(e)) 
    
    return "Unexpected error"


def question_answering(query: str):
    """ Documentation goes here """

    print("Query passed: ", query)

    fileNameWithExtension = os.listdir("Documents/QA/NewlyUploaded/")[0]
    print("Filename: ", fileNameWithExtension)

    pdfPath = "Documents/QA/NewlyUploaded/" + str(fileNameWithExtension)
    print("PDF path: ", pdfPath)

    fileName, extension = fileNameWithExtension.split(".")

    pdfContent = ""

    pdfContent = read_pdf_text("Documents/QA/NewlyUploaded/" + str(fileNameWithExtension))

    extractedInformation = answerUsingLLM(pdfContent, query)

    print(extractedInformation)

    return extractedInformation



if __name__ == "__main__":
    question_answering("What was the order total?")


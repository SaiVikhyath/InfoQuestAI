

# Author: Sai Vikhyath Kudhroli
# Date: 5 July 2024


import os
import json
import fitz
import base64
import pytesseract
import pandas as pd
from PIL import Image, ImageEnhance
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatOllama
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


def encode_string(string: str):
    """ Documentation goes here """
    encodedBytes = base64.b64encode(string.encode("utf-8"))
    encodedID = encodedBytes.decode("utf-8")
    return encodedID


def decode_ID(ID: str):
    """ Documentation goes here """
    decodedBytes = base64.b64decode(ID.encode("utf-8"))
    decodedString = decodedBytes.decode("utf-8")
    return decodedString


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

    if not os.path.exists("Documents/Insights/AlreadyRead/" + str(fileName) + ".txt"):

        pdfDocument = fitz.open("Documents/Insights/NewlyUploaded/" + str(fileName) + "." + str(extension))

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_page, range(pdfDocument.page_count))
            for result in results:
                pdfText += result + "\n\n"
        
        with open("Documents/Insights/AlreadyRead/" + str(fileName) + ".txt", "w") as file:
            file.write(pdfText)
        
        pdfDocument.close()
    
    else:
        with open("Documents/Insights/AlreadyRead/" + str(fileName) + ".txt", "r") as file:
            pdfText = file.read()
    
    return pdfText


def extract_insights_using_llm(text: str):
    """ Documentation goes here """

    def create_chunks(text: str) -> list:
        """ Documentation goes here """
        
        textSplitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=7424, chunk_overlap=128)
        chunks = textSplitter.split_text(text)
        return chunks

    try:
        
        llm = ChatOllama(model="llama3")
        chain = load_qa_chain(llm=llm)
        chunks = create_chunks(text)

        try:
            
            documents = []
            for chunk in chunks:
                document =  Document(page_content=chunk)
                documents.append(document)

            prompt = """
                Given a text, Extract the following information ONLY from the text - Order ID, Name of person who placed the order, Items that were ordered, Date the order was placed, Total cost of the order, Payment method used. Also output the result EXACTLY in the format given below.

                Output format:
                {
                    "Order ID": "extracted order id",
                    "Order Placed By": "extracted name of the person who placed the order",
                    "Items Ordered": {
                        "extracted item 1": "price of the extracted item 1",
                        "extracted item 2": "price of the extracted item 2",
                        "extracted item 3": "price of the extracted item 3",
                        "extracted item 4": "price of the extracted item 4",
                    },
                    "Total Cost": "extracted total cost of the order",
                    "Payment Method": "extracted payment method used"
                }

                Instructions:
                    1. Pay ATMOST attention to the format. ONLY USE double quotes, DON'T use single quotes. Ensure proper opening and closing of braces.
                    2. The details may be present in the input documents either directly or indirectly, it is also possible that the details may not be in the input documents.
                    3. If the details are directly or indirectly present in the input documents, please output in the above given format.
                    4. If the details are not present in the input documents, please output "Not found"
                    5. PAY CLOSE ATTENTION TO ITEMS ORDERED FORMAT. The output returned SHOULD EXACTLY MATCH the output format given above.

            """

            with get_openai_callback() as callback:
                answer = chain.run(input_documents=documents, question=prompt)
                return answer
        
        except Exception as e:
            print("Vector Store exception: " + str(e))

    except Exception as e:
        print("Exception in vector search: " + str(e)) 
    
    return "Unexpected error"


def extract_json_from_text(text: str, trying: int):
    """ Documentation goes here """

    trying += 1

    if trying > 3:
        return {
                    "Order ID": "Unable to extract",
                    "Order Placed By": "Unable to extract",
                    "Items Ordered": {
                    },
                    "Total Cost": "Unable to extract",
                    "Payment Method": "Unable to extract"
                }

    text = text.replace("'", '"')

    try:
        startIndex = text.find("{")
        endIndex = text.rfind("}") + 1
        jsonString = text[startIndex:endIndex]
        jsonObject = json.loads(jsonString)
        return jsonObject

    except Exception as e:
        print("Error extracting JSON: " + str(e) + " trying again...")
        try:
            llm = ChatOllama(model="llama3")
            chain = load_qa_chain(llm=llm)

            try:
                document =  [Document(page_content=text)]

                prompt = """
                    Given a JSON object and an error, please correct the error and return the JSON object in the below format.

                    Output format:
                    {
                        "Order ID": "extracted order id",
                        "Order Placed By": "extracted name of the person who placed the order",
                        "Items Ordered": {
                            "extracted item 1": "price of the extracted item 1",
                            "extracted item 2": "price of the extracted item 2",
                            "extracted item 3": "price of the extracted item 3",
                        }
                        "Total Cost": "extracted total cost of the order",
                        "Payment Method": "extracted payment method used"
                    }

                    Instructions:
                        1. Pay ATMOST attention to the format. ONLY USE double quotes, DON'T use single quotes. Ensure proper opening and closing of braces.
                        2. The details may be present in the input documents either directly or indirectly, it is also possible that the details may not be in the input documents.
                        3. If the details are directly or indirectly present in the input documents, please output in the above given format.
                        4. If the details are not present in the input documents, please output "Not found"
                        5. PAY CLOSE ATTENTION TO ITEMS ORDERED FORMAT. The output returned SHOULD EXACTLY MATCH the output format given above.
                """

                with get_openai_callback() as callback:
                    answer = chain.run(input_documents=document, question=prompt)
                    jsonObject = extract_json_from_text(answer, trying)
                    if type(jsonObject) is not dict:
                        return None
                    return jsonObject
            
            except Exception as e:
                print("Vector Store exception: " + str(e))

        except Exception as e:
            print("Exception in vector search: " + str(e)) 


def generate_insights():
    """ Documentation goes here """

    allFileNames = ""

    insightsDataframe = pd.DataFrame(columns=["File Name", "Order ID", "Order Placed By", "Total Cost", "Payment Method", "Item", "Item Cost"])

    fileNamesWithExtension = os.listdir("Documents/Insights/NewlyUploaded/")

    for  fileNameWithExtension in fileNamesWithExtension:

        allFileNames += fileNameWithExtension

        pdfContent = read_pdf_text("Documents/Insights/NewlyUploaded/" + str(fileNameWithExtension))

        extractedInsights = extract_insights_using_llm(pdfContent)

        insightsJSON = extract_json_from_text(extractedInsights, 0)

        orderedItems = insightsJSON.pop("Items Ordered")
        itemsDataframe = pd.DataFrame(list(orderedItems.items()), columns=["Item", "Item Cost"])

        insightsJSON["File Name"] = fileNameWithExtension
        orderDataframe = pd.DataFrame([insightsJSON] * len(itemsDataframe))

        finalDataframe = pd.concat([orderDataframe, itemsDataframe], axis=1)

        insightsDataframe = pd.concat([insightsDataframe, finalDataframe], ignore_index=True)

    uniqueID = encode_string(allFileNames)

    insightsDataframe.to_csv("Documents/Insights/Results/" + str(uniqueID) + ".csv", index=False, mode="w", header=True)

    return insightsDataframe


if __name__ == "__main__":
    insights = generate_insights()
    print(insights)
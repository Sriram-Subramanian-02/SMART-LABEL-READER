import os
import google.generativeai as genai
import cv2
import matplotlib.pyplot as plt
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = "AIzaSyA71WwAQXEojvbcI_Wq3_tC28Rc2Z5gzSs"


def get_ingredients(image_path="F:\\psg\\sem_9\\ir\\package\\input_data\\hide_and_seek.jpeg"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA71WwAQXEojvbcI_Wq3_tC28Rc2Z5gzSs"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # image = cv2.imread(f'{image_path}')

    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_rgb)
    # plt.axis('off')
    # plt.show()

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract all the ingredients and nutrients information from the given product and return it as a string. If the details are not available or clear return the name of the product. Don't include any other text in the output.",
            },
            {
                "type": "image_url", "image_url": f"{image_path}"
            },
        ]
    )

    res = llm.invoke([message])
    import time
    time.sleep(2)

    return res.content


def get_harmful_ingredient(ingredients):
    genai.configure()
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Give only one ingredient from the following that is most harmful to the human body.\n{ingredients}")

    return response.text
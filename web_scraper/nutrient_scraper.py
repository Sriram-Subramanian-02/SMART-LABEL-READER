import requests
from bs4 import BeautifulSoup


def get_base_url(age, sex, height, weight, activity):
    activity_mapper = {
        "Basal Metabolic Rate (BMR)": 1,
        "Sedentary: little or no exercise": 1.2,
        "Light: exercise 1-3 times/week": 1.375,
        "Moderate: exercise 4-5 times/week": 1.465,
        "Active: daily exercise or intense exercise 3-4 times/week": 1.55,
        "Very Active: intense exercise 6-7 times/week": 1.725,
        "Extra Active: very intense exercise daily, or physical job": 1.9
    }

    return f"""
        https://www.calculator.net/macro-calculator.html?ctype=metric&cage={age}&csex={sex}&cheightmeter={height}&ckg={weight}&cactivity={activity_mapper[activity]}&x=Calculate
    """


def extractor(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to load page: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    result_boxes = soup.find_all('td', class_='result_box')

    return [box.decode_contents() for box in result_boxes]



def get_nutrients(html_contents):
    data = {}
    range_data = {}

    keys = ['protein', 'carbs', 'fat', 'sugar', 'saturated_fat', 'energy']

    for index, content in enumerate(html_contents):
        inner_soup = BeautifulSoup(content, 'html.parser')
        text = inner_soup.get_text(strip=True)

        if index < len(keys):
            if "Range" in content:
                clean_text = text.split("Range")[0].strip()
                data[keys[index]] = clean_text

                range_part = content.split("Range:")[-1].strip().split('</div>')[0].strip()
                range_data[keys[index]] = range_part
            else:
                data[keys[index]] = text

    return data, range_data


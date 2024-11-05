import requests


def analyze_claim(claim, ingredients):
    base_url = "https://cwbackend-a3332a655e1f.herokuapp.com/claims/analyze"
    params = {
        'claim': claim,
        'ingredients': ingredients
    }
    
    try:
        # Send a GET request to the API with the claim and ingredients as parameters
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse and return the JSON response
            print(response.json())
            return response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Error while making the request: {e}")
        return None

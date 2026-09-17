import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


def get_listings():
    """Fetch vehicle listings from Auto.dev API."""

    load_dotenv()

    api_key = os.getenv("AUTO_DEV_API_KEY")
    base_url = os.getenv("AUTO_DEV_BASE_URL")

    if not api_key:
        raise ValueError("AUTO_DEV_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    url = f"{base_url}/listings"

    params = {
        "vehicle.make": "Toyota",
        "vehicle.year": "2020-2026",
        "retailListing.price": "2000-11000",
        "limit": 20,
        "page": 1
    }

    all_listings = []

    while url:

        response = requests.get(
            url,
            headers=headers,
            params=params if url.endswith("/listings") else None,
            timeout=30
        )

        # Handle API rate limiting
        if response.status_code == 429:
            print("\nAPI rate limit reached.")
            print(f"Retrieved {len(all_listings)} listings before the limit.")
            print("Please wait before making another API request.")
            break

        response.raise_for_status()

        data = response.json()

        all_listings.extend(data.get("data", []))

        print(f"Retrieved {len(all_listings)} listings...")

        url = data.get("links", {}).get("next")

        params = None

        # Small delay between API requests
        time.sleep(1)

    return pd.json_normalize(all_listings)


if __name__ == "__main__":
    df = get_listings()
    print(f"\nTotal listings retrieved: {len(df)}")

# old script saved
# # src/data_acquisition.py

# import os
# import requests
# import pandas as pd
# from dotenv import load_dotenv


# def get_listings():
#     """Fetch vehicle listings from Auto.dev API."""

#     load_dotenv()

#     api_key = os.getenv("AUTO_DEV_API_KEY")
#     base_url = os.getenv("AUTO_DEV_BASE_URL")

#     if not api_key:
#         raise ValueError("AUTO_DEV_API_KEY not found")

#     headers = {
#         "Authorization": f"Bearer {api_key}"
#     }

#     url = f"{base_url}/listings"

#     params = {
#     # "vehicle.make": "Toyota,Honda,Nissan",
#     "vehicle.make": "Toyota",
#     "vehicle.year": "2020-2026",
#     "retailListing.price": "2000-11000",
#     "limit": 20,
#     "page": 1
#     }


#     all_listings = []

#     while url:

#         response = requests.get(
#             url,
#             headers=headers,
#             params=params if url.endswith("/listings") else None,
#             timeout=30
#         )

#         response.raise_for_status()

#         data = response.json()

#         all_listings.extend(data.get("data", []))

#         url = data.get("links", {}).get("next")

#         params = None

#     return pd.json_normalize(all_listings)


# if __name__ == "__main__":
#     df = get_listings()
#     print(f"Retrieved {len(df)} listings")
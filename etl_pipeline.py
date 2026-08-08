import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import lyricsgenius as lg
import billboard
import pandas as pd
import json
import os



scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

def main():

    # Get the Billboard Hot 100 chart data
    chart = billboard.ChartData('hot-100')

    if chart:
        data = chart.json()

    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


    # Set up the YouTube API client

    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = os.getenv("CLIENT_SECRET_PATH")  # Path to your client_secret.json file

    if not client_secrets_file:
        raise ValueError("CLIENT_SECRET_PATH environment variable is not set.")
    

    # Get credentials 
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_local_server(port=0)

    # Build the YouTube API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id="UC_x5XG1OV2P6uZZ5FSM9Ttw"
    )
    response = request.execute()

    if response:
        data = response

    with open('youtube_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


    # Set up the Genius API client 
    access_token = os.getenv("ACCESS_TOKEN")
    if not access_token:
        raise ValueError("ACCESS_TOKEN environment variable is not set.")

    genius = lg.Genius(access_token)

    search_popularity = genius.search_artist("Taylor Swift", max_songs=3, sort="popularity")

    if search_popularity:
        with open('genius_data.json', 'w', encoding='utf-8') as file:
            json.dump(search_popularity.to_dict(), file, indent=4)


   


if __name__ == "__main__":
    main()

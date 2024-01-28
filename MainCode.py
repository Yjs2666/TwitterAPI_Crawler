import string
import tokenize
import time
from urllib.error import URLError
from http.client import BadStatusLine
import json
import twitter
from nltk import *
from nltk.corpus import stopwords
from textblob import TextBlob


def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.

    CONSUMER_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    CONSUMER_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    OAUTH_TOKEN = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    OAUTH_TOKEN_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e

        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes

        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e  # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'.format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


# Storing the analysis information of each set of tweets. E.g. Hot pot in New York
class Analysis:
    def __init__(
            self,
            food_type: str,
            count: int,
            general_result: str,
            positive_rate: float,
            weakly_positive_rate: float,
            neutral_rate: float,
            weakly_negative_rate: float,
            negative_rate: float,
            curved_score: float
    ):
        self.food_type = food_type
        self.count = count
        self.general_result = general_result
        self.positive_rate = positive_rate
        self.weakly_positive_rate = weakly_positive_rate
        self.neutral_rate = neutral_rate
        self.weakly_negative_rate = weakly_negative_rate
        self.negative_rate = negative_rate
        self.curved_score = curved_score


# Lemmatize text
def lemmatize(input_text: str) -> str:
    stop = stopwords.words('english')
    tokens = tokenize.word_tokenize(input_text)
    tokens_filtered = [w for w in tokens if w.lower() not in stop and w.lower() not in string.punctuation]
    return ' '.join(tokens_filtered)


# Calculate the division rate
def calculate_rate(part, total) -> float:
    value = float(part) / float(total)
    return value


# Analyze a set of tweets e.g. all tweets for Hot Pot
def analyze_tweets(tweets: list[str], food_type: str) -> Analysis:
    # Initialize variables
    count = 0
    total_polarity = 0
    general_result = ""
    positive_count = 0
    weakly_positive_count = 0
    negative_count = 0
    weakly_negative_count = 0
    neutral_count = 0

    # Loop through all tweets in the set
    for tweet in tweets:
        count += 1
        lemma_text = lemmatize(tweet)
        tb = TextBlob(lemma_text)
        total_polarity += tb.sentiment.polarity

        # Count number of tweets of each type of sentiment
        if tb.sentiment.polarity == 0:
            neutral_count += 1
        elif 0 < tb.sentiment.polarity <= 0.4:
            weakly_positive_count += 1
        elif 0.4 < tb.sentiment.polarity <= 1:
            positive_count += 1
        elif -0.6 < tb.sentiment.polarity <= 0:
            weakly_negative_count += 1
        elif -1 < tb.sentiment.polarity <= -0.6:
            negative_count += 1

    # Calculate rates of each food type
    positive_rate = calculate_rate(positive_count, count)
    weakly_positive_rate = calculate_rate(weakly_positive_count, count)
    neutral_rate = calculate_rate(neutral_count, count)
    weakly_negative_rate = calculate_rate(weakly_negative_count, count)
    negative_rate = calculate_rate(negative_count, count)

    # Calculate overall polarity
    polarity = total_polarity / (count - neutral_count)

    # Calculate general result
    if polarity == 0:
        general_result = "Neutral"
    elif 0 < polarity <= 0.4:
        general_result = "Weakly Positive"
    elif 0.4 < polarity <= 1:
        general_result = "Positive"
    elif -0.6 < polarity <= 0:
        general_result = "Weakly Negative"
    elif -1 < polarity <= -0.6:
        general_result = "Negative"

    # Curve score
    base_score = 80
    curved_score = (polarity * 100) + base_score

    return Analysis(food_type, count, general_result, positive_rate, weakly_positive_rate, neutral_rate,
                    weakly_negative_rate, negative_rate, curved_score)


# Format a float to string in `decimal` numbers of decimals
def float_format_decimal(value: float, decimal: int) -> str:
    format_spec = '.' + str(decimal) + 'f'
    return format(value, format_spec)


def main():

    twitter_api = oauth_login()

    cities = {
        "New York": (40.712776, -74.005974),
        "Los Angeles": (34.052235, -118.24368),
        "San Francisco": (37.774929, -122.419418),
        "Chicago": (41.878113, -87.629799),
        "Houston": (29.760427, -95.369804),
        "Phoenix": (33.448376, -112.074036),
        "San Antonio": (29.424122, -98.493629),
        "Philadelphia": (39.952583, -75.165222),
        "San Diego": (32.715736, -117.161087),
        "Dallas": (32.776665, -96.796989),
        "San Jose": (37.338207, -121.886330),
        "Fort Worth": (32.755489, -97.330765),
        "Charlotte": (35.227085, -80.843124),
        "Columbus": (39.961178, -82.998795),
        "Indianapolis": (39.768402, -86.158066),
        "Seattle": (47.606209, -122.332069),
        "Denver": (39.739235, -104.990250),
        "Washington": (38.907192, -77.036873),
        "Boston": (42.360081, -71.058884),
        "Oklahoma City": (35.467560, -97.516426),
        "Las Vegas": (36.169941, -115.139832),
        "Portland": (45.512230, -122.658722),
        "Detroit": (42.331429, -83.045753),
        "Memphis": (35.149532, -90.048981),
        "Milwaukee": (43.038902, -87.906471),
        "Baltimore": (39.290386, -76.612190),
        "Atlanta": (33.748997, -84.387985),
        "Sacramento": (38.581573, -121.494400),
        "Miami": (25.761681, -80.191788)
    }

    food_types = [
        "Dumpling",
        "Hot pot",
        "Sushi",
        "Sashimi",
        "Ramen",
        "Pho",
        "Kimchi",
        "Green curry",
        "Pad thai",
        "Butter chicken"
    ]

    # Initialize a dict to store all tweets of each food type
    results_tweets = {}
    for food_type in food_types:
        results_tweets[food_type] = []

    tweets_count = 0

    # Fetch tweets from Api and append them to `results_tweets`
    for food_type in food_types:
        for cityName, cityLatLong in cities.items():
            geocode = str(cityLatLong[0]) + ',' + str(cityLatLong[1]) + ',' + '50mi'
            response = make_twitter_request(twitter_api.search.tweets, q=food_type, geocode=geocode, count=100)
            for i in response['statuses']:
                results_tweets[food_type].append(i['text'])
                tweets_count += 1

    # Output all tweets to all_tweets.txt
    with open('all_tweets.txt', 'w') as output_file:
        output_file.write(json.dumps(results_tweets, indent=4))

    # Initialize a dict to store all tweets analysis of each food type
    results_analysis = {}
    for food_type in food_types:
        results_analysis[food_type] = None

    # Analyze all tweets
    results_analysis = {}
    for food_type, tweets in results_tweets.items():
        results_analysis[food_type] = analyze_tweets(tweets, food_type)

    # Create a dictionary to sort all food types based on curved_score
    result_analysis_of_curved_scores = {}
    for food_type, analysis in results_analysis.items():
        result_analysis_of_curved_scores[food_type] = analysis.curved_score

    # Rank all food types based on their curved_score
    rankings = sorted(result_analysis_of_curved_scores, key=result_analysis_of_curved_scores.get, reverse=True)
    rankings_string = " > ".join(rankings)

    # Output report
    with open('report.txt', 'w') as output_file:
        output_file.write("Report\n")
        output_file.write("------------\n")
        output_file.write("\n")
        for food_type, analysis in results_analysis.items():
            output_file.write("Fetched " + str(analysis.count) + " tweets on " + analysis.food_type + "\n")
            output_file.write("General Result: " + analysis.general_result + "\n")
            output_file.write("Positive Rate: " + float_format_decimal(analysis.positive_rate * 100, 2) + "%\n")
            output_file.write("Weakly Positive Rate: " + float_format_decimal(analysis.weakly_positive_rate * 100, 2) + "%\n")
            output_file.write("Neutral Rate: " + float_format_decimal(analysis.neutral_rate * 100, 2) + "%\n")
            output_file.write("Weakly Negative Rate: " + float_format_decimal(analysis.weakly_negative_rate * 100, 2) + "%\n")
            output_file.write("Negative Rate: " + float_format_decimal(analysis.negative_rate * 100, 2) + "%\n")
            output_file.write("-------------------------------------\n")
        output_file.write("\n")
        output_file.write("\n")
        output_file.write("Overall Result\n")
        output_file.write("Total Tweets: " + str(tweets_count) + "\n")
        output_file.write("-------------------\n")
        output_file.write("Scores:" + "\n")
        for food_type, analysis in results_analysis.items():
            output_file.write(analysis.food_type + ": " + float_format_decimal(analysis.curved_score, 2) + "\n")
        output_file.write("\n")
        output_file.write("Rankings: " + rankings_string + "\n")
        output_file.close()


if __name__ == '__main__':
    main()

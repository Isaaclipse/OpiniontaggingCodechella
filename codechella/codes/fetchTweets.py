import tweepy
import json

URL = "https://api.twitter.com/2/tweets?ids=1278747501642657792"
CONSUMER_API_KEY = "SYs3QoOHodtFpkA1SCpI5NIeI"
CONSUMER_API_KEY_SECRET = "mY6dkfPP2m2YaNN5SXBoH4klIAKLQafjC1wmmzK6Lp0hbBvcMj"
ACCESS_TOKEN = "887894057615323136-2kp00uTfxCJx9e23WpeEQA7URfjqzLQ"
ACCESS_TOKEN_SECRET = "oCWBcclkv8YcAdC5zRluyGVGDP5Nrv0W7dJ2MTdA0DmPb"

auth = tweepy.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

def pull_tweets(userhandle, num_tweet):
    api = tweepy.API(auth, wait_on_rate_limit=True)
    data = tweepy.Cursor(api.user_timeline, id=userhandle,\
     tweet_mode='extended').items(num_tweet)
    data = processData(data)
    return data

def processData(inputData):
    cnt = 0
    allTweets = []
    for eachTweet in inputData:
        dummyJson = {}
        dummyJson['userid'] = eachTweet.user.name
        dummyJson['screen_name'] = eachTweet.user.screen_name

        if 'retweeted_status' in eachTweet._json:
            dummyJson['text'] = eachTweet._json['retweeted_status']['full_text']
        else:
            dummyJson['text'] = eachTweet.full_text
        dummyJson['hashtag'] = eachTweet.entities['hashtags']
        allTweets.append(dummyJson)
    return allTweets

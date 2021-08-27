import re  # regex library
import nltk  # sentiment analysis library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.downloader.download('vader_lexicon')


# python function for cleaning the raw tweet
# INPUT: raw tweet string
# OUTPUT: tweet string without any special characters
def CleanTweetContent(tweet):
    # Regex
    cleanTweet = re.sub('[^A-Za-z0-9]+', ' ', tweet).lower()

    return cleanTweet


# usage of the 'CleanTweetContent' function
# tweetCleaned = CleanTweetContent("hello, my name is Josip")

# class encapsulating results of the analysis
class AnalysisResult:
    def __init__(self, content, polarity, subjectivity, sentiment, neg, neu, pos, comp):
        self.content = content
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.sentiment = sentiment
        self.neg = neg
        self.neu = neu
        self.pos = pos
        self.comp = comp


# python function for analysis of one clean tweet
# INPUT: clean tweet string
# OUTPUT: class containing all important data about the particular tweet
def Analyze(cleanTweet):
    polaritySubjectivity = TextBlob(cleanTweet).sentiment
    polarity = polaritySubjectivity[0]
    subjectivity = polaritySubjectivity[1]

    score = SentimentIntensityAnalyzer().polarity_scores(cleanTweet)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']

    if neg > pos:
        sentiment = "negative"
    elif pos > neg:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return AnalysisResult(cleanTweet, polarity, subjectivity, sentiment, neg, neu, pos, comp)

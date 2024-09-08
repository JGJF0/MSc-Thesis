import tldextract
import ipaddress
import editdistance
import pandas
import re
import textstat
import spacy
from datetime import datetime, timedelta


# Load the large english NLP model from spaCy
nlp = spacy.load('en_core_web_lg')


def add_sender_category(row):
    # Get sender id
    sender_id = str(row['Sender']) if pandas.notna(row['Sender']) else ''

    # Stop if no sender ID
    if sender_id == '':
        return 'Unknown'

    # Regex patterns
    shortcode_pattern = re.compile(r'^\d{3,8}$')
    email_pattern = re.compile(r'^.+@.+$')
    phone_pattern = re.compile(r'^(\+?\d{1,3}[\s-]?)?(\(\d{1,4}\)[\s-]?)?(\d[\s-]?){2,14}\d$')

    # Classify sender ID
    if shortcode_pattern.match(sender_id):
        return 'Short code'
    elif email_pattern.match(sender_id):
        return 'Email'
    elif phone_pattern.match(sender_id):
        return 'Phone number'
    else:
        return 'Unrecognized'


def add_url_category(row):
    # Get URL and brand
    url = str(row['URL']).lower() if pandas.notna(row['URL']) else ''
    brand_name = str(row['Brand']).lower() if pandas.notna(row['Brand']) else ''
    url_shorteners = {'bit.ly', 'tinyurl.com', 'tiny.cc', 'cutt.ly', 't.ly', 'goo.gl', 't.co', 'rebrand.ly', 'b.link',
                      'page.link', 'tinyurl.llc', 'is.gd', 'ffm.to', 's.id', 'app.link', 'rb.gy', 'ow.ly', 'kaywa.me',
                      'qrco.de', 'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'shorte.st', 'shorturl.at'}

    # Stop if no url
    if url == '':
        return ''

    # Prepare url and brand name
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    sld = extracted.domain
    tld = extracted.suffix
    full_domain = f"{sld}.{tld}" if sld and tld else sld
    brand_name = brand_name.replace(' ', '')

    # Check for IP address
    try:
        ipaddress.ip_address(sld)
        return 'IP address'
    except ValueError:
        pass

    # Classify URL
    if full_domain in url_shorteners:
        return 'URL shortener'
    elif sld == brand_name:
        return 'Deceptive top-level domain'
    elif (brand_name != '') and ((brand_name in sld) or (editdistance.eval(sld, brand_name) == 1)):
        return 'Deceptive second-level domain'
    elif (((brand_name != '') and (subdomain != '')) and
          ((brand_name in subdomain) or detect_brand_typo_in_subdomain(brand_name, subdomain))):
        return 'Deceptive subdomain'
    else:
        return 'Random domain'


def detect_brand_typo_in_subdomain(brand, subdomain):
    brand_length = len(brand)
    # Iterate through possible substrings of length of brand in the subdomain
    for i in range(len(subdomain) - brand_length + 1):
        # Extract the substring
        substring = subdomain[i:i + brand_length]
        # Calculate the difference
        distance = editdistance.eval(substring, brand)
        # At most 1 typo
        if distance <= 1:
            return True
    return False


def add_date_of_msg_sent(row):
    # Get text and submission date
    text = row['ImageText']
    submission_date_str = row['TimeSubmitted']
    # Parse submission date into datetime
    submission_date = datetime.strptime(submission_date_str, '%d/%m/%Y %H:%M')

    # Check if text is a string and not NaN
    if pandas.isna(text) or not isinstance(text, str):
        return ''

    # Regex patterns for different date formats
    date_patterns = [
        # ddd, mmm DD, HH:MM - no year
        (re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2},'
                    r' \d{2}:\d{2}\b'), '%a, %b %d, %H:%M', False),
        # day, month DD, YYYY - with year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),'
                    r' (January|February|March|April|May|June|July|August|September|October|November|December)'
                    r' \d{1,2}, \d{4}\b'), '%A, %B %d, %Y', True),
        # day, DD month, YYYY - with year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),'
                    r' \d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December),'
                    r' \d{4}\b'), '%A, %d %B, %Y', True),
        # ddd, mmm DD, HH:MM AM/PM - no year
        (re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2},'
                    r' \d{1,2}:\d{2} (AM|PM)\b'), '%a, %b %d, %I:%M %p', False),
        # day, mmm DD HH:MM AM/PM - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),'
                    r' (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2} \d{1,2}:\d{2}'
                    r' (AM|PM)\b'), '%A, %b %d %I:%M %p', False),
        # day HH:MM AM/PM - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d{1,2}:\d{2}'
                    r' (AM|PM)\b'), '%A %I:%M %p', False),
        # Today HH:MM AM/PM - no year
        (re.compile(r'\bToday \d{1,2}:\d{2} (AM|PM)\b'), 'Today %I:%M %p', False),
        # Yesterday HH:MM AM/PM - no year
        (re.compile(r'\bYesterday \d{1,2}:\d{2} (AM|PM)\b'), 'Yesterday %I:%M %p', False),
        # day HH:MM am/pm - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d{1,2}:\d{2}'
                    r' (am|pm)\b'), '%A %I:%M %p', False),
        # Today HH:MM am/pm - no year
        (re.compile(r'\bToday \d{1,2}:\d{2} (am|pm)\b'), 'Today %I:%M %p', False),
        # Yesterday HH:MM am/pm - no year
        (re.compile(r'\bYesterday \d{1,2}:\d{2} (am|pm)\b'), 'Yesterday %I:%M %p', False),
        # ddd, mmm DD - no year
        (re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\b'),
         '%a, %b %d', False),
        # mmm DD, YYYY - with year
        (re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b'), '%b %d, %Y', True),
        # day, mmm DD - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), '
                    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\b'), '%A, %b %d', False),
        # day * HH:MM - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \* \d{2}:\d{2}\b'), '%A * %H:%M',
         False),
        # HH:MM AM/PM, mmm DD - no year
        (re.compile(r'\b\d{1,2}:\d{2} (AM|PM), (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\b'),
         '%I:%M %p, %b %d', False),
        # day, month DD - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),'
                    r' (January|February|March|April|May|June|July|August|September|October|November|December)'
                    r' \d{1,2}\b'), '%A, %B %d', False),
        # day, DD month - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), \d{1,2}'
                    r' (January|February|March|April|May|June|July|August|September|October|November|December)\b'),
         '%A, %d %B', False),
        # day, DD mmm - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), \d{1,2} '
                    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'), '%A, %d %b', False),
        # ddd DD mmm - no year
        (re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'),
         '%a %d %b', False),
        # ddd HH:MM AM/PM - no year
        (re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{1,2}:\d{2} (AM|PM)\b'), '%a %I:%M %p', False),
        # DD mmm YYYY at HH:MM - with year
        (re.compile(r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} at \d{2}:\d{2}\b'),
         '%d %b %Y at %H:%M', True),
        # DD month YYYY at HH:MM - with year
        (re.compile(
            r'\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4} at \d{2}:\d{2}\b'),
         '%d %B %Y at %H:%M', True),
        # X days ago - no year
        (re.compile(r'\b(\d+) days? ago\b'), 'days_ago', False),
        # X hours ago - no year
        (re.compile(r'\b(\d+) hours? ago\b'), 'hours_ago', False),
        # mmm DD, HH:MM AM/PM - no year
        (re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{1,2}:\d{2} (AM|PM)\b'), '%b %d, %I:%M %p', False),

        # HH:MM, mmm DD - no year
        (re.compile(r'\b\d{1,2}:\d{2}, (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\b'), '%H:%M, %b %d',
         False),
        # ddd, DD mmm at HH:MM - no year
        (re.compile(
            r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) at \d{1,2}:\d{2}\b'),
         '%a, %d %b at %H:%M', False)
    ]

    last_resort_date_patterns = [
        # Today - no year
        (re.compile(r'\bToday\b'), 'Today', False),
        # Yesterday - no year
        (re.compile(r'\bYesterday\b'), 'Yesterday', False),
        # day - no year
        (re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'), '%A', False),
        # HH:MM AM/PM - no year
        (re.compile(r'\b\d{1,2}:\d{2} (AM|PM)\b'), '%I:%M %p', False)
    ]

    # Variables for finding the first pattern in text
    first_match = None
    first_match_position = float('inf')

    # Find the first match for any normal pattern in text
    for pattern, date_format, contains_year in date_patterns:
        match = pattern.search(text)
        if match:
            if match.start() < first_match_position:
                first_match = (match.group(0), date_format, contains_year)
                first_match_position = match.start()

    # If no match was found, look for some identified partial date patterns as a last resort
    if first_match is None:
        for pattern, date_format, contains_year in last_resort_date_patterns:
            match = pattern.search(text)
            if match:
                if match.start() < first_match_position:
                    first_match = (match.group(0), date_format, contains_year)
                    first_match_position = match.start()

    if first_match:
        date_str, date_format, date_contains_year = first_match
        # Handle case - today dates and times
        if ('Today' in date_str) or (date_format == 'hours_ago') or (date_format == '%I:%M %p'):
            return submission_date.strftime('%d/%m/%Y')
        # Handle case - yesterday dates
        elif 'Yesterday' in date_str:
            return (submission_date - timedelta(days=1)).strftime('%d/%m/%Y')
        # If date contains year, return date of sending msg
        elif date_contains_year:
            parsed_date = datetime.strptime(date_str, date_format)
            final_date = parsed_date.strftime('%d/%m/%Y')
            return final_date
        # Find closest date to submission date to get date of sending msg
        else:
            final_date = find_closest_date_without_year(date_str, date_format, submission_date)
            return final_date
    return ''


def find_closest_date_without_year(date_str, date_format, submission_date):
    # Special case with 'X days ago' - calculate the date of the last day before submission date
    if date_format == 'days_ago':
        return find_closest_x_days_ago(date_str, submission_date)

    # Special case with only day - calculate the date of the last day before submission date
    if (date_format == '%A %I:%M %p' or date_format == '%A * %H:%M' or date_format == '%A' or
            date_format == '%a %I:%M %p'):
        full_day = True
        if date_format == '%a %I:%M %p':
            full_day = False
        return find_closest_day_to_submission_date(date_str, submission_date, full_day)

    # Calculate closest possible date either with the submission year or the year before
    for year in range(submission_date.year, submission_date.year - 2, -1):
        try:
            # Parse the date with the current year from the loop
            parsed_date = datetime.strptime(f"{date_str}, {year}", date_format + ", %Y")
            # Check that parsed date (date of msg sent) isn't after the submission date
            if parsed_date <= submission_date:
                closest_date = parsed_date.strftime('%d/%m/%Y')
                return closest_date
        except ValueError:
            continue    # Skip invalid dates
    return ''


def find_closest_day_to_submission_date(date_str, submission_date, full_day):
    # Mapping of weekday names to their respective indexes
    weekdays_full = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    weekdays_short = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

    # Get the day of the week from date string
    weekday_str = date_str.split()[0]

    # Get day index
    if full_day:
        weekday = weekdays_full[weekday_str]
    else:
        weekday = weekdays_short[weekday_str]

    # Calculate when was the last weekday
    days_ago = (submission_date.weekday() - weekday) % 7
    if days_ago == 0:
        days_ago = 7
    closest_date = submission_date - timedelta(days=days_ago)
    return closest_date.strftime('%d/%m/%Y')


def find_closest_x_days_ago(date_str, submission_date):
    days_ago = int(date_str.split()[0])

    # Calculate when was the last weekday
    closest_date = submission_date - timedelta(days=days_ago)
    return closest_date.strftime('%d/%m/%Y')


def add_lure_principle(row, threshold_length):
    # Get message text
    msg = str(row['MessageText']) if pandas.notna(row['MessageText']) else ''

    # Stop if no message
    if msg == '':
        return ''

    # Process message using pretrained model
    doc = nlp(msg)

    # Preprocess text - convert to lowercase (done), tokenization, lemmatization
    tokens, lemmas, ents = preprocess_txt(doc)

    principles = []

    # Identify lure principles
    # time principle
    time = identify_time(lemmas, ents, doc)
    if time:
        principles.append('Time')

    # social compliance principle
    social_compl = identify_social_compliance(ents)
    if social_compl:
        principles.append('Social Compliance')

    # herd principle
    herd = identify_herd(tokens, lemmas, ents, doc)
    if herd:
        principles.append('Herd')

    # kindness principle
    kindness = identify_kindness(lemmas)
    if kindness:
        principles.append('Kindness')

    # need and greed principle
    need_greed = identify_need_and_greed(lemmas, ents, doc)
    if need_greed:
        principles.append('Need and Greed')

    # distraction principle
    distraction = identify_distraction(row, threshold_length)
    if distraction:
        principles.append('Distraction')

    return ', '.join(principles)


def preprocess_txt(doc):
    # Tokenize and lemmatize text
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]

    # Identify entities using spaCy NER
    entities = doc.ents

    return tokens, lemmas, entities


def generate_ngram(tokens, n):
    # Generate n-grams
    ngram = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngram


def identify_time(lemmas, entities, doc):
    # Identify time-related entities
    time_entities = [ent for ent in entities if ent.label_ in ['DATE', 'TIME', 'DURATION']]
    # If time-related entities exist, check if they create valid n-grams indicating deadline
    if time_entities:
        for token in doc:
            # Check if token can create valid deadline n-grams using deadline keywords
            if token.text in ['within', 'until', 'by', 'in', 'till', 'before']:
                # If valid token is found, check for close time-related entities
                for ent in time_entities:
                    # Check if token with time-related entity create valid 2-gram or 3-gram
                    if token.i < ent.start and ent.start - token.i < 3:
                        return True

    # Urgency keywords to create n-grams indicating call to action
    urgency_keywords = ['immediately', 'promptly', 'now']
    for token in doc:
        # Check if token is a verb in base form (call to action form)
        if token.pos_ == 'VERB' and token.tag_ == 'VB':
            # Check if urgency keyword is close, creating valid 2-grams or 3-grams
            for i in range(token.i + 1, min(token.i + 3, len(doc))):
                if doc[i].lemma_ in urgency_keywords:
                    return True

    # Check for keywords or specific bigrams
    bigrams = generate_ngram(lemmas, 2)
    detected = (any(lemma in ['expire', 'deadline', 'valid', 'immediately', 'immediate', 'urgent', 'hurry', 'now'] for
                    lemma in lemmas)) or (('not', 'later') in bigrams) or (('last', 'day') in bigrams)
    return detected


def identify_social_compliance(entities):
    # Identify organization-related entities
    org_entities = [ent for ent in entities if ent.label_ == 'ORG']

    if org_entities:
        return True
    return False


def identify_herd(tokens, lemmas, entities, doc):
    # Define a list of collectives, and singular/plural people-related nouns
    collective_keywords = ['group', 'community', 'team', 'collective']
    singular_people = ['man', 'woman', 'person', 'taxpayer', 'participant', 'friend', 'user', 'member', 'customer']
    plural_people = ['men', 'women', 'people', 'taxpayers', 'participants', 'friends', 'users', 'members', 'customers',
                     'everyone']

    # Find collectives or plural nouns
    plural_detected = [token for token in tokens if token in plural_people]
    collective_detected = [lemma for lemma in lemmas if lemma in collective_keywords]
    if plural_detected or collective_detected:
        return True

    # Identify entities denoting numerical position, implying collective
    num_entities = [ent for ent in entities if ent.label_ in ['ORDINAL', 'CARDINAL']]
    # Words that imply collective
    multiplier_keywords = ['every', 'average', 'next', 'typical', 'many', 'several']

    for token in doc:
        # Check if token is a singular noun related to people
        if token.text in singular_people:
            # If entities denoting position exist, check if they create valid n-grams indicating collective
            if num_entities:
                for ent in num_entities:
                    # Check if token with entity precedes noun related to people to create valid 2-gram
                    if token.i > ent.end and token.i - 1 == ent.end:
                        return True
            # Check if multiplier precedes noun related to people to create valid 2-gram
            if doc[token.i-1].lemma_ in multiplier_keywords:
                return True
    return False


def identify_kindness(lemmas):
    # Define list of keywords and phrases indicating kindness
    keywords = ['help', 'donate', 'donation', 'helpful', 'children', 'kid', 'teen', 'sick', 'ill', 'support', 'care',
                'charity', 'kind', 'kindness', 'aid', 'contribute', 'contribution', 'save']
    key_bigrams = [('do', 'better'), ('in', 'memory'), ('in', 'need')]
    key_trigrams = [('make', 'a', 'difference'), ('make', 'a', 'change')]

    # Generate n-grams
    bigrams = generate_ngram(lemmas, 2)
    trigrams = generate_ngram(lemmas, 3)

    # Identify presence of keywords or 2-/3-grams, indicating kindness
    for lemma in lemmas:
        if lemma in keywords:
            return True

    for bigram in bigrams:
        if bigram in key_bigrams:
            return True

    for trigram in trigrams:
        if trigram in key_trigrams:
            return True
    return False


def identify_need_and_greed(lemmas, entities, doc):
    # Define keywords and phrases indicating need and greed
    keywords = ['profit', 'gift', 'cashback', 'settlement', 'salary', 'compensation', 'draw', 'free', 'savings',
                'refund', 'freebies', 'prize', 'reimburse', 'reimbursement', 'congratulation', 'win', 'winner',
                'winning', 'income']
    key_bigrams = [('owe', 'you'), ('offer', 'you'), ('daily', 'pay'), ('we', 'pay'), ('no', 'cost'), ('hand', 'out')]
    key_trigrams = [('you', 'can', 'earn'), ('you', 'can', 'make'), ('you', 'can', 'get'), ('you', 'can', 'win'),
                    ('you', 'can', 'obtain'), ('you', 'can', 'receive')]

    # Generate n-grams
    bigrams = generate_ngram(lemmas, 2)
    trigrams = generate_ngram(lemmas, 3)

    # Identify presence of keywords or 2-/3-grams, indicating need and greed
    for lemma in lemmas:
        if lemma in keywords:
            return True

    for bigram in bigrams:
        if bigram in key_bigrams:
            return True

    for trigram in trigrams:
        if trigram in key_trigrams:
            return True

    # Identify money-related entities
    time_entities = [ent for ent in entities if ent.label_ == 'MONEY']
    # If money-related entities exist, check for valid n-grams
    if time_entities:
        for token in doc:
            # Check if token can create valid n-grams indicating monetary gain
            if token.lemma_ in ['earn', 'make', 'get', 'obtain', 'receive']:
                # If valid token is found, check for close money-related entities
                for ent in time_entities:
                    # Check if token with time-related entity create valid 2--, 3-, or 4-gram
                    if token.i < ent.start and ent.start - token.i < 4:
                        return True
    return False


def identify_distraction(row, threshold):
    # Get message length and see if it is longer than threshold
    msg_length = row['MsgLength']

    if msg_length >= threshold:
        return True
    return False


def add_msg_length(row):
    # Get message
    message = str(row['MessageText']) if pandas.notna(row['MessageText']) else ''

    # Stop if no message
    if message == '':
        return 0

    # Calculate message length in words
    return textstat.lexicon_count(message)


if __name__ == "__main__":
    # Dedicated file names
    filename = 'smish_data.csv'
    output_filename = 'analysis_data.csv'

    # Load csv file into dataframe
    dataframe = pandas.read_csv(filename)

    # Apply relevant function to add relevant data
    dataframe['SenderType'] = dataframe.apply(add_sender_category, axis=1)
    dataframe['URLCategory'] = dataframe.apply(add_url_category, axis=1)
    dataframe['DateMsgSent'] = dataframe.apply(add_date_of_msg_sent, axis=1)
    dataframe['MsgLength'] = dataframe.apply(add_msg_length, axis=1)

    # Calculate the median length of messages in words, and the threshold for distraction principle
    median_length = dataframe['MsgLength'].median()
    threshold_length = median_length * 1.5

    # Apply add_lure_principle function to add lure principles
    dataframe['LurePrinciple'] = dataframe.apply(add_lure_principle, args=(threshold_length,), axis=1)

    # Save updated dataframe
    dataframe.to_csv(output_filename, index=False)

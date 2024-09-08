import asyncio
import tldextract
import csv
import requests
import vt
import pytesseract
import aiohttp
from pyppeteer import launch
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError
from io import BytesIO


# Path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'<PATH_TO_tesseract.exe>'


# Get page and process its content
async def fetch_page(page, url, entry_id, vt_client):
    # Go to the URL and wait until idle
    await page.goto(url, {'waitUntil': 'networkidle0', 'timeout': 60000})
    # Wait for the content
    await page.waitForSelector('.MuiPaper-root.jss8.MuiPaper-elevation1.MuiPaper-rounded', {'timeout': 60000})

    # Function to get content from tabs
    async def get_tab_content(button_selector):
        # Click the tab button
        await page.click(button_selector)
        # Wait for content
        await page.waitForSelector('.MuiPaper-root.jss8.MuiPaper-elevation1.MuiPaper-rounded', {'timeout': 60000})
        # Get content
        return await page.content()

    # Selector for each tab button
    summary_selector = 'button[role="tab"]:nth-of-type(1)'
    virus_selector = 'button[role="tab"]:nth-of-type(2)'
    whois_selector = 'button[role="tab"]:nth-of-type(3)'
    url_selector = 'button[role="tab"]:nth-of-type(4)'

    # Get content of each tab
    mainpage_content = await page.content()
    summary_content = await get_tab_content(summary_selector)
    virus_content = await get_tab_content(virus_selector)
    whois_content = await get_tab_content(whois_selector)
    url_content = await get_tab_content(url_selector)

    # Parse the contents
    mainpage_soup = BeautifulSoup(mainpage_content, 'html.parser')
    summary_soup = BeautifulSoup(summary_content, 'html.parser')
    virus_soup = BeautifulSoup(virus_content, 'html.parser')
    whois_soup = BeautifulSoup(whois_content, 'html.parser')
    url_soup = BeautifulSoup(url_content, 'html.parser')

    # Extract all data
    msg_text = extract_message_data(mainpage_soup)
    img_text = await extract_text_from_img(mainpage_soup)
    summary_data = extract_summary_data(summary_soup)
    virus_data = extract_virus_data(virus_soup)
    whois_data = extract_whois_url_data(whois_soup)
    url_data = extract_whois_url_data(url_soup)

    # Extract data of interest
    await extract_data(entry_id, msg_text, img_text, summary_data, virus_data, whois_data, url_data, page, vt_client)


def extract_message_data(page_soup):
    # Find the messageContentField
    msg_field = page_soup.find('span', class_='messageContentField')
    if msg_field:
        return msg_field.get_text(strip=True)
    return ''


async def extract_text_from_img(page_soup):
    async def get_image(image_url):
        # Create asynchronous HTTP session
        async with aiohttp.ClientSession() as session:
            # GET request to image url
            async with session.get(image_url) as response:
                # Return image as raw bytes
                return await response.read()

    # Get image url from img tag
    img_tag = page_soup.find('img', {'class': 'messagePicture'})
    if img_tag and img_tag['src']:
        img_url = img_tag['src']

        # Get image data
        img_data = await get_image(img_url)
        if img_data:
            try:
                img = Image.open(BytesIO(img_data))
                # Extract text from the image
                extracted_text = pytesseract.image_to_string(img)
                return extracted_text
            except UnidentifiedImageError as err:
                return ''
            except TypeError as err:
                return ''
    return ''


def extract_summary_data(page_soup):
    summary_data = {}
    # Find divs with class 'msgSection'
    data_sections = page_soup.find_all('div', class_='msgSection')
    # Extract data from each section
    for section in data_sections:
        label = section.find('label')
        if label:
            label_text = label.text.replace(':', '').strip()

            if label_text == 'Is this a SMS Phishing Attack?':
                continue

            # Find the next sibling of parent of label that contains the data
            data_label = section.find_next_sibling('span', class_='URLlinkdata')
            if data_label:
                summary_data[label_text] = data_label.get_text(strip=True)
            else:
                summary_data[label_text] = ''
    return summary_data


def extract_virus_data(page_soup):
    virus_data = {}
    # Find spans with class 'msgSectionTitle'
    labels = page_soup.find_all('span', class_='msgSectionTitle')
    for label in labels:
        label_text = label.text.replace(':', '').strip()

        if label_text == 'Message Info':
            continue

        # Find the next sibling of parent element that contains the data
        data_label = label.parent.find_next_sibling()
        if data_label:
            virus_data[label_text] = data_label.get_text(strip=True)
        else:
            virus_data[label_text] = ''
    return virus_data


def extract_whois_url_data(page_soup):
    data = {}
    # Find divs with class 'whoisResults'
    labels = page_soup.find_all('div', class_='whoisResults')
    for label in labels:
        label_text = label.get_text(strip=True)
        data_label = label.find_next_sibling('div')
        if data_label:
            data[label_text] = data_label.get_text(strip=True)
        else:
            data[label_text] = ''
    return data


async def extract_data(msg_id, msg_text, img_text, summary_data, virus_data, whois_data, url_data, page, vt_client):
    # Gather additional data
    # Virustotal data about final url
    detected, malicious, malware, phishing, suspicious, final_url = await get_virustotal_info(
        process_url(summary_data.get('URL', '')), vt_client)
    # Domain data about final url - if data cannot be found, use scraped data
    domain_status = await get_domain_status(final_url, page)
    registrar_name, domain_created, domain_last_update = get_domain_info(final_url)
    if registrar_name == '':
        registrar_name = whois_data.get('Registrar', '')
    if domain_created == '':
        domain_created = process_date_at_submission(whois_data.get('Created', ''))
    if domain_last_update == '':
        domain_last_update = process_date_at_submission(whois_data.get('Last Update', ''))

    # Create data entry
    data = {'MessageID': msg_id,
            'Sender': summary_data.get('Sender', ''),
            'TimeSubmitted': process_time_submitted(summary_data['Time Submitted']),
            'MessageText': process_msg_text(msg_text),
            'URL': process_url(summary_data.get('URL', '')),
            'Subdomain': process_subdomain(process_url(summary_data.get('URL', ''))),
            'SLD': process_sld(process_url(summary_data.get('URL', ''))),
            'TLD': process_tld(process_url(summary_data.get('URL', ''))),
            'RedirectedURL': '' if final_url == process_url(summary_data.get('URL', '')) else final_url,
            'Brand': summary_data.get('Brand', ''),
            'DomainRegistrar': registrar_name,
            'DomainCreationDate': domain_created,
            'DomainLastUpdateAtSubmission': process_date_at_submission(whois_data.get('Last Update', '')),
            'DomainLastUpdate': domain_last_update,
            'DomainStatus': domain_status,
            'DomainActiveFor': get_domain_activity_span(domain_created, domain_last_update, domain_status),
            'Detected': detected,
            'Malicious': malicious,
            'Malware': malware,
            'Phishing': phishing,
            'Suspicious': suspicious,
            'ImageText': img_text
            }

    write_into_csv(data)


def process_time_submitted(time_submitted):
    # Parse the input time and date
    input_format = '%m/%d/%Y, %I:%M:%S %p'
    datetime_obj = datetime.strptime(time_submitted, input_format)

    # Round minutes
    if datetime_obj.second >= 30:
        datetime_obj += timedelta(minutes=1)  # Add 1min

    # Convert to "DD/MM/YYYY HH:MM"
    output_format = '%d/%m/%Y %H:%M'
    formatted_time = datetime_obj.strftime(output_format)
    return formatted_time


def process_msg_text(msg):
    msg = msg.lower()
    return msg


def process_subdomain(url):
    extracted = tldextract.extract(url)
    return extracted.subdomain


def process_sld(url):
    extracted = tldextract.extract(url)
    return extracted.domain


def process_tld(url):
    extracted = tldextract.extract(url)
    return extracted.suffix


def process_date_at_submission(date):
    if date == '':
        return ''
    try:
        # Parse the input time and date
        datetime_obj = datetime.strptime(date,  '%Y-%m-%d %H:%M:%S')
    except ValueError:
        # If format doesn't match, return original format
        return date

    # Convert to "DD/MM/YYYY"
    formatted_date = datetime_obj.strftime('%d/%m/%Y')
    return formatted_date


def process_url(url):
    processed_url = url.replace(' ', '')
    return processed_url


def get_domain_info(url):
    if url != '':
        # Extract domain and create URL to look up
        sld = process_sld(url)
        tld = process_tld(url)
        whois_url_query = f"https://who.is/whois/{sld}.{tld}"

        # Send HTTP GET request
        response = requests.get(whois_url_query)
        # Parse content
        pagesoup = BeautifulSoup(response.text, 'html.parser')

        # Find registrar name, registered and last updated data
        registrar = ''
        registered_date = ''
        last_updated_date = ''
        data_sections = pagesoup.find_all('div', class_='queryResponseBodyRow')
        for section in data_sections:
            key = section.find('div', class_='queryResponseBodyKey')
            if key:
                key_text = key.get_text(strip=True)
                if 'Name' in key_text:
                    registrar = section.find('div', class_='queryResponseBodyValue').get_text(strip=True)
                if 'Registered On' in key_text:
                    registered_value = section.find('div', class_='queryResponseBodyValue').get_text(strip=True)
                    registered_date = process_date(registered_value)
                if 'Updated On' in key_text:
                    update_value = section.find('div', class_='queryResponseBodyValue').get_text(strip=True)
                    last_updated_date = process_date(update_value)
        return registrar, registered_date, last_updated_date
    return '', '', ''


async def get_domain_status(url, page):
    if url != '':
        # Extract domain and create URL to look up
        sld = process_sld(url)
        tld = process_tld(url)
        whois_url_query = f"https://who.is/whois/{sld}.{tld}"

        # Go to the URL
        await page.goto(whois_url_query)
        # Wait for JavaScript to execute
        await page.waitForFunction(
            "() => document.querySelector('#siteStatusStatus') && document.querySelector("
            "'#siteStatusStatus').innerText.trim() !== ''",
            {'timeout': 60000}
        )
        # Extract the status using JavaScript evaluation
        status = await page.evaluate("() => document.querySelector('#siteStatusStatus').innerText")
        return status
    return ''


def process_date(date):
    if date == '':
        return ''

    try:
        # Parse the input date
        datetime_obj = datetime.strptime(date, '%Y-%m-%d')

        # Convert to "DD/MM/YYYY"
        formatted_date = datetime_obj.strftime('%d/%m/%Y')
        return formatted_date
    except ValueError as err:
        return ''


def get_domain_activity_span(create_date, last_update_date, status):
    date_format = '%d/%m/%Y'
    # To calculate activity span, (create_date + active status) or (create_date + last_update_date) is needed
    # If domain active, calculate till present
    if create_date != '' and status == 'Active':
        try:
            # Parse the input date
            date_created = datetime.strptime(create_date, date_format)
        except ValueError:
            return ''

        # Get today's date in DD/MM/YYYY and parse it
        date_today = datetime.now()
        date_today = datetime.strptime(date_today.strftime(date_format), date_format)

        # Calculate the number of days of activity till present
        activity = date_today - date_created
        return activity.days + 1    # +1 to account for creation day

    # If domain inactive, use last_update_date
    if create_date != '' and last_update_date != '':
        try:
            # Parse the input dates
            date_created = datetime.strptime(create_date, date_format)
            date_last_update = datetime.strptime(last_update_date, date_format)
        except ValueError:
            return ''   # If formats don't match, return ''

        # Calculate the number of days of activity till last update
        activity = date_last_update - date_created
        return activity.days + 1    # +1 to account for creation day
    return ''


async def get_virustotal_info(url, vt_client):
    if url == '':
        return '', '', '', '', '', ''

    try:
        # Scan the URL
        scan = await vt_client.scan_url_async(url)
        # Get the URL identifier
        url_id = vt.url_id(url)

        # Get the analysis results
        analysis = await vt_client.get_object_async(f"/urls/{url_id}")
        results = analysis.last_analysis_results
        final_url = analysis.last_final_url

        # If redirection occurs, scan the final url
        if final_url != url:
            return await get_virustotal_info(final_url, vt_client)

        # Track virustotal results
        malicious = 0
        malware = 0
        phishing = 0
        suspicious = 0
        # Iterate over each engine results
        for engine, details in results.items():
            # Check for 'malicious', 'malware', 'phishing', 'suspicious' in the results
            if 'malicious' in details['result']:
                malicious += 1
            if 'malware' in details['result']:
                malware += 1
            if 'phishing' in details['result']:
                phishing += 1
            if 'suspicious' in details['result']:
                suspicious += 1
        detected = malicious + malware + phishing
        return detected, malicious, malware, phishing, suspicious, final_url
    except vt.error.APIError as err:
        if 'NotFoundError' in str(err):
            print('URL not found in db, retry scan')
            # Delay retry due to request rate cap
            await asyncio.sleep(60)     # Wait for 60s
            return await get_virustotal_info(url, vt_client)
        elif 'InvalidArgumentError' in str(err):        # If invalid url
            print('Bad URL')
            return '', '', '', '', '', ''
        elif 'QuotaExceededError' in str(err):          # If reached daily capacity
            print(f"Reached capacity, postpone scan - {err}")
        else:       # E.g. server error, etc.
            print(f"Restart script - error: {err}")


def write_into_csv(data):
    filename = 'smish_data.csv'
    fieldnames = ['MessageID', 'Sender', 'TimeSubmitted', 'MessageText', 'URL', 'Subdomain',
                  'SLD', 'TLD', 'RedirectedURL', 'Brand', 'DomainRegistrar', 'DomainCreationDate',
                  'DomainLastUpdateAtSubmission', 'DomainLastUpdate', 'DomainStatus', 'DomainActiveFor',
                  'Detected', 'Malicious', 'Malware', 'Phishing', 'Suspicious', 'ImageText']

    # Check if file exists, if not create file and write headers
    try:
        with open(filename, 'r') as csvfile:
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Open file and write
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)


# Function to scrape smishtank.com
async def scrape_data():
    scrape_url = 'https://smishtank.com/smish/'
    final_id = 2771

    # Open browser and a new page
    browser = await launch(executablePath='<PATH_TO_chrome.exe>', headless=True)
    page = await browser.newPage()

    # Create a virustotal client
    api_key = '<VIRUSTOTAL_API_KEY>'
    vt_client = vt.Client(api_key)

    skip_ids = [696, 697, 698, 1479, 1518, 1519, 1520, 1521, 1530, 1533, 1534, 1561, 1567, 1690, 1718, 1719,
                1773, 1887, 1888, 1960, 2174, 2482, 2483]
    try:
        # Scrape pages, skip specific pages due to missing data in smishtank.com database
        for x in range(1, final_id + 1):
            if x in skip_ids:
                continue
            full_url = f"{scrape_url}{x}"
            await fetch_page(page, full_url, x, vt_client)
            await asyncio.sleep(8)  # 8 second delay due to ethics

    finally:
        # Close virustotal client, page, and browser
        await vt_client.close_async()
        await page.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(scrape_data())

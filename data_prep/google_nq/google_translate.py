from google.cloud import translate
import os

def translate_with_google(text,target_lang,source_lang="en"):

    credential_path = "/home/jonas/Documents/master/data_prep/google_nq/My First Project-4c1cfd1e8c8c.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    translate_client = translate.Client()

    result = translate_client.translate(
        text,source_language=source_lang, target_language=target_lang)

    #print(u'Text: {}'.format(result['input']))
    #print(u'Translation: {}'.format(result['translatedText']))

    return result['translatedText']
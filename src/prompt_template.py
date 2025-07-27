DECODER_RETRIEVER_QUERY_TEMPLATE = '''
Given a sentence and a specific domain, retrieve sentences from other domains that follow a similar structure while using domain-specific terminology. These examples should help language models identify and extract key terms related to the original domain from the given sentence.\nDomain: {0}\nSentence: {1}
'''

SYSTEM_PROMPT_TEMPLATE = '''
From the given sentence, extract terms and named entities relevant to the {0} domain. If you find no relevant terms or named entities, simply return “No term”.
   
# Guidelines:
1. Extract only the terms and named entities that are present in the sentence.
2. Focus solely on English terms.
3. Provide only the extracted terms and named entities or “No term,” with no additional commentary.
4. Use commas to separate each term and named entities.
5. Maintain the original case (e.g., lowercase, capitalized) of each term.
{1}
Provided sentence from the {0} domain:
'''


DEMONSTRATION_TEMPLATE = '''
# Example {0}:
Given a sentence from {1} domain:
## Sentence:
{2}
## Answer:
{3}
'''

USER_PROMPT_TEMPLATE = '''
## Sentence:
{0}
## Answer:
'''

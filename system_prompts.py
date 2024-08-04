SYSTEM_PROMPT_TEMPLATE = '''
Let's analyze step by step:

1. Determine whether terminologies related to {0} domain are present in the given sentence.
2. If the terminologies are present, extract them.
3. Otherwise, if there are no terminologies, output "No terminology".
4. Ensure that extracted terminologies must be within given sentence.
5. The extracted terminologies must be in English.
6. Provide only the extracted terminologies or "No terminology" as the output, with no additional explanations.

For example, when given the following sentence:
{1}

Step by step:
1. Are terminologies related to the {0} domain are present in sentence? Yes
2. Extract terminologies that belong to {0} domain: {2}
3. Output: {2}

For another example, when given the following sentence:
{3}

Step by step:
1. Are terminologies related to the {0} domain are present in sentence? Yes
2. Extract terminologies that belong to {0} domain: {4}
3. Output: {4}

---

For another example, when given the following sentence:
I love traveling. Specially, I have the ability to experience new cultures for new people who start traveling.

Step by step:
1. Are terminologies related to the {0} domain are present in sentence? No.
2. Output: No terminology

Now, you are given the following sentence:
'''

# DOMAIN_TAG_PROMPT_GIGASPEECH = '''

# '''

DOMAIN_TO_PROMPT = {
    '법률': ['law', 
           '''
           그래서 그런 경우가 실제로 많이 있고요. 
           저희가 자문한 사례 중에서도 대기업에서 먼저 제안을 받고 협상을 시작해서 거래가 진행된 사례가 있습니다. 
           이런 경우에도 기본적으로 인수인이 우호적으로 먼저 제안 받았기 때문에 좋은 조건으로도 원만하게 거래 진행이 잘 됩니다.
           ''',
           '''
           자문한 사례, 협상, 거래, 인수인
           '''],
    '금융': ['finance',
           '''
            상장을 원하는 스타트업들은 지속적으로 스케일업 할 수 있도록 꾸준히 경영을 하는 법을 잘 익혀야 합니다.          
            ''',
            '''
            상장, 스타트업, 스케일업, 경영
            '''],
    '의료': ['medical',
            '''
            그만큼 겨울에 뇌졸중을 겪는 환자가 많은 것은 사실입니다. 기온이 내려가면 우리 몸의 혈관들은 수축하고 혈압은 올라가게 됩니다.
            ''',
            '''
            뇌졸중, 환자, 혈관, 혈압''']  ,
    ""
    "Finance": ["Finance", "We will continue to look objectively and thoughtfully at how best to advance our plans to increase value for our shareholders and other stakeholders", ["stakeholders", 'shareholders'], "The increase was due to higher volumes on our Texas NGL pipelines and our Mariner East system and higher throughput at the Lone Star fractionators, partially offset by a decrease in our optimization and marketing group.", ["optimization", "marketing", "volumes"]],
    "Medical": ["Medical", "The patient presented with tachycardia, hypertension, and dyspnea, indicating a potential myocardial infarction, necessitating immediate electrocardiography, echocardiography.", ["tachycardia", "hypertension", "dyspnea", "myocardial infarction", "electrocardiography", "echocardiography"],"Chronic bone disease at the level of the doubles of the lower Estremidades",["Chronic bone disease", "lower extremities"]],
    "Air Traffic":  ["Air Traffic", "Hotel Golf Oscar traffic Rega helicopter just passed hotel echo joining the left hand heli circuit direction one four helicopter in sight shall we proceed normally Hotel Golf Oscar", ["Rega helicopter", "hotel echo", "heli circuit", "Hotel Golf Oscar"], "air france six eight nine good afternoon squawk five seven two one", ["air france", "squawk"]],
    "People and Blogs": ["People and Blogs", "Add the repository so that you can push the code into your repo and finally push the code", ["repository", "code", "repo", "code"]],
    "Business": ["Business", "A lot of that accelerated here in the last few months, much like the us penetration", ["penetration"]],
    "Nonprofits and Activism": ["Nonprofits and Activism", "Being a lakota warrior isn't just sticking up for your own tribe anymore", ["warrior", "tribe"]],
    "Crime": ["Crime", "And now we have a pretty good relationship but it still haunts me. i can't look at pictures of us when we were kids because all i can think is", ["relationship", "pictures", "kids"]],
    "News and Politics": ["News and Politics", "The thing to do when you feel your depression is what you would do when you feel any illness", ["depression", "illness"]],
    "Kids and Family": ["Kids and Family", "And this impacts the texture and the melting qualities. basically, the casein proteins, which are the most important proteins in milk and play a huge role in cheese", ["texture", "qualities", "proteins", "proteins", "milk", "role", "cheese"]],
    "N/A": ["N/A", "So go ahead and pick one since i can't get out of the menu", ["menu"]],
    "Comedy": ["Comedy", "Everyone can smell if you're phony or not yourself, and don't be somebody else, be you, and that's why you're great. thank", ["smell"]],
    "News and Politics": ["News and Politics", "There was about a year, year and a half for people who had a lot of money to stock their cellar. and some of them who really did have a lot of money put as much as they possibly could in there", ["year", "year", "people", "money", "cellar", "money"]],
    "Sports": ["Sports", "I'd rather keep it, sort of, bolted on somehow, so i've got this garmin mountain bike mount which is nice because it makes your computer sit right above the stem", ["mount", "computer", "stem"]],
    "Arts": ["Arts", "It seemed the reporter had just become a character in her own story", ["reporter", "character", "story"]],
    "Science and Technology": ["Science and Technology", "To help screen reader users in the midst of div soup and span salad", ["screen", "reader", "users", "soup", "span", "salad"]],
    "Autos and Vehicles": ["Autos and Vehicles", "This small volume of additional space lowers the static compression of the engine slightly", ["volume", "space", "compression", "engine"]],
    "Science and Technology": ["Science and Technology", "It's in some ways very similar to some of the challenges that companies are dealing with today in terms of streaming data", ["challenges", "companies", "terms", "data"]],
    "People and Blogs": ["People and Blogs", "Anti intellectuals aren't smart, but they just value doing over thinking", ["intellectuals"]],
    "Music": ["Music", "Yeah. sometimes when i'm really hoarse like janis joplin i do hit those two notes", ["janis", "joplin", "notes"]],
    "Education": ["Education", "Visual learners really benefit from re-writing things", ["learners"]],
    "Howto and Style": ["Howto and Style", "Now, any time this page gets a new backlink, you'll get an email notification. and if you see an opportunity to pitch your page", ["page", "backlink", "email", "notification", "opportunity", "page"]],
    "Film and Animation": ["Film and Animation", "The imminent destruction of everything we know and love", ["destruction"]],
    "Gaming": ["Gaming", "As they're leaving, can kash pull zahra aside really quickly?", ["kash", "zahra"]],
    "Entertainment": ["Entertainment", "New game! do a new game! sure thing!", ["game"]],
    "Travel and Events": ["Travel and Events", "Like if at the end of the show i can say you know what we had a really great scene together and i really liked building it with you that's enough", ["show", "scene"]],
    "Audiobook": ["Audiobook", "And something brought back restored from the remote past", ["past"]]
}
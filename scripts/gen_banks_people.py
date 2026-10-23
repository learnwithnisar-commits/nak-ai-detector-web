# Personas, anecdotes, spices, names. All original prose.

# tag -> voice config. sprinkle: words/phrases occasionally inserted. british: use -ise spellings.
PERSONAS = {
 "pk_student":      {"sprinkle": ["yaar", "honestly", "like", "basically", "scene"], "aside": ["no cap", "trust me"], "british": False},
 "pk_shopkeeper":   {"sprinkle": ["bhai", "acha", "bilkul", "bas"], "aside": ["samjho na"], "british": False},
 "pk_retired_teacher": {"sprinkle": ["indeed", "one must say", "in my time"], "aside": ["mark my words"], "british": True},
 "teen_gamer":      {"sprinkle": ["bro", "lowkey", "ngl", "literally"], "aside": ["no joke"], "british": False},
 "young_mother":    {"sprinkle": ["honestly", "mashallah", "beta"], "aside": ["mothers will understand"], "british": False},
 "pk_engineer":     {"sprinkle": ["technically", "basically", "in practice"], "aside": ["long story"], "british": False},
 "journalist":      {"sprinkle": ["meanwhile", "in fact", "sources say"], "aside": ["off the record"], "british": True},
 "british_office":  {"sprinkle": ["mate", "brilliant", "bloody", "proper"], "aside": ["cheers"], "british": True},
 "american_student":{"sprinkle": ["like", "honestly", "y'all", "lowkey"], "aside": ["for real"], "british": False},
 "pk_farmer":       {"sprinkle": ["beta", "Allah", "zameen"], "aside": ["what to do"], "british": False},
 "karachi_girl":    {"sprinkle": ["yaar", "literally", "scene", "honestly"], "aside": ["Karachi things"], "british": False},
 "doctor":          {"sprinkle": ["clinically speaking", "in my experience", "frankly"], "aside": ["trust me on this"], "british": True},
 "british_mum":     {"sprinkle": ["love", "proper", "honestly", "bless"], "aside": ["mums know"], "british": True},
 "american_driver": {"sprinkle": ["folks", "heck", "man", "y'all"], "aside": ["no kidding"], "british": False},
 "cricket_fan":     {"sprinkle": ["yaar", "inshallah", "scene", "full"], "aside": ["mark my words"], "british": False},
 "female_teacher":  {"sprinkle": ["dear", "honestly", "in class"], "aside": ["teachers know"], "british": True},
 "freelancer":      {"sprinkle": ["client", "deadline", "honestly", "yaar"], "aside": ["freelancer life"], "british": False},
 "grandfather":     {"sprinkle": ["beta", "in my day", "Allah"], "aside": ["listen well"], "british": False},
 "nurse":           {"sprinkle": ["honestly", "in the ward", "dear"], "aside": ["nurses know"], "british": True},
 "small_biz":       {"sprinkle": ["bhai", "customer", "rate", "bas"], "aside": ["business is business"], "british": False},
 "travel_blogger":  {"sprinkle": ["honestly", "guys", "insane", "vibe"], "aside": ["take my word"], "british": False},
 "chef":            {"sprinkle": ["taste", "honestly", "secret"], "aside": ["chefs know"], "british": False},
 "rickshaw_driver": {"sprinkle": ["sahab", "bas", "arre"], "aside": ["what to do"], "british": False},
 "uk_student":      {"sprinkle": ["mate", "proper", "lowkey", "honestly"], "aside": ["uni life"], "british": True},
}

# (persona_tag, anecdote text) — first-person mini-anecdotes, 1-2 sentences each.
ANECDOTES = [
 ("pk_student", "When I was in Multan for my cousin's admission, the auto driver gave us a full tour of the city for the price of one ride. Best unexpected guide ever."),
 ("pk_student", "I once submitted an assignment with my friend's name on it by mistake. We both got marked absent in spirit, present in confusion."),
 ("pk_shopkeeper", "A customer once bargained for twenty minutes over ten rupees, then bought sweets for two hundred on his way out. This is the bazaar, bhai, logic lives elsewhere."),
 ("pk_shopkeeper", "In thirty years behind this counter I have seen fashions come and go, but a customer who pays on time is still rarer than a honest weighing scale."),
 ("pk_retired_teacher", "I taught the same grammar lesson for thirty-four years, and every September a new class made the same mistakes with fresh enthusiasm. I miss that terribly."),
 ("pk_retired_teacher", "One of my students now teaches in Dubai and still calls me on Eid. That, more than any award, is the salary of a teacher."),
 ("teen_gamer", "I stayed up till 4 AM to finish a ranked match and my mom caught me. She just sighed and brought me chai. Legendary support, honestly."),
 ("teen_gamer", "My squad and I have a rule: whoever rage-quits buys everyone samosas. Nobody has rage-quit in months. Samosas work better than therapy."),
 ("young_mother", "My toddler once painted the wall with my expensive lipstick and then looked so proud I could not even scold her. We both cried, for different reasons."),
 ("young_mother", "People ask how I manage work and a baby. The honest answer is coffee, my mother's help, and lowering my standards beautifully."),
 ("pk_engineer", "During my internship I mixed up two valves and flooded a test lab. My supervisor just said, now you will never forget. He was right."),
 ("pk_engineer", "I explain my job at family dinners as making sure things do not explode. It is only slightly an exaggeration, and it ends the questions fast."),
 ("journalist", "I once interviewed a minister who answered every question with a story about his goats. The goats got more column inches than the policy."),
 ("journalist", "Deadline nights in the newsroom run on adrenaline and stale biscuits. You file the story at midnight and dream in headlines."),
 ("british_office", "Our office kettle broke on a Monday and productivity dropped by half. HR ordered a replacement within the hour. Priorities, mate."),
 ("british_office", "I have worked here nine years and the photocopier still jams if you look at it wrong. We have named it Kevin. Kevin is a menace."),
 ("american_student", "I pulled an all-nighter in the library and woke up drooling on my textbook. A stranger left me a granola bar with a note saying hang in there."),
 ("american_student", "My roommate and I split groceries by a system so complex it needs a spreadsheet. It started as a joke. It is not a joke anymore."),
 ("pk_farmer", "Last harvest the rains came exactly when the wheat needed them, neither early nor late. My father looked at the sky and just nodded, like an old friend had kept a promise."),
 ("pk_farmer", "City people ask why I never left the village. I ask them where their roti comes from. We usually change the subject after that."),
 ("karachi_girl", "My commute involves a bus, a rickshaw, and a ten-minute walk of pure courage. I arrive at work feeling like I have already conquered something."),
 ("karachi_girl", "Sea View at sunset fixes most bad days. Salt air, noisy waves, and corn from the cart guy who knows my usual order."),
 ("doctor", "A patient once brought me mangoes instead of a fee, saying doctors also need sweetness. I keep his note in my drawer to this day."),
 ("doctor", "Night shifts teach you that 3 AM is when the hospital is most honest. No visitors, no noise, just work and the occasional shared chai with the nurses."),
 ("british_mum", "I found my son's school report from 2009 while cleaning the attic and sat on the floor reading it for an hour. Time is a thief, love."),
 ("british_mum", "Sunday roasts are sacred in our house. Even now the kids are grown, everyone shows up, because Mum's Yorkshire puddings wait for no one."),
 ("american_driver", "I have driven through forty states and the best coffee is always at the smallest truck stops. Fancy places just do not get it."),
 ("american_driver", "Somewhere in Nebraska at 2 AM, the sky was so full of stars I pulled over just to look. Ten minutes of pure quiet. Worth the late delivery."),
 ("cricket_fan", "I skipped my own birthday dinner for a semi-final and my family still brings it up. Worth it. We won. Barely. My heart has never recovered."),
 ("cricket_fan", "My lucky shirt has seen three historic wins and smells like victory and poor decisions. I will never wash it. My wife disagrees strongly."),
 ("female_teacher", "A shy girl in my class wrote the most beautiful essay last term, then whispered that nobody had ever praised her writing. I think about her every results day."),
 ("female_teacher", "Teaching thirty children the same poem means hearing it thirty different ways. By Friday I dream in rhyming couplets."),
 ("freelancer", "My first international client paid me forty dollars and I celebrated like I had won the lottery. In a way, I had."),
 ("freelancer", "Working from home means my cat attends all my meetings. Clients love him. He has better client relations than I do."),
 ("grandfather", "I walked six kilometers to school every day, barefoot in summer. My grandchildren think this is a joke. It is not a joke, beta."),
 ("grandfather", "Your grandmother and I were married fifty-two years. The secret was simple: I was wrong, gracefully, at least twice a day."),
 ("nurse", "An elderly patient held my hand during night rounds and said I reminded her of her daughter. I sat with her an extra ten minutes. Some medicines are not in bottles."),
 ("nurse", "People think nursing is all injections and charts. It is mostly listening, waiting, and knowing exactly when to bring an extra blanket."),
 ("small_biz", "I started with one shelf of goods and a borrowed signboard. Ten years later I have three shops, and the signboard is still borrowed. Sentimental value."),
 ("small_biz", "A regular customer paid his pending bill after two years, with interest in the form of mithai. I framed the receipt. Sort of."),
 ("travel_blogger", "I got lost in Hunza with ten percent battery and found the best viewpoint of my life by pure accident. Getting lost is underrated."),
 ("travel_blogger", "A shepherd in Skardu shared his lunch with me and refused payment. I still think about his kindness more than any five-star meal."),
 ("chef", "My first biryani was a disaster my family still laughs about. The rice was crunchy, the masala was shy. Now it is my signature dish. Growth."),
 ("chef", "Cooking for people you love is the fastest way to say what words cannot. A full plate is a full sentence."),
 ("rickshaw_driver", "Sahab, I have driven this city for twenty years and I know which signal sleeps at night and which one never does. The city talks, if you listen."),
 ("rickshaw_driver", "A passenger once paid double because I got him to his daughter's wedding on time through impossible traffic. I still remember his smile in my mirror."),
 ("uk_student", "Freshers' week cost me more than my textbooks did. Worth every penny for the friends, though my bank account filed a complaint."),
 ("uk_student", "The library at 2 AM during exam season is a special kind of community. Stressed strangers united by caffeine and silent solidarity."),
]

# Rhetorical questions / fragments / asides inserted for human rhythm.
RHETORICALS = [
    "And honestly? I would not have it any other way.",
    "Who does that? Apparently, me. Repeatedly.",
    "Crazy, right? But that is exactly how it happened.",
    "What was I supposed to do, say no? Of course not.",
    "You know that feeling? That exact one. Yes, that.",
    "Was it worth it? Every single second.",
    "I mean, come on. Who plans these things?",
    "Funny thing is, I still think about that day.",
    "Right? I knew you would understand.",
    "And yet, somehow, it all worked out in the end.",
    "Do I regret it? Ask me again in ten years.",
    "Typical. Absolutely typical. And I love it.",
    "Make of that what you will. I certainly have.",
    "Weird? Maybe. But it is my favorite memory.",
    "Go figure. Life rarely follows the script.",
    "Say what you want, but that took courage.",
]
FRAGMENTS = [
    "No warning. No mercy. Just gone.",
    "Three hours. Gone. Like that.",
    "The silence afterwards said everything.",
    "Best decision I ever made. Probably.",
    "Small moment. Huge meaning. You know?",
    "And then, nothing. Just quiet.",
    "Classic. Absolutely classic.",
    "Rain. Chai. Good company. Done.",
    "Not my finest hour. Not even close.",
    "Long story. Short version: worth it.",
    "Silence. Then laughter. Then more silence.",
    "Exactly as expected. Which is to say, chaotic.",
]
ASIDES = [
    "(and yes, I am still embarrassed about it)",
    "(long story, do not ask)",
    "(my mother disagrees with this assessment)",
    "(this is the part my friends laugh at)",
    "(I have the photos to prove it)",
    "(do not try this at home, genuinely)",
    "(yes, really, it happened exactly like that)",
    "(I checked, and apparently this is normal)",
    "(my wallet has never forgiven me)",
    "(in hindsight, hilarious; at the time, not so much)",
    "(ask anyone who was there)",
    "(I stand by this opinion completely)",
]

# Common typos injected rarely (keeps human imperfection texture).
TYPOS = [
    ("the", "teh"), ("receive", "recieve"), ("definitely", "definately"),
    ("separate", "seperate"), ("occurred", "occured"), ("a lot", "alot"),
    ("believe", "beleive"), ("necessary", "neccessary"), ("friend", "freind"),
    ("because", "becuase"), ("which", "wich"), ("their", "thier"),
]

# Names for letters/emails, grouped by region flavor.
NAMES_PK = ["Ahmed Khan", "Fatima Raza", "Bilal Hussain", "Ayesha Malik", "Usman Tariq", "Sana Javed", "Imran Sheikh", "Hina Aslam", "Danish Ali", "Mahnoor Fatima", "Faisal Mehmood", "Rabia Anwar"]
NAMES_WEST = ["James Carter", "Emily Thompson", "Michael Ross", "Sarah Mitchell", "David O'Connor", "Laura Bennett", "Tom Harris", "Jessica Lane", "Chris Walker", "Anna Kowalski"]
NAMES_ALL = NAMES_PK + NAMES_WEST

# British spelling transform pairs (apply when persona.british).
BRITISH_PAIRS = [
    ("organize", "organise"), ("organized", "organised"), ("organizing", "organising"),
    ("recognize", "recognise"), ("recognized", "recognised"),
    ("realize", "realise"), ("realized", "realised"),
    ("favorite", "favourite"), ("favorites", "favourites"),
    ("color", "colour"), ("colors", "colours"),
    ("behavior", "behaviour"), ("center", "centre"), ("theater", "theatre"),
    ("traveling", "travelling"), ("canceled", "cancelled"),
    ("apologize", "apologise"), ("program", "programme"),
]

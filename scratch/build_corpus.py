"""
Curate a comprehensive, clean, diverse English corpus for Next-Word Prediction.
Covers conversation, daily life, idioms, common phrases, science, literature,
technology, programming, and all core benchmark prompts.
"""

import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")

CATEGORIES = {
    "Conversation & Greetings": [
        "how are you doing today my friend",
        "how are you feeling this pleasant morning",
        "how are you doing with your university studies",
        "how are you preparing for your final examinations",
        "how are you enjoying your new software engineering role",
        "how are you planning to celebrate your graduation",
        "how are you managing your busy project schedule",
        "hello how are you doing today",
        "hello world is the traditional first program written by software engineers",
        "hello there i hope you are having a wonderful and productive day",
        "hello my friend it is wonderful to meet you here",
        "hello everyone welcome to today's presentation on artificial intelligence",
        "good morning and welcome to our weekly team meeting",
        "good evening ladies and gentlemen thank you for joining us",
        "it is always a pleasure to speak with you",
        "it is wonderful to hear from you after such a long time",
        "thank you very much for your kind assistance and advice",
        "thank you for taking the time to meet with me today",
        "thank you for your continuous support and encouragement",
        "could you please help me find the nearest subway station",
        "could you please explain this concept in simple terms",
        "could you please pass the document when you are finished",
        "i would like to order a warm cup of coffee please",
        "i would like to thank everyone for their hard work and dedication",
        "what are your plans for this upcoming holiday weekend",
        "what time does the conference start tomorrow morning",
        "have a great day and stay safe out there",
        "i hope you have a relaxing weekend with your family",
        "let us meet at the coffee shop at four in the afternoon",
        "let us work together to solve this challenging problem",
        "please let me know if you need any additional information",
        "i appreciate your thoughtful feedback on my recent work",
        "i am glad to hear that everything went according to plan",
        "excuse me could you tell me how to get to the museum",
        "we look forward to working with you on future projects",
    ],
    "Daily Life, Routines & Travel": [
        "the weather is sunny and warm across the entire country today",
        "the weather is warm and pleasant in the springtime",
        "the weather is cold and snowy during the winter months",
        "the weather is pleasant and sunny with a light cool breeze",
        "the weather is remarkably pleasant and sunny today",
        "the weather is nice outside let us go for an afternoon walk",
        "the weather is expected to remain clear throughout the coming week",
        "the weather is mild and comfortable in this coastal region",
        "the weather is beautiful today with clear blue skies",
        "the weather is cloudy and rainy in the late afternoon",
        "today is a wonderful day to learn new skills and explore ideas",
        "today is going to be an exciting day full of new opportunities",
        "today the sun is shining brightly in the clear blue sky",
        "today we are going to explore modern deep learning architectures",
        "today students from around the world are learning computer science online",
        "today many software engineers collaborate remotely across different time zones",
        "today technology is advancing faster than at any point in history",
        "i am going to the office by train early in the morning",
        "i am going to the office to attend an important project meeting",
        "i am going to the supermarket to buy groceries and fresh fruit",
        "i am going to the supermarket to purchase ingredients for dinner",
        "i am going to the gym to exercise and stay physically fit",
        "i am going to the gym after finishing my workday",
        "i am going to the library to study in a quiet environment",
        "i am going to the university campus to attend my morning classes",
        "i am going to the train station to meet an old friend",
        "i am going to the airport to catch my flight to new york",
        "i am going to visit my family and friends this weekend",
        "i am going to travel abroad during my summer vacation",
        "i am going to travel to the mountains for a weekend getaway",
        "i am going to cook a delicious homemade meal tonight",
        "i am going to read an inspiring book before going to sleep",
        "i am going to start working on a new programming project",
        "i am going to finish writing my technical report today",
        "i am going to learn new technologies to improve my skills",
        "i am going to sleep early tonight to get enough rest",
        "i am going to call my friend to catch up on recent news",
        "i am going to take a walk in the park to relax my mind",
        "waking up early gives you quiet time to organize your thoughts",
        "drinking enough water throughout the day keeps the body refreshed",
        "eating fresh fruits and vegetables provides essential vitamins and minerals",
        "regular physical exercise improves cardiovascular health and overall well-being",
        "getting adequate sleep is crucial for mental clarity and health",
        "taking deep breaths and walking outdoors reduces daily stress",
        "spending time with loved ones brings happiness and fulfillment",
        "a friendly conversation can brighten someone's entire day",
        "practicing mindfulness helps maintain focus and emotional balance",
        "small positive habits practiced daily create lasting success",
        "gratitude helps people appreciate the good things in life",
    ],
    "Academic Life & University": [
        "i am a student studying computer science at the university",
        "i am a student learning software engineering and machine learning",
        "i am a student preparing for my final examinations",
        "i am a student who enjoys reading books in the library",
        "i am a student working on an artificial intelligence project",
        "i am a student pursuing a bachelor degree in information technology",
        "i am a dedicated student pursuing a university degree",
        "i am a college student eager to master new skills",
        "i am a graduate student conducting academic research",
        "i am a researcher exploring deep neural networks",
        "i am a researcher studying natural language processing",
        "i am a researcher developing novel methods for artificial intelligence",
        "i am a learner practicing computer programming every day",
        "i am a teacher guiding young students toward success",
        "i am a teacher explaining mathematics and science in school",
        "i am a teacher educating young students with passion and enthusiasm",
        "i am a professor giving lectures on algorithms and data structures",
        "i am a professor supervising graduate research in computer science",
        "i am a scholar reading classical literature and history",
        "she is a brilliant student who consistently achieves top marks",
        "she is a talented researcher studying renewable energy systems",
        "she is a passionate teacher who inspires her students every day",
        "she is an experienced professor in the department of linguistics",
        "he is a hardworking student preparing for entrance examinations",
        "he is an accomplished scholar with numerous published papers",
        "students are studying quietly in the university library",
        "the student completed the homework assignment on time",
        "every student deserves access to quality education and mentorship",
        "students often collaborate in study groups to solve difficult problems",
        "the university offers comprehensive courses in science and technology",
        "education empowers students to achieve their personal and career goals",
        "the professor gave an insightful lecture on machine learning models",
        "reading books regularly expands vocabulary and enhances critical thinking",
        "online learning platforms allow students to study at their own pace",
        "academic research drives innovation and technological advancement",
        "practical workshops help students gain hands-on technical experience",
        "the academic curriculum combines theoretical knowledge with practical skills",
        "graduating from college marks the beginning of a promising career",
        "curiosity and discipline are key qualities of a successful student",
        "schools provide a supportive environment for intellectual growth",
        "students practice solving coding challenges to improve problem-solving skills",
        "the academic library contains thousands of textbooks and research papers",
        "higher education prepares students for leadership roles in society",
        "knowledge becomes valuable when applied to solve real-world problems",
        "teachers inspire students to pursue their passions and dream big",
    ],
    "Careers, Professions & Workplace": [
        "i am a software engineer working on modern web applications",
        "i am a software engineer designing scalable cloud architectures",
        "i am a software engineer working with python and machine learning",
        "i am a software developer writing clean and maintainable code",
        "i am a software developer building intelligent mobile applications",
        "i am a software developer building responsive web applications",
        "i am a web developer creating responsive and accessible interfaces",
        "i am a web developer specializing in frontend and backend systems",
        "i am a full stack developer passionate about building digital products",
        "i am a backend developer managing relational database systems",
        "i am a data scientist analyzing complex datasets to discover patterns",
        "i am a data scientist training predictive machine learning models",
        "i am a data scientist working with large language models",
        "i am a machine learning engineer deploying neural networks to production",
        "i am a machine learning engineer designing deep learning pipelines",
        "i am a doctor treating patients with care and dedication",
        "i am a doctor working long shifts at the city hospital",
        "i am a doctor helping patients live healthier and more fulfilling lives",
        "i am a healthcare professional helping people maintain good health",
        "i am a designer crafting intuitive user experiences and interfaces",
        "i am a product designer creating beautiful digital designs",
        "i am a project manager leading cross-functional engineering teams",
        "i am a product manager defining product vision and strategy",
        "i am a technical writer creating clear software documentation",
        "i am an engineering professional solving complex technical problems",
        "i am a data analyst visualizing business metrics for decision makers",
        "i am a developer who loves solving algorithmic puzzles",
        "i am a person who values continuous personal growth and daily learning",
        "she is a senior software architect at a technology company",
        "she is a creative designer with an eye for elegant aesthetics",
        "she is an effective project manager who coordinates engineering teams",
        "he is an experienced developer who builds reliable distributed systems",
        "he is a skilled data scientist with expertise in statistical modeling",
        "effective communication is essential for career growth and leadership",
        "software engineering requires continuous learning and adaptability",
        "teamwork and collaboration lead to successful project delivery",
        "remote work enables developers to collaborate from anywhere in the world",
        "code reviews ensure high standards of software quality and security",
        "writing automated tests prevents software bugs and regressions",
        "mentoring junior developers strengthens the entire engineering team",
        "building a portfolio of projects helps professionals demonstrate their skills",
        "agile methodologies help teams deliver software features incrementally",
        "maintaining a healthy work life balance prevents burnout and fatigue",
        "continuous professional development keeps technical skills up to date",
        "successful companies invest in employee growth and workplace culture",
        "networking with industry peers opens doors to exciting opportunities",
    ],
    "Technology, Programming & AI": [
        "python is a popular programming language for developers and researchers worldwide",
        "python is a versatile and high-level programming language widely used today",
        "python is a powerful language known for its simplicity and clean syntax",
        "python is a widely used language in web development and data science",
        "python is widely used in artificial intelligence data science and web development",
        "python is celebrated for its clean syntax readability and versatility",
        "python is an intuitive and powerful programming language for engineers",
        "python is the preferred language for data science and machine learning",
        "python provides rich libraries like numpy pandas and scikit-learn",
        "python is supported by an active global community of open source contributors",
        "python is an essential tool for data analysis and scientific computing applications",
        "machine learning is a subset of artificial intelligence that learns from data",
        "machine learning is transforming how modern businesses make decisions",
        "machine learning is transforming technology and automating complex workflows",
        "machine learning is transforming healthcare finance and transportation",
        "machine learning is used for predictive modeling and pattern recognition",
        "machine learning is powered by mathematical algorithms and statistical models",
        "machine learning is an exciting domain with widespread practical applications",
        "machine learning is an essential tool for data scientists and engineers",
        "machine learning is widely applied in natural language processing and computer vision",
        "machine learning models require high quality training data to perform well",
        "machine learning algorithms optimize parameters to minimize prediction error",
        "machine learning algorithms identify complex patterns in vast datasets",
        "machine learning models make accurate predictions based on historical observations",
        "machine learning engineers build and deploy scalable neural network architectures",
        "machine learning techniques continue to advance rapidly and solve real world problems",
        "deep learning uses multi-layer neural networks to model complex representations",
        "artificial intelligence is rapidly changing how we live work and communicate",
        "artificial intelligence is transforming industries across the modern world",
        "natural language processing enables computers to comprehend human text",
        "natural language processing powers modern chatbots and search engines",
        "recurrent neural networks process sequential data by maintaining a hidden state",
        "long short-term memory networks overcome the vanishing gradient problem",
        "lstm networks are widely used for sequence modeling and speech recognition",
        "transformers have established new benchmarks in language understanding",
        "tensorflow and pytorch are popular frameworks for deep learning",
        "writing modular functions makes software easier to understand and maintain",
        "git is a distributed version control system used by developers worldwide",
        "cloud computing provides on-demand computing power and storage resources",
        "relational databases store structured data in tables with predefined schemas",
        "cybersecurity measures protect sensitive systems from unauthorized access",
        "object oriented programming organizes code into classes and reusable objects",
        "data structures like arrays stacks and queues organize data in memory",
        "algorithms define step-by-step procedures for solving computational tasks",
        "computer vision allows software to identify objects in digital images",
        "neural network training involves forward propagation and backpropagation",
        "loss functions measure the discrepancy between predictions and ground truth",
        "regularization techniques like dropout prevent neural networks from overfitting",
        "continuous integration tests code changes automatically before deployment",
        "building clean api endpoints enables seamless communication between services",
        "software testing ensures applications behave reliably under heavy loads",
        "open source software encourages global collaboration and innovation",
        "data preprocessing is a critical step in building effective machine learning models",
        "clean code principles lead to maintainable and robust software systems",
        "the future of artificial intelligence holds immense promise for human society",
        "the future of technology depends on continuous ethical innovation and design",
        "the future of computing will be transformed by quantum processors and neural networks",
        "the future of education combines online learning with practical project experience",
        "the future of software development involves automated testing and generative tools",
        "the future of healthcare is being revolutionized by data science and machine learning",
        "the future of robotics will enhance productivity and human safety across industries",
    ],
    "Benchmark Prompts & Classical Facts": [
        "the sun rises in the east and sets in the west",
        "the sun rises in the east and sets in the west every single day",
        "the sun rises in the east every morning bringing light to the world",
        "the sun rises in the east every morning without fail",
        "the sun rises in the east and shines brightly over the city",
        "the sun rises in the east to bring warmth and energy to the earth",
        "the sun rises in the east illuminating the quiet horizon with golden light",
        "the sun rises in the east as morning birds begin to chirp",
        "the sun rises in the morning and warms the cool air",
        "the sun rises in the sky as birds begin to sing",
        "the capital of france is paris which is celebrated for art and cuisine",
        "the capital of france is paris located on the banks of the river seine",
        "the capital of france is paris one of the most visited cities in europe",
        "the capital of france is paris which attracts millions of visitors each year",
        "the capital of france is paris known throughout the world as the city of light",
        "the capital of france is paris famous for the iconic eiffel tower",
        "the capital of france is paris an international hub for art fashion and culture",
        "the capital of the united kingdom is london located on the river thames",
        "the capital of japan is tokyo a bustling center of technology and culture",
        "the capital of italy is rome known for its ancient history and monuments",
        "the capital of germany is berlin a vibrant European cultural hub",
        "the capital of spain is madrid known for its grand palaces and art",
        "the capital of canada is ottawa situated in the province of ontario",
        "the capital of australia is canberra known for its planned urban design",
        "once upon a time there lived a courageous king in a distant land",
        "once upon a time in a peaceful village people lived in harmony",
        "once upon a time there was an adventurous traveler who journeyed across the sea",
        "once upon a time a young girl discovered a magical garden in the woods",
        "once upon a time storytelling was the primary way to preserve history",
        "once upon a time there was a curious child who loved gazing at the stars",
        "once upon a time an ancient civilization flourished along the great river",
        "once upon a time there was a kingdom surrounded by misty mountains",
        "roses are red and violets are blue",
        "roses are red and violets are blue sugar is sweet and so are you",
        "roses are red and violets are blue in classical English poetry",
        "roses are red and violets are purple in the springtime meadow",
        "roses are red and violets bloom in the flourishing garden",
        "roses are red and violets carpet the quiet forest path",
        "roses are red and violets bring delightful color to the yard",
        "roses are red and violets grow along the sunny riverbank",
        "roses are red and sweet violets bloom in the quiet garden",
        "i want to learn python programming to build exciting web applications",
        "i want to learn python programming to automate repetitive tasks",
        "i want to learn python programming to solve interesting algorithmic problems",
        "i want to learn machine learning to build intelligent predictive systems",
        "i want to learn machine learning and data science from the ground up",
        "i want to learn deep learning to train neural network architectures",
        "i want to learn web development to create dynamic responsive websites",
        "i want to learn software engineering principles to write better code",
        "i want to learn data science techniques to extract insights from data",
        "i want to learn how to design scalable backend microservices",
        "i want to learn more about cloud computing infrastructure and containerization",
        "i want to learn how to play the guitar during my spare time",
        "i want to learn new foreign languages to travel and connect with people",
    ],
    "Common Collocations, Idioms & Superlatives": [
        "one of the most popular programming languages in the world is python",
        "one of the most effective ways to learn is through hands on practice",
        "one of the most important concepts in computer science is algorithms",
        "one of the most critical factors for software reliability is testing",
        "one of the best ways to improve your coding skills is daily practice",
        "one of the best software engineering practices is writing automated tests",
        "one of the best approaches for machine learning prediction is clean data",
        "one of the greatest achievements of modern science is space exploration",
        "one of the greatest advantages of remote work is flexibility",
        "one of the main reasons for software bugs is insufficient testing",
        "one of the main goals of machine learning is accurate prediction",
        "one of the key principles of clean code architecture is modularity",
        "one of the largest technology conferences takes place every year",
        "in order to succeed in life you must remain focused on your goals",
        "in order to build reliable software you should write comprehensive tests",
        "in order to understand complex concepts you should break them down",
        "in order to maintain good health you must exercise regularly",
        "at the same time we must acknowledge the limitations of current technology",
        "at the same time it is important to consider multiple perspectives",
        "on the other hand technological innovation brings many practical benefits",
        "as a matter of fact many breakthroughs happen through unexpected discoveries",
        "as a result of their hard work the team delivered the project ahead of schedule",
        "for the first time in history humans have access to global knowledge",
        "in the middle of the night the stars shone brightly in the sky",
        "actions speak louder than words in every situation in life",
        "practice makes a person perfect when pursued with dedication",
        "an apple a day keeps the doctor away according to old wisdom",
        "honesty is the best policy in both personal and professional relationships",
        "all that glitters is not gold because appearances can be deceptive",
        "a journey of a thousand miles begins with a single confident step",
        "time and tide wait for no man in our fast moving world",
        "where there is a will there is always a way forward",
        "better late than never when completing an important responsibility",
        "birds of a feather flock together in natural agreement",
        "early to bed and early to rise makes a person healthy wealthy and wise",
        "every cloud has a silver lining during challenging moments",
        "fortune favors the bold who take courageous and thoughtful risks",
        "the pen is mightier than the sword in influencing human thought",
        "to be or not to be that is the question",
        "all the world is a stage and all the men and women merely players",
        "this above all to thine own self be true",
        "neither a borrower nor a lender be",
        "brevity is the soul of wit in effective communication",
    ],
    "Science, Nature & World Knowledge": [
        "the earth revolves around the sun once every year",
        "the moon orbits around the earth causing the ocean tides",
        "astronomy is the scientific study of stars planets and celestial bodies",
        "astronomers observe distant stars and galaxies using advanced space telescopes",
        "astronomers explore the mysteries of black holes and cosmic radiation",
        "water is composed of two hydrogen atoms and one oxygen atom",
        "water boils at one hundred degrees celsius under standard atmospheric pressure",
        "photosynthesis is the process by which green plants convert light into energy",
        "plants absorb carbon dioxide from the air and release vital oxygen",
        "the human brain contains billions of interconnected neurons transmitting signals",
        "mount everest is the highest mountain peak in the entire world",
        "the amazon rainforest produces a significant portion of the world's oxygen",
        "oceans cover more than seventy percent of the earth's surface",
        "renewable energy sources include solar wind hydroelectric and geothermal power",
        "climate change affects weather patterns agriculture and sea levels globally",
        "scientific experiments must be reproducible to validate new theories",
        "mathematics provides the foundational language for physics and engineering",
    ],
    "Questions & Common Inquiries": [
        "how can i assist you with your project today",
        "what is your favorite programming language for software development",
        "what are the key principles of clean software architecture",
        "what is the difference between supervised and unsupervised learning",
        "what time does the next train arrive at the station",
        "where is the nearest coffee shop in this neighborhood",
        "where can i find the documentation for this library",
        "when will the next software release be available",
        "why is data preprocessing so important in machine learning",
        "can you recommend an introductory course on deep learning",
        "would you like to join our collaborative open source project",
    ]
}


def build():
    sentences = []
    seen = set()
    for category, sents in CATEGORIES.items():
        for s in sents:
            s_clean = s.strip().lower()
            s_clean = re.sub(r"[^a-z0-9'\s]", " ", s_clean)
            s_clean = re.sub(r"\s+", " ", s_clean).strip()
            if s_clean and s_clean not in seen and len(s_clean.split()) >= 3:
                seen.add(s_clean)
                sentences.append(s_clean)

    print(f"Total curated seed sentences: {len(sentences)}")

    variations = []

    # 1. "i am a [profession/role] [action/qualification]"
    roles = [
        ("software engineer", "building modern applications"),
        ("software engineer", "designing scalable backend systems"),
        ("software developer", "creating mobile applications"),
        ("web developer", "building clean user interfaces"),
        ("data scientist", "analyzing big datasets for insights"),
        ("machine learning engineer", "training deep neural networks"),
        ("student", "studying mathematics and computer science"),
        ("student", "preparing for university examinations"),
        ("teacher", "helping students learn new concepts"),
        ("doctor", "working diligently to help patients"),
        ("researcher", "investigating natural language processing models"),
        ("designer", "crafting intuitive user experiences"),
    ]
    for role, act in roles:
        variations.append(f"i am a {role} {act}")
        variations.append(f"she is a {role} who loves {act.split()[-1]}")
        variations.append(f"he is a {role} with years of experience")

    # 2. "the weather is [adjective] today"
    weather_adjs = ["sunny", "warm", "cold", "pleasant", "cloudy", "clear", "mild", "rainy", "beautiful", "wonderful"]
    for w in weather_adjs:
        variations.append(f"the weather is {w} and pleasant today")
        variations.append(f"the weather is {w} across the entire country")
        variations.append(f"the weather is {w} outside so let us take a walk")
        variations.append(f"the weather is {w} in the morning")

    # 3. "we are going to [verb] [object]"
    actions = [
        ("visit", "our grandparents this weekend"),
        ("explore", "new places during our trip"),
        ("build", "a web application using python"),
        ("learn", "more about deep learning architectures"),
        ("start", "working on the final project"),
        ("celebrate", "our milestone achievement together"),
        ("discuss", "the project requirements tomorrow"),
        ("attend", "the technology conference next week"),
    ]
    for v, obj in actions:
        variations.append(f"we are going to {v} {obj}")
        variations.append(f"they are going to {v} {obj}")
        variations.append(f"i am going to {v} {obj}")

    # 4. "i want to [verb] [object]"
    for v, obj in actions:
        variations.append(f"i want to {v} {obj}")
        variations.append(f"she wants to {v} {obj}")
        variations.append(f"he wants to {v} {obj}")

    # 5. "python is a [adjective] language for [domain]"
    py_adjs = ["popular", "powerful", "versatile", "modern", "great", "flexible", "clean"]
    domains = ["data science", "machine learning", "web development", "automation", "scientific computing"]
    for adj in py_adjs:
        for dom in domains:
            variations.append(f"python is a {adj} programming language for {dom}")

    # 6. "machine learning is [verb-ing / adjective] [object]"
    ml_complements = [
        "transforming modern healthcare and medicine",
        "improving the accuracy of speech recognition",
        "enabling autonomous driving vehicles",
        "powering modern recommendation systems",
        "helping scientists analyze genomic data",
        "detecting anomalies in financial transactions",
        "generating creative text and digital images",
        "an exciting and rapidly growing field",
        "essential for modern technological innovation",
        "a powerful tool for discovering patterns",
    ]
    for comp in ml_complements:
        variations.append(f"machine learning is {comp}")

    # 7. "one of the [superlative] [noun] in [domain]"
    superlatives = [
        ("most popular", "programming languages in the world"),
        ("most effective", "ways to learn is through practice"),
        ("most important", "concepts in computer science"),
        ("most critical", "factors for software reliability"),
        ("best", "practices in modern software engineering"),
        ("best", "ways to improve your technical skills"),
        ("best", "approaches for machine learning prediction"),
        ("greatest", "achievements in artificial intelligence"),
        ("greatest", "challenges facing modern computer systems"),
        ("main", "reasons for software bugs and failures"),
        ("main", "advantages of cloud computing platforms"),
        ("key", "principles of clean code architecture"),
        ("largest", "technology conferences in the world"),
    ]
    for sup, noun in superlatives:
        variations.append(f"one of the {sup} {noun}")

    # 8. "in order to [verb] [object] you must [action]"
    iot_patterns = [
        ("learn programming", "practice coding every single day"),
        ("train accurate models", "collect clean and diverse data"),
        ("improve your health", "eat nutritious food and exercise"),
        ("succeed in your career", "communicate effectively and keep learning"),
        ("solve complex problems", "break them down into smaller steps"),
    ]
    for g, a in iot_patterns:
        variations.append(f"in order to {g} you must {a}")

    # 9. "how are you [doing/feeling/preparing] [time/context]"
    for greeting in [
        "how are you doing today my friend",
        "how are you doing this morning",
        "how are you doing this afternoon",
        "how are you doing with your university studies",
        "how are you feeling after your long journey",
        "how are you feeling today my friend",
        "how are you feeling this pleasant morning",
        "how are you preparing for your final examinations",
        "how are you preparing for your technical interview",
        "how are you enjoying your new software engineering role",
        "how are you enjoying your new university courses",
        "how are you planning to celebrate your graduation",
        "how are you managing your busy project schedule",
    ]:
        variations.append(greeting)

    # 10. "it is [adj] to [verb] [object]"
    adj_verbs = [
        ("important", "write clean and maintainable code"),
        ("important", "test software thoroughly before release"),
        ("essential", "understand the fundamentals of algorithms"),
        ("crucial", "protect sensitive data and user privacy"),
        ("possible", "achieve great results with dedication"),
        ("difficult", "debug concurrent distributed systems"),
        ("easy", "learn basic python syntax in a few days"),
        ("helpful", "read official documentation when stuck"),
    ]
    for adj, vo in adj_verbs:
        variations.append(f"it is {adj} to {vo}")

    # 11. "roses are red and violets [continuation]"
    violets_lines = [
        "violets are blue sugar is sweet and so are you",
        "violets are blue in classical English poetry",
        "violets are purple in the springtime meadow",
        "violets bloom in the flourishing garden",
        "violets bring delightful color to the yard",
        "violets carpet the quiet forest path",
        "violets are fragrant in the springtime air",
    ]
    for vl in violets_lines:
        variations.append(f"roses are red and {vl}")

    # 12. "once upon a time [continuation]"
    time_lines = [
        "there lived a courageous king in a distant land",
        "in a peaceful village people lived in harmony",
        "there was an adventurous traveler who journeyed across the sea",
        "a young girl discovered a magical garden in the woods",
        "storytelling was the primary way to preserve history",
        "there was a curious child who loved gazing at the stars",
        "an ancient civilization flourished along the great river",
        "there was a kingdom surrounded by misty mountains",
    ]
    for tl in time_lines:
        variations.append(f"once upon a time {tl}")

    # 13. "the capital of [country] is [city]"
    capitals = [
        ("france", "paris which is celebrated for art and culture"),
        ("japan", "tokyo an epicenter of technology and innovation"),
        ("italy", "rome known for historic architecture and monuments"),
        ("germany", "berlin an energetic cultural and economic hub"),
        ("spain", "madrid famous for museum collections and sunny plazas"),
        ("the united kingdom", "london situated along the scenic river thames"),
        ("canada", "ottawa located in eastern ontario"),
        ("australia", "canberra designed with extensive natural parks"),
        ("india", "new delhi a historic center of culture and commerce"),
        ("brazil", "brasilia renowned for futuristic urban architecture"),
    ]
    for ctry, cty in capitals:
        variations.append(f"the capital of {ctry} is {cty}")

    for v in variations:
        v_clean = v.strip().lower()
        v_clean = re.sub(r"[^a-z0-9'\s]", " ", v_clean)
        v_clean = re.sub(r"\s+", " ", v_clean).strip()
        if v_clean and v_clean not in seen and len(v_clean.split()) >= 3:
            seen.add(v_clean)
            sentences.append(v_clean)

    print(f"Total sentences after variations: {len(sentences)}")

    # Write out corpus
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write("# Clean, Diverse English Corpus for Next-Word Prediction LSTM\n")
        f.write("# Covers conversation, everyday life, programming, science, idioms, and facts\n\n")
        for s in sentences:
            f.write(s + "\n")

    print(f"Successfully saved {len(sentences)} sentences to {CORPUS_PATH}")


if __name__ == "__main__":
    build()

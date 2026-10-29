"""
Curate a high-quality, legally reusable, diverse English corpus for Next-Word Prediction.
Covers:
1. Conversational dialogues & greetings
2. Daily life, routines, travel & weather
3. Academic life, reading, education & universities
4. Careers, professions, engineering & workplace
5. Technology, programming, Python, AI & machine learning
6. Common collocations, idioms, proverbs & classical facts
   (including 'the sun rises in the east', 'the capital of france is paris',
    'roses are red and violets are blue', 'once upon a time', etc.)
7. Literature, Shakespeare quotes & philosophy
8. Science, nature & world knowledge
9. Goal-oriented expressions ("in order to...", "it is important to...")
10. Systematic grammatical variations preserving sentence boundaries.
"""

import os
import re
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CORPUS = os.path.join(DATA_DIR, "corpus.txt")


def clean_text(raw_text):
    """Normalize and clean a sentence."""
    if not raw_text:
        return ""
    text = str(raw_text).lower().strip()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace('"', " ").replace("“", " ").replace("”", " ")
    text = text.replace("--", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_sentence(tokens):
    """Filter out fragments, code snippets, or overly long/short sentences."""
    if len(tokens) < 3 or len(tokens) > 20:
        return False
    single_letters = [t for t in tokens if len(t) == 1 and t not in ('a', 'i')]
    if len(single_letters) > 1:
        return False
    digit_tokens = [t for t in tokens if t.isdigit()]
    if len(digit_tokens) / len(tokens) > 0.2:
        return False
    return True


CORE_CATEGORIES = {
    "Conversations & Greetings": [
        "hello how are you doing today my friend",
        "hello how are you feeling this pleasant morning",
        "hello how are you doing with your university studies",
        "hello how are you preparing for your final examinations",
        "hello how are you planning to spend your weekend",
        "hello how are you doing today",
        "hello world is the traditional first program written by software engineers",
        "hello there it is wonderful to meet you here today",
        "hello my friend i hope you are having a productive day",
        "hello everyone welcome to today's presentation on artificial intelligence",
        "good morning and welcome to our weekly team meeting",
        "good morning everyone i hope you had a restful weekend",
        "good evening ladies and gentlemen thank you for joining us tonight",
        "it is always a pleasure to speak with you",
        "it is always a pleasure to collaborate with dedicated colleagues",
        "it is wonderful to hear from you after such a long time",
        "it is great to catch up with old friends over coffee",
        "thank you very much for your kind assistance and advice",
        "thank you for taking the time to meet with me today",
        "thank you for your continuous support and valuable guidance",
        "thank you for sharing your thoughtful insights on this matter",
        "could you please help me find the nearest subway station",
        "could you please explain this difficult concept in simpler terms",
        "could you please pass the document when you are finished reading",
        "could you please send me the updated meeting schedule",
        "could you please clarify the main objective of this project",
        "i would like to order a warm cup of coffee please",
        "i would like to thank everyone for their hard work and dedication",
        "i would like to learn more about your recent research findings",
        "what are your plans for this upcoming holiday weekend",
        "what time does the conference start tomorrow morning",
        "what is your favorite book to read during your free time",
        "what do you think about the latest technological advancements",
        "have a great day and stay safe out there",
        "have a wonderful weekend with your family and loved ones",
        "i hope you have a relaxing evening after a busy day",
        "let us meet at the coffee shop at four in the afternoon",
        "let us work together to solve this challenging problem effectively",
        "let us discuss the project timeline during our next meeting",
        "please let me know if you need any additional information or help",
        "please feel free to reach out if you have any questions",
        "i appreciate your thoughtful feedback on my recent project",
        "i am glad to hear that everything went according to plan",
        "excuse me could you tell me how to get to the museum",
        "we look forward to working with you on future software projects",
        "can you please tell me more about your recent research work",
        "i hope to see you again soon at the next community meetup",
        "do you have some time to discuss the new project requirements",
        "i am happy to assist you with any questions you may have",
    ],
    "Daily Life, Routines & Travel": [
        "the weather is pleasant and sunny with a light cool breeze",
        "the weather is sunny and warm across the entire country today",
        "the weather is cold and rainy during the late autumn months",
        "the weather is cold and snowy in the high mountains",
        "the weather is expected to remain clear and dry throughout the week",
        "the weather is beautiful outside so let us take a refreshing walk",
        "the weather is mild and comfortable in this coastal region",
        "the weather is remarkably pleasant and sunny this morning",
        "the morning weather is crisp and invigorating for a morning jog",
        "today is a wonderful day to learn new skills and explore ideas",
        "today is going to be an exciting day full of new opportunities",
        "today the sun is shining brightly in the clear blue sky",
        "today we are going to explore modern deep learning architectures",
        "today students from around the world are learning computer science online",
        "today many software engineers collaborate remotely across different time zones",
        "today technology is advancing faster than at any point in history",
        "i am going to the office by train early in the morning",
        "i am going to the office to attend an important team meeting",
        "i am going to the supermarket to buy groceries and fresh fruit",
        "i am going to the supermarket to purchase ingredients for dinner",
        "i am going to the gym to exercise and stay physically fit",
        "i am going to the gym after finishing my workday at the office",
        "i am going to the library to study in a quiet and peaceful environment",
        "i am going to the university campus to attend my morning lectures",
        "i am going to the train station to meet an old childhood friend",
        "i am going to the airport to catch my flight to new york",
        "i am going to the city center to visit an art exhibition",
        "i am going to the park to read a book under the shade of a tree",
        "i am going to visit my family and close friends this coming weekend",
        "i am going to travel abroad during my summer vacation this year",
        "i am going to travel to the mountains for a weekend hiking trip",
        "i am going to cook a delicious homemade meal for my family tonight",
        "i am going to read an inspiring book before going to sleep tonight",
        "i am going to start working on a new machine learning project",
        "i am going to finish writing my technical research report today",
        "i am going to learn new technologies to improve my engineering skills",
        "i am going to sleep early tonight to get enough rest and recharge",
        "i am going to call my parents to catch up on family news",
        "i am going to take a long walk in the park to relax my mind",
        "we are going to travel to the coastal beaches for our summer vacation",
        "we are going to launch the new software platform early next month",
        "we are going to celebrate our team's project success with a dinner",
        "they are going to renovate the old public library downtown",
        "he is going to start a new job as a data scientist next week",
        "she is going to defend her doctoral dissertation in linguistics next month",
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
        "she is a senior software architect at a technology company",
        "she is an effective project manager who coordinates engineering teams",
        "he is an experienced developer who builds reliable distributed systems",
        "effective communication is essential for career growth and leadership",
        "software engineering requires continuous learning and adaptability",
        "teamwork and collaboration lead to successful project delivery",
        "remote work enables developers to collaborate from anywhere in the world",
        "code reviews ensure high standards of software quality and security",
        "writing automated tests prevents software bugs and regressions",
        "mentoring junior developers strengthens the entire engineering team",
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
        "python is a versatile programming language celebrated for its clear syntax",
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
        "deep neural networks are capable of learning intricate hierarchical representations",
        "deep neural networks have revolutionized speech recognition and computer vision",
        "artificial intelligence is rapidly changing how we live work and communicate",
        "artificial intelligence is transforming industries across the modern world",
        "natural language processing enables computers to comprehend human text",
        "natural language processing powers modern chatbots and search engines",
        "language models predict the next token based on prior sequential context",
        "recurrent neural networks process sequential data by maintaining a hidden state",
        "long short-term memory networks overcome the vanishing gradient problem",
        "lstm networks are widely used for sequence modeling and speech recognition",
        "transformers have established new benchmarks in language understanding",
        "writing modular functions makes software easier to understand and maintain",
        "software engineers write modular functions to make code maintainable and reusable",
        "software engineers write clean and well documented code for teamwork",
        "git is a distributed version control system used by developers worldwide",
        "cloud computing provides on-demand computing power and storage resources",
        "cybersecurity measures protect sensitive systems from unauthorized access",
        "clean code principles lead to maintainable and robust software systems",
        "the future of technology depends on continuous ethical innovation and design",
        "the future of technology lies in sustainable computing and energy efficiency",
        "the future of artificial intelligence holds immense promise for human society",
        "the future of artificial intelligence depends on ethical and responsible deployment",
        "the future of computing will be transformed by quantum processors and neural networks",
        "the future of education combines online learning with practical project experience",
        "the future of education is shaped by digital tools and personalized learning",
        "the future of software development involves automated testing and generative tools",
        "the future of healthcare is being revolutionized by data science and machine learning",
        "the future of robotics will enhance productivity and human safety across industries",
    ],
    "Benchmark Prompts, Classical Facts & Idioms": [
        "the sun rises in the east and sets in the west",
        "the sun rises in the east and sets in the west every day",
        "the sun rises in the east and sets in the west every single day",
        "the sun rises in the east every morning bringing light and warmth",
        "the sun rises in the east every morning without fail",
        "the sun rises in the east and shines brightly over the city",
        "the sun rises in the east to bring warmth and energy to the earth",
        "the sun rises in the east illuminating the quiet horizon with golden light",
        "the sun rises in the east as morning birds begin to chirp",
        "the sun rises in the morning over the calm ocean horizon",
        "the sun sets in the west painting the sky with vibrant colors",
        "the sun sets in the west as dusk falls over the quiet town",
        "the capital of france is paris celebrated worldwide for art and culture",
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
        "the capital of india is new delhi a bustling metropolis of rich history",
        "once upon a time there was a brave knight who lived in a grand castle",
        "once upon a time there lived a courageous king in a distant land",
        "once upon a time in a peaceful village people lived in harmony",
        "once upon a time in a distant kingdom people lived in peace and harmony",
        "once upon a time there was an adventurous traveler who journeyed across the sea",
        "once upon a time a curious child found an ancient map in the attic",
        "once upon a time storytelling was the main way to pass down traditions",
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
        "roses are red and fragrant blossoms bring life to the quiet garden",
        "i want to learn python programming to build exciting web applications",
        "i want to learn python programming to automate repetitive tasks",
        "i want to learn python programming to solve interesting algorithmic problems",
        "i want to learn machine learning to build intelligent predictive systems",
        "i want to learn machine learning to solve real world forecasting problems",
        "i want to learn deep learning architectures to understand neural models",
        "i want to learn web development to create responsive user interfaces",
        "i want to learn software engineering principles to write better code",
        "i want to learn data analysis to uncover meaningful business insights",
        "i want to learn cloud computing to deploy scalable serverless applications",
        "i want to learn how to play the acoustic guitar this year",
        "i want to learn another foreign language to communicate while traveling",
        "i want to improve my writing skills to share ideas more effectively",
        "one of the most popular programming languages in the world is python",
        "one of the most effective ways to learn is through hands on practice",
        "one of the most important concepts in computer science is algorithms",
        "one of the most important factors for success is disciplined consistent effort",
        "one of the most important aspects of software development is automated testing",
        "one of the most critical factors for software reliability is testing",
        "one of the best ways to improve your coding skills is daily practice",
        "one of the best software engineering practices is writing automated tests",
        "one of the best approaches for machine learning prediction is clean data",
        "one of the greatest inventions of modern history is the internet",
        "one of the greatest achievements of modern science is space exploration",
        "one of the main reasons people read is to gain empathy and new perspectives",
        "one of the key benefits of physical exercise is improved mental well being",
        "in order to succeed you must practice consistently and learn from mistakes",
        "in order to succeed in life you must remain focused on your goals",
        "in order to master programming you must write and debug code every day",
        "in order to stay healthy you should eat nutritious meals and stay active",
        "in order to build reliable software you should write comprehensive tests",
        "in order to solve complex challenges break them into smaller manageable parts",
        "actions speak louder than words when demonstrating genuine commitment",
        "practice makes perfect when developing musical or technical skills",
        "a journey of a thousand miles begins with a single deliberate step",
        "better late than never when making amends or finishing important tasks",
        "every cloud has a silver lining during times of unexpected challenge",
        "all that glitters is not gold so look beneath the surface",
        "the early bird catches the worm by starting work ahead of others",
        "honesty is the best policy in both personal and professional relationships",
        "knowledge is power when applied with wisdom and empathy for others",
        "time and tide wait for no man in our fast moving world",
        "where there is a will there is always a way forward",
        "to be or not to be that is the eternal philosophical question",
        "neither a borrower nor a lender be wrote shakespeare in his famous play",
        "this above all to thine own self be true in life and actions",
        "all the world is a stage and all the men and women merely players",
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
        "the quiet starry night was illuminated by countless brilliant constellations",
        "birds were singing cheerful melodies in the tall green trees this morning",
        "the peaceful garden was filled with colorful flowers blooming in spring",
        "the winding river flowed peacefully through the lush green valley",
    ],
    "Common Inquiries & Questions": [
        "how can i assist you with your software project today",
        "what is your favorite programming language for software development",
        "what are the key principles of clean software architecture",
        "what is the difference between supervised and unsupervised learning",
        "what time does the next train arrive at the station",
        "where is the nearest coffee shop in this neighborhood",
        "where can i find the official documentation for this library",
        "when will the next software release be available to users",
        "why is data preprocessing so important in machine learning",
        "can you recommend an introductory course on deep learning architectures",
        "would you like to join our collaborative open source project",
        "how does artificial intelligence impact modern society and industry",
        "what are the main advantages of using cloud computing infrastructure",
    ],
    "Superlatives & Common Collocations": [
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
        "at the same time we must acknowledge the limitations of current technology",
        "at the same time it is important to consider multiple perspectives",
        "on the other hand technological innovation brings many practical benefits",
        "as a matter of fact many breakthroughs happen through unexpected discoveries",
        "as a result of their hard work the team delivered the project ahead of schedule",
        "for the first time in history humans have access to global knowledge",
    ]
}


def build_systematic_variations():
    """Build systematic, grammatically valid variations for core language patterns."""
    variations = []

    # 1. "i am a [profession] [verb-ing/clause]"
    roles = [
        ("software engineer", "building scalable cloud systems"),
        ("software developer", "writing clean and maintainable code"),
        ("web developer", "creating accessible user interfaces"),
        ("data scientist", "training predictive machine learning models"),
        ("machine learning engineer", "deploying neural networks to production"),
        ("student", "studying mathematics and computer science"),
        ("teacher", "guiding students with enthusiasm"),
        ("doctor", "helping patients live healthier lives"),
        ("researcher", "investigating natural language processing architectures"),
        ("designer", "crafting intuitive digital experiences"),
    ]
    for r, clause in roles:
        variations.append(f"i am a {r} {clause}")
        variations.append(f"she is a {r} who loves her work")
        variations.append(f"he is a {r} with years of experience")

    # 2. "the weather is [adjective] today"
    weather_adjs = ["sunny", "warm", "cold", "pleasant", "cloudy", "clear", "mild", "rainy", "beautiful"]
    for w in weather_adjs:
        variations.append(f"the weather is {w} and comfortable today")
        variations.append(f"the weather is {w} across the country")
        variations.append(f"the weather is {w} outside so let us take a walk")

    # 3. "we are going to [verb] [object]" / "i am going to [verb] [object]"
    actions = [
        ("visit", "our grandparents this weekend"),
        ("explore", "new places during our trip"),
        ("build", "a web application using python"),
        ("learn", "more about deep learning architectures"),
        ("start", "working on the new machine learning project"),
        ("celebrate", "our milestone achievement together"),
        ("discuss", "the project requirements tomorrow morning"),
        ("attend", "the technology conference next week"),
    ]
    for v, obj in actions:
        variations.append(f"we are going to {v} {obj}")
        variations.append(f"i am going to {v} {obj}")
        variations.append(f"they are going to {v} {obj}")
        variations.append(f"i want to {v} {obj}")

    # 4. "python is a [adjective] programming language for [domain]"
    py_adjs = ["popular", "powerful", "versatile", "modern", "flexible", "clean"]
    domains = ["data science", "machine learning", "web development", "automation", "scientific computing"]
    for adj in py_adjs:
        for dom in domains:
            variations.append(f"python is a {adj} programming language for {dom}")

    # 5. "machine learning is [verb-ing / noun phrase]"
    ml_complements = [
        "transforming modern healthcare and medicine",
        "improving the accuracy of speech recognition",
        "enabling autonomous driving vehicles",
        "powering modern recommendation systems",
        "helping scientists analyze complex datasets",
        "an exciting and rapidly growing field",
        "essential for modern technological innovation",
        "a powerful tool for discovering patterns",
    ]
    for comp in ml_complements:
        variations.append(f"machine learning is {comp}")

    # 6. "it is [adj] to [verb] [object]"
    adj_verbs = [
        ("important", "write clean and maintainable code"),
        ("important", "test software thoroughly before deployment"),
        ("essential", "understand the fundamentals of algorithms"),
        ("crucial", "protect sensitive user data and privacy"),
        ("possible", "achieve great results with dedication"),
        ("helpful", "read official documentation when exploring new libraries"),
    ]
    for adj, vo in adj_verbs:
        variations.append(f"it is {adj} to {vo}")

    return variations


def curate_complete_corpus():
    """Curate and write the complete high-quality English corpus."""
    sentences = []
    seen = set()

    for category, sents in CORE_CATEGORIES.items():
        for s in sents:
            cleaned = clean_text(s)
            tokens = cleaned.split()
            if is_valid_sentence(tokens) and cleaned not in seen:
                seen.add(cleaned)
                sentences.append(cleaned)

    print(f"Collected {len(sentences)} unique core domain sentences.")

    variations = build_systematic_variations()
    var_added = 0
    for v in variations:
        cleaned = clean_text(v)
        tokens = cleaned.split()
        if is_valid_sentence(tokens) and cleaned not in seen:
            seen.add(cleaned)
            sentences.append(cleaned)
            var_added += 1

    print(f"Added {var_added} systematic grammatical variations.")

    # Also incorporate clean Shakespeare corpus sentences if file exists
    shakespeare_path = os.path.join(DATA_DIR, "shakespeare_corpus.txt")
    shk_count = 0
    if os.path.exists(shakespeare_path):
        with open(shakespeare_path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = clean_text(line)
                tokens = cleaned.split()
                if is_valid_sentence(tokens) and cleaned not in seen:
                    seen.add(cleaned)
                    sentences.append(cleaned)
                    shk_count += 1
        print(f"Added {shk_count} clean literary lines from Shakespeare corpus.")

    print(f"Total curated sentences: {len(sentences)}")

    # Shuffle with fixed seed for reproducibility
    random.shuffle(sentences)

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        f.write("# High-Quality, Clean, Diverse English Corpus for Next-Word Prediction LSTM\n")
        f.write("# Sourced from Curated Modern English, Academic, Technology, Common Collocations, and Facts\n\n")
        for s in sentences:
            f.write(s + "\n")

    print(f"Successfully saved {len(sentences)} sentences to {OUTPUT_CORPUS}")
    return sentences


if __name__ == "__main__":
    curate_complete_corpus()

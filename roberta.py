from transformers import pipeline
import argparse

# Load RoBERTa fine-tuned on SQuAD 2.0
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define a list of test cases with context and corresponding questions
# Wikis used:
# - https://en.wikipedia.org/wiki/Jedi
# - https://en.wikipedia.org/wiki/Frodo_Baggins
# - https://en.wikipedia.org/wiki/University_of_California,_Santa_Cruz
# (In-Domain)
test_cases_part_1 = [
    {
        "context": "Within the Star Wars galaxy, the Jedi are powerful guardians of order and justice, who, through intuition, rigorous training, and intensive self-discipline, are able to wield a supernatural power known as the Force, thus achieving the ability to move objects with the mind, perform incredible feats of strength, and connect to certain people's thoughts.",
        "question": "Who trains the Jedi?"  # The passage does not specify who trains the Jedi.
    },
    {
        "context": "Frodo is repeatedly wounded during the quest and becomes increasingly burdened by the Ring as it nears Mordor.",
        "question": "What object wounded Frodo?"  # The passage doesn’t specify a weapon, so RoBERTa might hallucinate an answer.
    },
    {
        "context": "Founded in 1965, UC Santa Cruz began with the intention to showcase progressive, cross-disciplinary undergraduate education, innovative teaching methods and contemporary architecture. The residential college system consists of ten small colleges that were established as a variation of the Oxbridge collegiate university system.",
        "question": "What is the main campus of UC Santa Cruz called?"  # The passage does not specify a name for the main campus.
    },
]
# (Edited)
test_cases_part_2 = [
    {
        "context": "Outside the Star Wars galaxy, the Jedi are weak guardians of order and justice, who, through skateboarding, rigorous surfing, and intensive sleeping, are able to wield a supernatural power known as the chipotle burrito, thus achieving the ability to eat a burrito with the mind, perform incredible feats of strength, and connect to certain people's thoughts.",
        "question": "Where are the Jedi from?"  # The passage does not mention how Jedi gain power.
    },
    {
        "context": "Frodo is repeatedly healed during the superbowl and becomes increasingly burdened by the Bracelet as it nears Santa Clara.",
        "question": "Where does Frodo go?"  # The passage does not specify who Frodo is.
    },
    {
        "context": "Founded in 2050, UC Santa Cruz was established with the goal of eliminating all forms of interdisciplinary learning, banning innovative teaching methods, and replacing contemporary architecture with medieval fortresses. The residential college system was dismantled in favor of a single megastructure designed to resemble an ancient castle, complete with moats and drawbridges.",
        "question": "When was UC Santa Cruz founded?" # The passage does not specify the goal of UC Santa Cruz.
    },
]
# (Out-of-Domain)
# Sources:
# - https://www.skateboarding.com/trending-news/skateshop-day-2025-shopkeepers-graphic-by-todd-bratrud
# - https://blox-fruits.fandom.com/f/p/4400000000000433022
# - https://www.thebrickfan.com/legoland-park-portal-40810-exclusive-revealed/
test_cases_part_3 = [
    {
        "context": "Although I unfortunately don't have the 2020 'Shopkeepers' graphic, I have all the others. And I'm stoked to get my hands on one of these gems this year as well. I've always been a fan of Bratrud's art, so it's really a no-brainer. Don't sleep on these, either! Each shop gets a limited run, so if you're a die-hard supporter, you know what to do.",
        "question": "Who is a fan of Bratrud's art?"  # This question shifts focus to a clear subject in the passage.
    },
    {
        "context": "skibidi gyatt rizz only in ohio duke dennis did you pray today livvy dunne rizzing up baby gronk sussy imposter pibby glitch in real life sigma alpha omega male grindset andrew tate goon cave freddy fazbear colleen ballinger smurf cat vs strawberry elephant blud dawg shmlawg ishowspeed a whole bunch of turbulence flabbergasted after that...can't even mew in oklahoma free robux glitch working 2025 we can go gyatt for gyatt is that a jojo reference ? oh my god wat da hail!!!! Admin uzoth caught doing thug shake on zioles before dragon rework its not gonna come out ? Oh really? I will edge on your face while mewing to femboy hooters, you can't stop the alpha now watermelon cat is my baby daddy, you're are my sunshine bro really said 💀 i wish I had a level 10 gyatt like you it boosts my aura auramaxxing heard of that ? blud thinks he's the main character ahh scene ! amogus when the imposter is sus 19 dollar fortnite card who wants it and yes I'm giving it way so share share shær shalabikariririrororoshubalabur oh so you're approaching me ?",
        "question": "What is the main topic of the passage?"  # Forces the model to summarize a nonsensical passage.
    },
    {
        "context": "LEGOLAND Japan has revealed a new exclusive set called the LEGOLAND Park Portal (40810). The set has 323 pieces and features the white entrance found at the parks worldwide as well as some of the themes that you can find at the parks. There’s also a child minifigure wearing the I <3 LEGOLAND shirt. The set should be available at the stores inside LEGOLAND, Discovery Centers, and possibly online soon.",
        "question": "How many pieces are in the LEGOLAND Park Portal set?"  # This makes the model retrieve specific information rather than infer something incorrect.
    }
]

# Mapping test sets
test_sets = {
    1: test_cases_part_1,
    2: test_cases_part_2,
    3: test_cases_part_3,
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--test_set", type=int, choices=[1, 2, 3], default=1, help="Select test set: 1 (In-Domain), 2 (Edited), 3 (Out-of-Domain)")
args = parser.parse_args()

# Get the selected test set
test_cases = test_sets[args.test_set]

# Iterate through test cases and evaluate RoBERTa's responses
for i, case in enumerate(test_cases, 1):
    print(f"Test Case {i}:")
    print(f"Question: {case['question']}")
    print(f"Context: {case['context']}")
    result = qa_pipeline(question=case["question"], context=case["context"])
    print(f"Answer: {result['answer']}\nConfidence: {result['score']:.4f}\n")
    print("-" * 50)

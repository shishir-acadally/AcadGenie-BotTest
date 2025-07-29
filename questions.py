from dotenv import load_dotenv
load_dotenv()
import os
import random
from motor.motor_asyncio import AsyncIOMotorClient
from bs4 import BeautifulSoup
import traceback

async def getQuestions(board, grade, subject):
    """
    Fetches questions based on the provided board, grade, subject, and chapter.
    
    Args:
        board (str): The educational board (e.g., 'CBSE', 'ICSE').
        grade (str): The grade or class (e.g., '10', '12').
        subject (str): The subject for which questions are needed (e.g., 'Math', 'Science').
        chapter (str): The specific chapter within the subject.
    
    Returns:
        list: A list of questions related to the specified criteria.
    """
    try:
        client = AsyncIOMotorClient(os.environ["db_uri"])
        db = client[os.environ["db_name"]]
        collection = db['QuestionCollection']

        query = {
            'board': board,
            'grade': grade,
            'subject': subject.lower(),
            'status': {'$in': ['verified', 'pushed']},
            'publication': 'LMS'
        }
        projection = {
            '_id': 0,
            'question': 1,
            'options': 1,
            'hint': 1,
            'solution': 1,
        }

        questions = await collection.find(query, projection).to_list(length=10)
        for que in questions:
            # set as correct or incorrect question
            what_to_set = random.randint(0, 1)
            if what_to_set == 0:
                que['is_correct'] = False
            else:
                que['is_correct'] = True

            which_wrong_option_to_choose = random.randint(0, 2)
            print(which_wrong_option_to_choose)
            if what_to_set == 0:
                opt_index =  [ind for ind, opt in enumerate(que.get('options', [])) if opt.get('is_correct') == False][which_wrong_option_to_choose]
            else:
                opt_index = [ind for ind, opt in enumerate(que.get('options', [])) if opt.get('is_correct') == True][0]

            que['correct_option'] = [ind for ind, opt in enumerate(que.get('options', [])) if opt.get('is_correct') == True][0]
            
            que['question'] = BeautifulSoup(que['question'][0]['content'], 'html.parser').get_text()
            if 'options' in que:
                que['options'] = [BeautifulSoup(opt['option'][0]['content'], 'html.parser').get_text() for opt in que['options']]
            if 'hint' in que:
                que['hint'] = BeautifulSoup(que['hint'][0]['content'], 'html.parser').get_text()
            if 'solution' in que:
                que['solution'] = BeautifulSoup(que['solution'][0]['content'], 'html.parser').get_text()
            que['chosen_option'] = que['options'][opt_index]

            
        return questions
    except Exception as e:
        print(f"An error occurred while fetching questions: ", type(e).__name__, str(e), traceback.format_exc())
        return []
    finally:
        client.close()
    
    
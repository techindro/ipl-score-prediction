import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_mock_ipl_data(num_rows=5000, output_path='data/ipl_dataset.csv'):
    np.random.seed(42)
    random.seed(42)

    teams = [
        'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
        'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
        'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 
        'Lucknow Super Giants'
    ]
    
    venues = [
        'M Chinnaswamy Stadium', 'Wankhede Stadium', 'Eden Gardens', 
        'MA Chidambaram Stadium', 'Arun Jaitley Stadium', 'Narendra Modi Stadium',
        'Rajiv Gandhi International Stadium', 'Punjab Cricket Association IS Bindra Stadium',
        'Sawai Mansingh Stadium', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
        'Maharashtra Cricket Association Stadium', 'Holkar Cricket Stadium', 'Barsapara Cricket Stadium'
    ]
    
    batsmen = [
        'MS Dhoni', 'Virat Kohli', 'Rohit Sharma', 'Shubman Gill', 'Suryakumar Yadav', 
        'KL Rahul', 'Ruturaj Gaikwad', 'Rishabh Pant', 'Shreyas Iyer', 'Sanju Samson', 
        'Hardik Pandya', 'Ishan Kishan', 'Faf du Plessis', 'David Warner', 'Glenn Maxwell', 
        'Nicholas Pooran', 'Heinrich Klaasen', 'Yashasvi Jaiswal', 'Rinku Singh', 'Shivam Dube', 
        'Travis Head', 'Sai Sudharsan', 'Tilak Varma', 'Rajat Patidar'
    ]
    
    bowlers = [
        'Jasprit Bumrah', 'Rashid Khan', 'Mohammed Shami', 'Trent Boult', 'Ravindra Jadeja', 
        'Yuzvendra Chahal', 'Kagiso Rabada', 'Mohammed Siraj', 'Arshdeep Singh', 
        'Matheesha Pathirana', 'Kuldeep Yadav', 'Ravi Bishnoi', 'Pat Cummins', 'Mitchell Starc', 
        'Sunil Narine', 'Varun Chakaravarthy', 'Harshal Patel', 'Bhuvneshwar Kumar', 
        'Avesh Khan', 'Sandeep Sharma', 'Mohit Sharma', 'T Natarajan', 'Axar Patel'
    ]
    
    data = []
    
    start_date = datetime(2026, 3, 20)
    
    for i in range(num_rows):
        mid = i // 120 + 1 # rough match id
        date = (start_date + timedelta(days=mid//2)).strftime('%Y-%m-%d')
        
        team1, team2 = random.sample(teams, 2)
        venue = random.choice(venues)
        
        batsman = random.choice(batsmen)
        non_striker = random.choice(batsmen)
        while non_striker == batsman:
            non_striker = random.choice(batsmen)
            
        bowler = random.choice(bowlers)
        
        overs = round(random.uniform(5.0, 19.5), 1)
        runs = int(overs * random.uniform(6.0, 10.0))
        wickets = random.randint(0, 9)
        
        runs_last_5 = int(random.uniform(30, 60))
        if runs_last_5 > runs: runs_last_5 = runs
        
        wickets_last_5 = random.randint(0, min(wickets, 5))
        
        striker = random.choice([0, 1])
        
        # Total score for the innings
        total = runs + int((20 - overs) * random.uniform(6.0, 12.0))
        
        row = {
            'date': date,
            'mid': mid,
            'bat_team': team1,
            'bowl_team': team2,
            'venue': venue,
            'batsman': batsman,
            'bowler': bowler,
            'non_striker': non_striker,
            'runs': runs,
            'wickets': wickets,
            'overs': overs,
            'runs_last_5': runs_last_5,
            'wickets_last_5': wickets_last_5,
            'striker': striker,
            'total': total
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of mock 2026 IPL data at '{output_path}'")

if __name__ == '__main__':
    generate_mock_ipl_data()

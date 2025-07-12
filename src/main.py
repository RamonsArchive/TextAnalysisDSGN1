import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os


def load_and_clean_data():
    """Load and clean the survey data"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "..", "responses.csv"))

    df.columns = df.columns.str.strip()

    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    return df


def analyze_swing_preference_optimized(response):
    """Optimized function to analyze swing preference with enhanced pattern matching"""
    if pd.isna(response) or not response.strip():
        return 'no_preference'
    
    response_lower = response.lower().strip()
    
    # Enhanced pattern matching for better accuracy
    inward_patterns = [
        r'swing.*in(?:ward)?(?:s|\b)', r'open.*in(?:ward)?(?:s|\b)', r'into.*(?:the\s+)?(?:stall|toilet|bathroom)',
        r'inward(?:s|\b)', r'swing.*inside', r'door.*(?:opens?\s+)?in(?:ward)?(?:s|\b)',
        r'comes?\s+in(?:ward)?(?:s|\b)', r'opening?\s+in(?:ward)?(?:s|\b)'
    ]
    
    outward_patterns = [
        r'swing.*out(?:ward)?(?:s|\b)', r'open.*out(?:ward)?(?:s|\b)', r'outward(?:s|\b)',
        r'swing.*outside', r'door.*(?:opens?\s+)?out(?:ward)?(?:s|\b)',
        r'comes?\s+out(?:ward)?(?:s|\b)', r'opening?\s+out(?:ward)?(?:s|\b)'
    ]
    
    # Detect swing direction mentions
    has_inward = any(re.search(pattern, response_lower) for pattern in inward_patterns)
    has_outward = any(re.search(pattern, response_lower) for pattern in outward_patterns)
    
    # Enhanced sentiment analysis with contextual clues
    strong_negative = [
        r'(?:don\'t|do\s+not|really\s+don\'t)\s+like', r'dislike', r'hate', r'can\'t\s+stand',
        r'really\s+(?:dislike|hate)', r'absolutely\s+(?:dislike|hate)', r'struggle.*(?:with|when)',
        r'difficult.*(?:to|when)', r'hard.*(?:to|when)', r'makes?\s+it\s+(?:difficult|hard)',
        r'nastier?', r'gross(?:er)?', r'(?:more\s+)?(?:cramped|crowded|tight)', r'no\s+room',
        r'leaves?\s+no\s+room', r'touch.*(?:things|stuff).*(?:rather\s+not|don\'t\s+want)',
        r'tend\s+to\s+touch', r'makes?\s+.*(?:cramped|tight|difficult)'
    ]
    
    moderate_negative = [
        r'not\s+(?:a\s+)?fan', r'not\s+great', r'not\s+ideal', r'could\s+be\s+better',
        r'issues?\s+with', r'problems?\s+with', r'annoying', r'frustrating',
        r'bothers?\s+me', r'wish.*(?:different|better|other)'
    ]
    
    positive_indicators = [
        r'(?:really\s+)?(?:like|love|enjoy)', r'good', r'fine', r'okay', r'no\s+problem',
        r'works?\s+(?:fine|well|good)', r'happy\s+with', r'satisfied\s+with',
        r'prefer.*(?:this|it)', r'better.*(?:this|it)', r'appreciate'
    ]
    
    preference_indicators = [
        r'prefer.*out(?:ward)?', r'better.*out(?:ward)?', r'should.*(?:swing|open).*out(?:ward)?',
        r'prefer.*in(?:ward)?', r'better.*in(?:ward)?', r'should.*(?:swing|open).*in(?:ward)?'
    ]
    
    # Contextual issue detection
    space_issues = [
        r'(?:no|little|not\s+enough|limited)\s+(?:room|space)', r'cramped', r'crowded',
        r'tight(?:\s+(?:space|fit))?', r'squeeze', r'squish', r'can\'t\s+(?:move|fit)'
    ]
    
    cleanliness_issues = [
        r'touch.*(?:things|stuff|walls|surfaces)', r'nasty', r'gross', r'dirty',
        r'rather\s+not\s+touch', r'don\'t\s+want\s+to\s+touch', r'hygiene', r'sanitary'
    ]
    
    # Analyze sentiment and context
    has_strong_negative = any(re.search(pattern, response_lower) for pattern in strong_negative)
    has_moderate_negative = any(re.search(pattern, response_lower) for pattern in moderate_negative)
    has_positive = any(re.search(pattern, response_lower) for pattern in positive_indicators)
    has_preference = any(re.search(pattern, response_lower) for pattern in preference_indicators)
    has_space_issues = any(re.search(pattern, response_lower) for pattern in space_issues)
    has_cleanliness_issues = any(re.search(pattern, response_lower) for pattern in cleanliness_issues)
    
    # Decision logic with enhanced context awareness
    if has_inward and (has_strong_negative or has_space_issues or has_cleanliness_issues):
        return 'inward_negative'
    elif has_outward and (has_strong_negative or has_moderate_negative):
        return 'outward_negative'
    elif has_outward and (has_positive or has_preference or re.search(r'prefer.*out', response_lower)):
        return 'outward_positive'
    elif has_inward and (has_positive or re.search(r'prefer.*in', response_lower)):
        return 'inward_positive'
    elif has_preference:
        if re.search(r'prefer.*out', response_lower):
            return 'outward_positive'
        elif re.search(r'prefer.*in', response_lower):
            return 'inward_positive'
    elif has_positive and (has_inward or has_outward):
        return 'neutral'
    elif 'no preference' in response_lower or 'don\'t mind' in response_lower:
        return 'no_preference'
    elif any(word in response_lower for word in ['fine', 'okay', 'good', 'works']):
        return 'neutral'
    else:
        return 'no_preference'


def analyze_privacy_gaps(df):
    """Analyze privacy concerns and gap-related issues"""
    print("\n" + "="*50)
    print("PRIVACY & GAP ANALYSIS")
    print("="*50)

    # Get gap-related responses from multiple columns
    gap_column = "How do you feel about the bathroom door gaps on the sides and on the top and bottom?"
    improvement_column = "What would you improve about bathroom doors?"
    security_column = "Do public restroom stall doors make you feel secure? What do you like or dislike?"
    want_column = "What do you want from a bathroom door?"
    
    # Combine all relevant columns for comprehensive gap analysis
    all_responses = []
    for col in [gap_column, improvement_column, security_column, want_column]:
        if col in df.columns:
            responses = df[col].dropna().str.lower()
            all_responses.extend(responses.tolist())
    
    all_responses_str = ' '.join(all_responses)

    gap_keywords = ['gap', 'crack', 'narrow', 'wide', 'privacy', 'see through', 'peek', 'cracks', 'gaps', 'come through', 'concern', 'invasive', 'vulnerable', 'secure', 'closure', 'seal', 'see through', 'look through', 'see me', 'covers', 'safety', 'security', 'narrow', 'no']
    gap_mentions = {}

    for keyword in gap_keywords:
        count = all_responses_str.count(keyword)
        gap_mentions[keyword] = count

    print("Gap-related keyword mentions across all relevant columns:")
    for keyword, count in sorted(gap_mentions.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {keyword}: {count}")

    # Analyze sentiment about gaps more comprehensively
    negative_gap_words = ['dislike', 'uncomfortable', 'hide', 'seal', 'narrower', 'weird', 'on the sides', 'reduce', 'less', 'no', 'lacking', 'invasive', 'awkward', 'vulnerable', 'anxious', 'concern', 'worried', 'insecure', 'don\'t like', 'hate', 'annoying', 'smaller', 'see me', 'look through', 'privacy', 'too big', 'big', 'see through']
    positive_gap_words = ['fine', 'okay', 'good', 'understand', 'no big deal', 'indifferent', 'acceptable']
    
    negative_count = sum(all_responses_str.count(word) for word in negative_gap_words)
    positive_count = sum(all_responses_str.count(word) for word in positive_gap_words)
    
    print(f"\nGap sentiment analysis:")
    print(f"  Negative sentiment mentions: {negative_count}")
    print(f"  Positive/neutral sentiment mentions: {positive_count}")
    
    # Privacy-specific analysis
    privacy_words = ['privacy', 'private', 'see me', 'see through', 'peek', 'look in', 'staring']
    privacy_concerns = sum(all_responses_str.count(word) for word in privacy_words)
    print(f"  Privacy concern mentions: {privacy_concerns}")
    
    return df


def analyze_door_swing_preferences(df):
    """Analyze door swing direction preferences with optimized sentiment analysis"""
    print("\n" + "="*50)
    print("DOOR SWING PREFERENCE ANALYSIS")
    print("="*50)
    
    swing_column = "How do you feel about how a bathroom door swings (how it opens and closes)? What do you like or dislike about it?"
    improvement_column = "What would you improve about bathroom doors?"
    
    # Combine swing-related responses
    swing_responses = []
    if swing_column in df.columns:
        swing_responses.extend(df[swing_column].dropna().tolist())
    if improvement_column in df.columns:
        improvement_responses = df[improvement_column].dropna()
        swing_related = improvement_responses[improvement_responses.str.lower().str.contains('swing|open|close|door', na=False)]
        swing_responses.extend(swing_related.tolist())
    
    # Analyze each response using optimized function
    swing_analysis = {
        'inward_negative': [],
        'inward_positive': [],
        'outward_negative': [],
        'outward_positive': [],
        'neutral': [],
        'no_preference': []
    }
    
    for response in swing_responses:
        result = analyze_swing_preference_optimized(response)
        swing_analysis[result].append(response)
    
    # Display results
    print("Swing preference analysis results:")
    for category, responses in swing_analysis.items():
        print(f"  {category}: {len(responses)}")
        if responses and len(responses) <= 3:  # Show examples for smaller categories
            for i, response in enumerate(responses[:2], 1):
                print(f"    {i}. {response[:100]}...")
    
    # Calculate key insights
    total_responses = len(swing_responses)
    inward_negative = len(swing_analysis['inward_negative'])
    outward_positive = len(swing_analysis['outward_positive'])
    
    print(f"\nKey insights:")
    print(f"  Total swing-related responses: {total_responses}")
    if total_responses > 0:
        print(f"  Negative about inward swing: {inward_negative} ({inward_negative/total_responses*100:.1f}%)")
        print(f"  Positive about outward swing: {outward_positive} ({outward_positive/total_responses*100:.1f}%)")
    
    # Add analysis results to dataframe
    if swing_column in df.columns:
        df['swing_preference_analysis'] = df[swing_column].apply(analyze_swing_preference_optimized)
    
    return df


def analyze_gender_differences(df):
    """Analyze differences between gender responses with enhanced emotional analysis"""
    print("\n" + "="*50)
    print("GENDER DIFFERENCE ANALYSIS")
    print("="*50)
    
    gender_col = "What's your gender?"
    security_col = "Do public restroom stall doors make you feel secure? What do you like or dislike?"
    avoidance_col = "Have you ever decided not to use a restroom because of the stall door? If so, why?"
    feeling_col = "How do the bathroom doors make you feel?"
    gaps_col = "How do you feel about the bathroom door gaps on the sides and on the top and bottom?"
    
    # Check if gender column exists
    if gender_col not in df.columns:
        print("Gender column not found in dataset")
        return df
    
    # Gender distribution
    gender_counts = df[gender_col].value_counts()
    print("Gender distribution:")
    print(gender_counts)
    
    # Enhanced emotional analysis by gender
    emotional_words = {
        'security': ['secure', 'safe', 'protected'],
        'insecurity': ['insecure', 'unsafe', 'vulnerable', 'anxious', 'uncomfortable'],
        'privacy': ['private', 'privacy', 'personal'],
        'exposure': ['exposed', 'seen', 'watched', 'stared'],
        'disgust': ['gross', 'dirty', 'nasty', 'disgusting'],
        'frustration': ['annoying', 'irritating', 'frustrated', 'struggle']
    }
    
    print("\nEmotional analysis by gender:")
    for gender in df[gender_col].unique():
        if pd.notna(gender):
            gender_df = df[df[gender_col] == gender]
            print(f"\n{gender}:")
            
            # Combine all emotional response columns
            all_responses = []
            for col in [security_col, feeling_col, gaps_col]:
                if col in df.columns:
                    responses = gender_df[col].dropna().str.lower()
                    all_responses.extend(responses.tolist())
            
            combined_text = ' '.join(all_responses)
            
            for emotion, words in emotional_words.items():
                count = sum(combined_text.count(word) for word in words)
                print(f"    {emotion}: {count}")

    # Avoidance behavior by gender with reasons
    if avoidance_col in df.columns:
        print("\nAvoidance behavior by gender:")
        for gender in df[gender_col].unique():
            if pd.notna(gender):
                gender_df = df[df[gender_col] == gender]
                avoidance_responses = gender_df[avoidance_col].dropna()
                
                # Count "Yes" responses and analyze reasons
                yes_responses = avoidance_responses[avoidance_responses.str.lower().str.contains('yes', na=False)]
                total_responses = len(avoidance_responses)
                
                if total_responses > 0:
                    avoidance_rate = (len(yes_responses) / total_responses) * 100
                    print(f"  {gender}: {len(yes_responses)}/{total_responses} ({avoidance_rate:.1f}%) have avoided restrooms")
                    
                    # Analyze reasons for avoidance
                    if len(yes_responses) > 0:
                        reasons = {
                            'broken_lock': sum(1 for r in yes_responses if 'broken' in r.lower() or 'lock' in r.lower()),
                            'privacy': sum(1 for r in yes_responses if 'privacy' in r.lower() or 'gap' in r.lower()),
                            'cleanliness': sum(1 for r in yes_responses if 'dirty' in r.lower() or 'clean' in r.lower()),
                            'security': sum(1 for r in yes_responses if 'secure' in r.lower() or 'unsafe' in r.lower())
                        }
                        
                        print(f"    Avoidance reasons:")
                        for reason, count in reasons.items():
                            if count > 0:
                                print(f"      {reason}: {count}")
    
    return df


def analyze_lock_security(df):
    """Analyze lock functionality and security concerns with enhanced sentiment analysis"""
    print("\n" + "="*50)
    print("LOCK SECURITY ANALYSIS")
    print("="*50)
    
    lock_col = "Are bathroom stall door locks sufficient? What do you like or dislike about them?"
    security_col = "Do public restroom stall doors make you feel secure? What do you like or dislike?"
    improvement_col = "What would you improve about bathroom doors?"
    
    # Combine all lock-related responses
    all_lock_responses = []
    for col in [lock_col, security_col, improvement_col]:
        if col in df.columns:
            responses = df[col].dropna().str.lower()
            all_lock_responses.extend(responses.tolist())
    
    combined_text = ' '.join(all_lock_responses)
    
    # Enhanced lock analysis
    lock_keywords = {
        'broken': combined_text.count('broken'),
        'secure': combined_text.count('secure'),
        'loose': combined_text.count('loose'),
        'tight': combined_text.count('tight'),
        'heavy duty': combined_text.count('heavy duty'),
        'flimsy': combined_text.count('flimsy'),
        'sturdy': combined_text.count('sturdy'),
        'weak': combined_text.count('weak'),
        'strong': combined_text.count('strong'),
        'reliable': combined_text.count('reliable')
    }
    
    print("Lock-related keyword mentions:")
    for keyword, count in sorted(lock_keywords.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {keyword}: {count}")
    
    # Lock problems analysis
    lock_problems = {
        'broken_lock': sum(1 for r in all_lock_responses if 'broken' in r and 'lock' in r),
        'loose_lock': sum(1 for r in all_lock_responses if 'loose' in r),
        'no_lock': sum(1 for r in all_lock_responses if 'no lock' in r),
        'doesnt_work': sum(1 for r in all_lock_responses if 'doesn\'t work' in r or 'not work' in r),
        'falls_off': sum(1 for r in all_lock_responses if 'fall' in r or 'drop' in r)
    }
    
    print(f"\nSpecific lock problems:")
    for problem, count in lock_problems.items():
        if count > 0:
            print(f"  {problem}: {count}")
    
    # Security satisfaction analysis
    if security_col in df.columns:
        security_responses = df[security_col].dropna().str.lower()
        
        positive_security = sum(1 for r in security_responses if any(word in r for word in ['yes', 'secure', 'safe', 'good', 'fine']))
        negative_security = sum(1 for r in security_responses if any(word in r for word in ['no', 'not secure', 'unsafe', 'vulnerable', 'gaps']))
        
        print(f"\nSecurity satisfaction:")
        print(f"  Positive responses: {positive_security}")
        print(f"  Negative responses: {negative_security}")
    
    return df


def create_visualizations(df):
    """Create enhanced visualizations for the analysis"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Gender distribution
    gender_col = "What's your gender?"
    if gender_col in df.columns:
        gender_counts = df[gender_col].value_counts()
        axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Gender Distribution')
    else:
        axes[0, 0].text(0.5, 0.5, 'Gender data\nnot available', ha='center', va='center')
        axes[0, 0].set_title('Gender Distribution')
    
    # 2. Security feelings by gender
    security_col = "Do public restroom stall doors make you feel secure? What do you like or dislike?"
    
    if gender_col in df.columns and security_col in df.columns:
        gender_security = {}
        for gender in df[gender_col].unique():
            if pd.notna(gender):
                gender_responses = df[df[gender_col] == gender][security_col].dropna().str.lower()
                secure_count = sum(1 for r in gender_responses if any(word in r for word in ['secure', 'safe', 'yes']))
                insecure_count = sum(1 for r in gender_responses if any(word in r for word in ['not secure', 'unsafe', 'no', 'vulnerable']))
                gender_security[gender] = {'secure': secure_count, 'insecure': insecure_count}
        
        if gender_security:
            genders = list(gender_security.keys())
            secure_counts = [gender_security[g]['secure'] for g in genders]
            insecure_counts = [gender_security[g]['insecure'] for g in genders]
            
            x = np.arange(len(genders))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, secure_counts, width, label='Secure', alpha=0.8)
            axes[0, 1].bar(x + width/2, insecure_counts, width, label='Insecure', alpha=0.8)
            axes[0, 1].set_xlabel('Gender')
            axes[0, 1].set_ylabel('Number of Responses')
            axes[0, 1].set_title('Security Feelings by Gender')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(genders)
            axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'Security/Gender data\nnot available', ha='center', va='center')
        axes[0, 1].set_title('Security Feelings by Gender')
    
    # 3. Avoidance behavior
    avoidance_col = "Have you ever decided not to use a restroom because of the stall door? If so, why?"
    if avoidance_col in df.columns:
        avoidance_responses = df[avoidance_col].dropna().str.lower()
        
        yes_avoid = sum(1 for r in avoidance_responses if 'yes' in r)
        no_avoid = sum(1 for r in avoidance_responses if 'no' in r)
        
        axes[0, 2].bar(['Yes', 'No'], [yes_avoid, no_avoid], color=['red', 'green'], alpha=0.7)
        axes[0, 2].set_title('Have You Avoided Restrooms Due to Door Issues?')
        axes[0, 2].set_ylabel('Number of Responses')
    else:
        axes[0, 2].text(0.5, 0.5, 'Avoidance data\nnot available', ha='center', va='center')
        axes[0, 2].set_title('Avoidance Behavior')
    
    # 4. Top improvement requests
    improvement_col = "What would you improve about bathroom doors?"
    if improvement_col in df.columns:
        improvement_responses = df[improvement_col].dropna().str.lower()
        
        improvements = {
            'Gaps/Cracks': sum(1 for r in improvement_responses if 'gap' in r or 'crack' in r),
            'Locks': sum(1 for r in improvement_responses if 'lock' in r),
            'Privacy': sum(1 for r in improvement_responses if 'privacy' in r),
            'Height/Size': sum(1 for r in improvement_responses if 'tall' in r or 'height' in r),
            'Cleanliness': sum(1 for r in improvement_responses if 'clean' in r or 'dirty' in r)
        }
        
        axes[1, 0].bar(improvements.keys(), improvements.values(), color='skyblue', alpha=0.8)
        axes[1, 0].set_title('Top Improvement Requests')
        axes[1, 0].set_ylabel('Number of Mentions')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'Improvement data\nnot available', ha='center', va='center')
        axes[1, 0].set_title('Top Improvement Requests')
    
    # 5. Swing preference analysis
    swing_col = "How do you feel about how a bathroom door swings (how it opens and closes)? What do you like or dislike about it?"
    if swing_col in df.columns:
        swing_responses = df[swing_col].dropna()
        
        # Use the optimized analysis function
        swing_results = [analyze_swing_preference_optimized(response) for response in swing_responses]
        result_counts = Counter(swing_results)
        
        # Group for better visualization
        swing_prefs = {
            'Negative about Inward': result_counts.get('inward_negative', 0),
            'Positive about Outward': result_counts.get('outward_positive', 0),
            'Neutral': result_counts.get('neutral', 0),
            'No Preference': result_counts.get('no_preference', 0)
        }
        
        axes[1, 1].bar(swing_prefs.keys(), swing_prefs.values(), color='lightcoral', alpha=0.8)
        axes[1, 1].set_title('Door Swing Preferences')
        axes[1, 1].set_ylabel('Number of Responses')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Swing data\nnot available', ha='center', va='center')
        axes[1, 1].set_title('Door Swing Preferences')
    
    # 6. Privacy concerns
    gap_col = "How do you feel about the bathroom door gaps on the sides and on the top and bottom?"
    if gap_col in df.columns:
        gap_responses = df[gap_col].dropna().str.lower()
        
        privacy_concerns = {
            'Gap Issues': sum(1 for r in gap_responses if 'gap' in r or 'crack' in r),
            'Privacy Loss': sum(1 for r in gap_responses if 'privacy' in r or 'see' in r),
            'Uncomfortable': sum(1 for r in gap_responses if 'uncomfortable' in r or 'dislike' in r),
            'Vulnerable': sum(1 for r in gap_responses if 'vulnerable' in r or 'invasive' in r)
        }
        
        axes[1, 2].bar(privacy_concerns.keys(), privacy_concerns.values(), color='orange', alpha=0.8)
        axes[1, 2].set_title('Privacy Concerns')
        axes[1, 2].set_ylabel('Number of Mentions')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, 'Gap data\nnot available', ha='center', va='center')
        axes[1, 2].set_title('Privacy Concerns')
    
    plt.tight_layout()
    plt.savefig('enhanced_bathroom_survey_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the enhanced analysis"""
    print("ENHANCED BATHROOM DOOR SURVEY ANALYSIS")
    print("=" * 60)
    
    try:
        df = load_and_clean_data()
        
        # Run all analyses
        df = analyze_privacy_gaps(df)
        df = analyze_door_swing_preferences(df)
        df = analyze_gender_differences(df)
        df = analyze_lock_security(df)
        
        # Create visualizations
        create_visualizations(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Enhanced analysis complete with optimized swing preference detection!")
        print("The analysis now includes:")
        print("- Comprehensive privacy and gap analysis")
        print("- Optimized door swing preference detection with advanced regex")
        print("- Gender-based emotional response analysis")
        print("- Lock security and functionality assessment")
        print("- Enhanced visualizations with error handling")
        
        return df
        
    except FileNotFoundError:
        print("Error: CSV file not found. Please ensure 'responses.csv' is in the correct path.")
        return None
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    df = main()
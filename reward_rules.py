"""reward_rules.py - Balanced reward system for EMPATH training"""

from dataclasses import dataclass, field
from typing import List, Set, Dict
import random
import re

@dataclass
class WordSets:
    """Smart word groupings for flexible matching"""
    # Emotional presence
    emotions: Set[str] = field(default_factory=lambda: {
        'warm', 'gentle', 'thoughtful', 'curious', 
        'concerned', 'interested', 'attentive', 'calm',
        'excited', 'sympathetic', 'understanding'
    })
    
    # Physical signals
    body: Set[str] = field(default_factory=lambda: {
        'leaning', 'sitting', 'standing', 'posture',
        'forward', 'back', 'straight', 'relaxed',
        'shoulders', 'head', 'hands', 'stance'
    })
    
    # Engagement markers
    engagement: Set[str] = field(default_factory=lambda: {
        'listening', 'nodding', 'focused', 'alert',
        'responsive', 'engaged', 'receptive', 'present',
        'attentive', 'interested', 'open'
    })

def extract_tags(text: str, tag: str) -> str:
    """Extract content between XML-style tags."""
    try:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    except:
        return ""

def flexible_word_match(text: str, word_sets: WordSets) -> float:
    """Smart but simple word matching"""
    words = set(text.lower().split())
    
    # Count matches from each set
    emotion_matches = len(words & word_sets.emotions)
    body_matches = len(words & word_sets.body)
    engagement_matches = len(words & word_sets.engagement)
    
    # Simple scoring
    score = 0.0
    if emotion_matches > 0: score += 0.4
    if body_matches > 0: score += 0.3
    if engagement_matches > 0: score += 0.3
    
    return min(score, 1.0)  # Cap at 1.0

def check_consistency(react: str, respond: str, reflect: str) -> float:
    """Simple consistency check across stages"""
    # Check lengths are reasonable
    if not all([react, respond, reflect]):
        return 0.0
        
    react_words = set(react.lower().split())
    respond_words = set(respond.lower().split())
    reflect_words = set(reflect.lower().split())
    
    # Look for some word overlap (shows connection)
    common_words = react_words & respond_words & reflect_words
    
    # Strict length checks with penalties
    react_len = len(react.split())
    respond_len = len(respond.split())
    reflect_len = len(reflect.split())
    
    # Base length score
    length_score = 0.5
    
    # Penalties for being too long
    if react_len > 20: length_score *= max(0.1, 1.0 - (react_len - 20) / 50)
    if respond_len > 150: length_score *= max(0.1, 1.0 - (respond_len - 150) / 200)
    if reflect_len > 100: length_score *= max(0.1, 1.0 - (reflect_len - 100) / 200)
    
    # Penalties for being too short
    if react_len < 10: length_score *= max(0.1, react_len / 10)
    if respond_len < 30: length_score *= max(0.1, respond_len / 30)
    if reflect_len < 20: length_score *= max(0.1, reflect_len / 20)
    
    # Score
    score = 0.0
    if len(common_words) >= 2: score += 0.5
    score += length_score  # Add length-based score
    
    return score

def format_reward_func(completions: List[str]) -> List[float]:
    """Check if response has all required tags in right order and properly nested"""
    scores = []
    for comp in completions:
        score = 0.0
        
        # Check for complete REACT section first
        react_pattern = r"<react>.*?</react>"
        if re.match(f"^{react_pattern}", comp.strip(), re.DOTALL):
            score += 0.2
            
            # Only check RESPOND if REACT was complete
            respond_pattern = r"<respond>.*?</respond>"
            if re.search(fr"{react_pattern}\s*{respond_pattern}", comp, re.DOTALL):
                score += 0.2
                
                # Only check REFLECT if both previous sections were complete and in order
                reflect_pattern = r"<reflect>.*?</reflect>"
                if re.search(fr"{react_pattern}\s*{respond_pattern}\s*{reflect_pattern}", comp, re.DOTALL):
                    score += 0.1
        
        scores.append(score)  # Max 0.5 as before, but more granular
    return scores

def tag_count_reward_func(completions: List[str]) -> List[float]:
    """Ensure exactly one of each tag pair in correct order"""
    scores = []
    for r in completions:
        score = 0.0
        # Check pairs are matched
        if r.count("<react>") == 1 and r.count("</react>") == 1: score += 0.1
        if r.count("<respond>") == 1 and r.count("</respond>") == 1: score += 0.1
        if r.count("<reflect>") == 1 and r.count("</reflect>") == 1: score += 0.1
        
        # Extra points for correct ordering
        if (r.find("<react>") < r.find("</react>") < 
            r.find("<respond>") < r.find("</respond>") < 
            r.find("<reflect>") < r.find("</reflect>")):
            score += 0.3
            
        scores.append(score)
    return scores

def format_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """Reward proper formatting and tag structure"""
    try:
        if not isinstance(completions, (list, tuple)) or not completions:
            return [0.0]
            
        scores = []
        for comp in completions:
            try:
                # Base format check (0.5 max)
                format_score = format_reward_func([comp])[0]
                
                # Tag count and order check (0.5 max)
                tag_score = tag_count_reward_func([comp])[0]
                
                total = format_score + tag_score
                scores.append(total)  # Max 1.0
                
            except Exception as e:
                print(f"\n❌ Format error: {str(e)}")
                scores.append(0.0)
                
        return scores
    except Exception as e:
        print(f"\n❌ Critical format error: {str(e)}")
        return [0.0] * (len(completions) if completions else 0)

def content_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """Reward quality of content within tags"""
    try:
        if not isinstance(completions, (list, tuple)) or not completions:
            return [0.0]
            
        word_sets = WordSets()
        scores = []
        
        for comp in completions:
            try:
                react = extract_tags(comp, "react")
                respond = extract_tags(comp, "respond")
                reflect = extract_tags(comp, "reflect")
                
                # Score each section's content with length awareness
                react_score = flexible_word_match(react, word_sets) * 0.4
                if len(react.split()) > 20:  # Penalize verbose reactions
                    react_score *= 0.5
                    
                # Response scoring
                respond_words = len(respond.split())
                if 30 <= respond_words <= 150:
                    respond_score = 0.3
                else:
                    respond_score = 0.3 * max(0.1, 1.0 - abs(respond_words - 90) / 200)
                    
                # Reflection scoring
                reflect_words = len(reflect.split())
                if 20 <= reflect_words <= 100:
                    reflect_score = 0.3
                else:
                    reflect_score = 0.3 * max(0.1, 1.0 - abs(reflect_words - 60) / 150)
                
                total = react_score + respond_score + reflect_score
                scores.append(total)  # Max 1.0
                
            except Exception as e:
                print(f"\n❌ Content error: {str(e)}")
                scores.append(0.0)
                
        return scores
    except Exception as e:
        print(f"\n❌ Critical content error: {str(e)}")
        return [0.0] * (len(completions) if completions else 0)

def consistency_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """Reward consistency and flow between sections"""
    try:
        if not isinstance(completions, (list, tuple)) or not completions:
            return [0.0]
            
        scores = []
        for comp in completions:
            try:
                react = extract_tags(comp, "react")
                respond = extract_tags(comp, "respond")
                reflect = extract_tags(comp, "reflect")
                
                # Check cross-section consistency
                consistency = check_consistency(react, respond, reflect)
                scores.append(consistency)  # Max 1.0
                
            except Exception as e:
                print(f"\n❌ Consistency error: {str(e)}")
                scores.append(0.0)
                
        return scores
    except Exception as e:
        print(f"\n❌ Critical consistency error: {str(e)}")
        return [0.0] * (len(completions) if completions else 0)

# Keep the original as a combined version for reference
def reward_func(prompts=None, completions=None, **kwargs) -> List[float]:
    """Combined reward function (for reference/backup)"""
    try:
        if not isinstance(completions, (list, tuple)) or not completions:
            return [0.0]
            
        format_scores = format_reward(completions=completions)
        content_scores = content_reward(completions=completions)
        consistency_scores = consistency_reward(completions=completions)
        
        # Combine with equal weights
        return [
            (f + c + con) / 3.0 
            for f, c, con in zip(format_scores, content_scores, consistency_scores)
        ]
        
    except Exception as e:
        print(f"\n❌ Critical combined error: {str(e)}")
        return [0.0] * (len(completions) if completions else 0) 
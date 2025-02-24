import json
from collections import defaultdict
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich import print as rprint

@dataclass
class DialogStats:
    total_turns: int
    avg_user_length: float
    avg_vh_length: float
    topics: List[str]
    emotions: List[str]
    has_resolution: bool

def extract_topics(dialog: str) -> List[str]:
    """Extract main topics from dialog using keyword matching."""
    topics = []
    topic_keywords = {
        'work': ['work', 'job', 'career', 'project', 'deadline', 'presentation'],
        'school': ['school', 'study', 'class', 'assignment', 'homework'],
        'stress': ['stress', 'overwhelm', 'anxiety', 'pressure'],
        'time management': ['time', 'schedule', 'deadline', 'prioritize'],
        'relationships': ['team', 'group', 'colleague', 'friend'],
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in dialog.lower() for keyword in keywords):
            topics.append(topic)
    return list(set(topics))

def extract_emotions(dialog: str) -> List[str]:
    """Extract emotions from dialog using keyword matching."""
    emotions = []
    emotion_keywords = {
        'overwhelmed': ['overwhelm', 'stress', 'pressure'],
        'anxious': ['anxious', 'worried', 'nervous'],
        'frustrated': ['frustrat', 'stuck', 'difficult'],
        'hopeful': ['hope', 'better', 'help', 'thank'],
    }
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in dialog.lower() for keyword in keywords):
            emotions.append(emotion)
    return list(set(emotions))

def has_positive_resolution(dialog: str) -> bool:
    """Check if dialog ends with a positive resolution."""
    positive_indicators = ['thank', 'help', 'great', 'good idea', 'better']
    last_messages = dialog.split('\n')[-4:]  # Look at last few messages
    return any(indicator in ' '.join(last_messages).lower() for indicator in positive_indicators)

def analyze_dialog(dialog: str) -> DialogStats:
    """Analyze a single dialog and return statistics."""
    turns = dialog.count('User:')
    
    # Calculate average lengths
    user_messages = re.findall(r'User:(.*?)(?=Virtual Human:|$)', dialog, re.DOTALL)
    vh_messages = re.findall(r'Virtual Human:(.*?)(?=User:|$)', dialog, re.DOTALL)
    
    avg_user_len = sum(len(msg.strip()) for msg in user_messages) / len(user_messages)
    avg_vh_len = sum(len(msg.strip()) for msg in vh_messages) / len(vh_messages)
    
    return DialogStats(
        total_turns=turns,
        avg_user_length=avg_user_len,
        avg_vh_length=avg_vh_len,
        topics=extract_topics(dialog),
        emotions=extract_emotions(dialog),
        has_resolution=has_positive_resolution(dialog)
    )

def analyze_all_dialogs(file_path: str) -> Dict:
    """Analyze all dialogs and return comprehensive statistics."""
    with open(file_path, 'r') as f:
        dialogs = json.load(f)
    
    all_stats = []
    topic_counts = defaultdict(int)
    emotion_counts = defaultdict(int)
    resolution_count = 0
    
    for dialog in dialogs:
        stats = analyze_dialog(dialog['dialogue'])
        all_stats.append(stats)
        
        for topic in stats.topics:
            topic_counts[topic] += 1
        for emotion in stats.emotions:
            emotion_counts[emotion] += 1
        if stats.has_resolution:
            resolution_count += 1
    
    return {
        'total_dialogs': len(dialogs),
        'avg_turns': sum(s.total_turns for s in all_stats) / len(all_stats),
        'avg_user_length': sum(s.avg_user_length for s in all_stats) / len(all_stats),
        'avg_vh_length': sum(s.avg_vh_length for s in all_stats) / len(all_stats),
        'topic_distribution': dict(topic_counts),
        'emotion_distribution': dict(emotion_counts),
        'resolution_rate': resolution_count / len(dialogs)
    }

def print_analysis_report(stats: Dict):
    """Print a beautiful analysis report using rich."""
    console = Console()
    
    console.print("\n[bold magenta]📊 Dialog Analysis Report[/bold magenta]\n")
    
    # General Statistics
    console.print("[bold cyan]General Statistics:[/bold cyan]")
    general_table = Table(show_header=True, header_style="bold blue")
    general_table.add_column("Metric")
    general_table.add_column("Value")
    
    general_table.add_row("Total Dialogs", str(stats['total_dialogs']))
    general_table.add_row("Average Turns", f"{stats['avg_turns']:.1f}")
    general_table.add_row("Average User Message Length", f"{stats['avg_user_length']:.1f}")
    general_table.add_row("Average VH Message Length", f"{stats['avg_vh_length']:.1f}")
    general_table.add_row("Resolution Rate", f"{stats['resolution_rate']*100:.1f}%")
    
    console.print(general_table)
    
    # Topic Distribution
    console.print("\n[bold cyan]Topic Distribution:[/bold cyan]")
    topic_table = Table(show_header=True, header_style="bold blue")
    topic_table.add_column("Topic")
    topic_table.add_column("Count")
    topic_table.add_column("Percentage")
    
    for topic, count in sorted(stats['topic_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_dialogs']) * 100
        topic_table.add_row(topic, str(count), f"{percentage:.1f}%")
    
    console.print(topic_table)
    
    # Emotion Distribution
    console.print("\n[bold cyan]Emotion Distribution:[/bold cyan]")
    emotion_table = Table(show_header=True, header_style="bold blue")
    emotion_table.add_column("Emotion")
    emotion_table.add_column("Count")
    emotion_table.add_column("Percentage")
    
    for emotion, count in sorted(stats['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_dialogs']) * 100
        emotion_table.add_row(emotion, str(count), f"{percentage:.1f}%")
    
    console.print(emotion_table)

def main():
    stats = analyze_all_dialogs('seed_dialogues.json')
    print_analysis_report(stats)

if __name__ == "__main__":
    main() 
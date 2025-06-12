import os
import configparser
import datetime as dt
import time
import logging
import praw
import pandas as pd
from typing import List, Dict, Optional
import json

class RedditDataCollector:
    """
    Optimized Reddit data collector with robust error handling,
    progress tracking, and resume functionality.
    """
    
    def __init__(self, config_path: str = 'config.ini'):
        self.setup_logging()
        self.reddit = self._initialize_reddit(config_path)
        self.stats = {'total_comments': 0, 'total_submissions': 0, 'errors': 0}
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_reddit(self, config_path: str) -> praw.Reddit:
        """Initialize Reddit API connection"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        return praw.Reddit(
            client_id=config['reddit']['client_id'],
            client_secret=config['reddit']['client_secret'],
            user_agent=config['reddit']['user_agent'],
            timeout=30  # Add timeout for better error handling
        )
    
    def save_progress(self, progress_data: Dict, filepath: str = 'data/progress.json'):
        """Save collection progress to resume later"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, default=str)
    
    def load_progress(self, filepath: str = 'data/progress.json') -> Optional[Dict]:
        """Load previous collection progress"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def save_batch_to_csv(self, comments_batch: List[Dict], output_path: str, 
                         is_first_batch: bool = False) -> None:
        """Optimized batch saving with error handling"""
        try:
            if not comments_batch:
                return
                
            df_batch = pd.DataFrame(comments_batch)
            mode = 'w' if is_first_batch else 'a'
            header = is_first_batch
            
            # Ensure consistent data types
            df_batch['created_utc'] = pd.to_datetime(df_batch['created_utc'])
            df_batch['comment_score'] = pd.to_numeric(df_batch['comment_score'], errors='coerce')
            
            df_batch.to_csv(output_path, mode=mode, header=header, index=False)
            self.logger.info(f"Saved batch of {len(comments_batch)} comments")
            
        except Exception as e:
            self.logger.error(f"Error saving batch: {e}")
            # Save to backup file
            backup_path = output_path.replace('.csv', '_backup.csv')
            df_batch.to_csv(backup_path, mode='a', header=False, index=False)
    
    def process_comment(self, comment, sub_name: str, submission) -> Optional[Dict]:
        """Process individual comment with validation"""
        try:
            # Enhanced validation
            if not hasattr(comment, 'body') or not comment.body:
                return None
                
            body = comment.body.strip()
            if body in ('[deleted]', '[removed]', '', None) or len(body) < 10:
                return None
            
            # Additional quality filters
            if len(body) > 10000:  # Skip extremely long comments (likely spam)
                return None
                
            return {
                'subreddit': sub_name,
                'comment_id': comment.id,
                'comment_body': body,
                'created_utc': dt.datetime.fromtimestamp(comment.created_utc, tz=dt.timezone.utc),
                'comment_score': getattr(comment, 'score', 0),
                'submission_id': submission.id,
                'submission_title': submission.title[:500],  # Limit title length
                'submission_score': getattr(submission, 'score', 0),
                'submission_num_comments': getattr(submission, 'num_comments', 0)
            }
        except Exception as e:
            self.logger.warning(f"Error processing comment {getattr(comment, 'id', 'unknown')}: {e}")
            return None
    
    def adaptive_rate_limiting(self, submission_count: int, error_count: int) -> float:
        """Adaptive rate limiting based on performance"""
        base_delay = 1.0
        
        # Increase delay if encountering errors
        if error_count > 5:
            base_delay *= 2
        elif error_count > 10:
            base_delay *= 3
            
        # Adjust based on submission processing rate
        if submission_count % 100 == 0 and submission_count > 0:
            base_delay *= 0.9  # Slightly decrease delay if going well
            
        return max(0.5, min(base_delay, 5.0))  # Keep between 0.5 and 5 seconds
    
    def collect_subreddit_data(self, sub_name: str, start_date: dt.datetime, 
                              end_date: dt.datetime, submission_limit: int = 1000) -> int:
        """Collect data from a single subreddit with enhanced error handling"""
        self.logger.info(f"Starting collection for r/{sub_name}")
        
        subreddit = self.reddit.subreddit(sub_name)
        batch_comments = []
        submission_count = 0
        comment_count = 0
        error_count = 0
        
        # Try multiple collection strategies
        strategies = [
            ('timestamp', f"timestamp:{int(start_date.timestamp())}..{int(end_date.timestamp())}"),
            ('recent_hot', None),
            ('recent_top', None)
        ]
        
        for strategy_name, query in strategies:
            self.logger.info(f"Trying strategy: {strategy_name}")
            
            try:
                if strategy_name == 'timestamp':
                    submissions = subreddit.search(query, sort='new', syntax='lucene', limit=submission_limit)
                elif strategy_name == 'recent_hot':
                    submissions = subreddit.hot(limit=submission_limit//2)
                else:  # recent_top
                    submissions = subreddit.top(time_filter='month', limit=submission_limit//2)
                
                for submission in submissions:
                    submission_count += 1
                    
                    # Filter by date for non-timestamp strategies
                    submission_date = dt.datetime.fromtimestamp(submission.created_utc, tz=dt.timezone.utc)
                    if not (start_date <= submission_date <= end_date):
                        continue
                    
                    try:
                        # Progressive comment loading with timeout
                        submission.comments.replace_more(limit=3, threshold=10)
                        
                        # Adaptive rate limiting
                        delay = self.adaptive_rate_limiting(submission_count, error_count)
                        time.sleep(delay)
                        
                        for comment in submission.comments.list()[:100]:  # Limit comments per post
                            comment_data = self.process_comment(comment, sub_name, submission)
                            if comment_data:
                                batch_comments.append(comment_data)
                                comment_count += 1
                        
                        # Save batch periodically
                        if len(batch_comments) >= 500:  # Smaller batches for better memory management
                            yield batch_comments
                            batch_comments = []
                            
                    except Exception as e:
                        error_count += 1
                        self.logger.warning(f"Error processing submission {submission.id}: {e}")
                        
                        if error_count > 20:  # Stop if too many errors
                            self.logger.error("Too many errors encountered, stopping collection")
                            break
                        continue
                    
                    # Progress reporting
                    if submission_count % 25 == 0:
                        self.logger.info(f"r/{sub_name}: {submission_count} submissions, {comment_count} comments, {error_count} errors")
                
                # If we got enough data from this strategy, break
                if comment_count > 100:
                    break
                    
            except Exception as e:
                self.logger.error(f"Strategy {strategy_name} failed: {e}")
                continue
        
        # Return remaining comments
        if batch_comments:
            yield batch_comments
            
        self.stats['total_comments'] += comment_count
        self.stats['total_submissions'] += submission_count
        self.stats['errors'] += error_count
        
        self.logger.info(f"Finished r/{sub_name}: {comment_count} comments from {submission_count} submissions")
        return comment_count
    
    def collect_data(self, subreddits: List[str] = None, days_back: int = 180) -> None:
        """Main data collection method with resume capability"""
        if subreddits is None:
            subreddits = ['technology', 'startups']
        
        # Define time window
        end_date = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        start_date = end_date - dt.timedelta(days=days_back)
        
        self.logger.info(f"Collecting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        output_path = 'data/raw/combined_comments.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check for previous progress
        progress = self.load_progress()
        completed_subreddits = progress.get('completed_subreddits', []) if progress else []
        
        first_batch = not os.path.exists(output_path)
        
        for sub_name in subreddits:
            if sub_name in completed_subreddits:
                self.logger.info(f"Skipping r/{sub_name} (already completed)")
                continue
            
            try:
                for batch_comments in self.collect_subreddit_data(sub_name, start_date, end_date):
                    self.save_batch_to_csv(batch_comments, output_path, first_batch)
                    first_batch = False
                
                # Mark subreddit as completed
                completed_subreddits.append(sub_name)
                self.save_progress({
                    'completed_subreddits': completed_subreddits,
                    'last_updated': dt.datetime.now(),
                    'stats': self.stats
                })
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from r/{sub_name}: {e}")
                continue
        
        # Final statistics
        self.logger.info(f"Collection complete! Total stats: {self.stats}")
        
        # Validate final dataset
        self.validate_dataset(output_path)
    
    def validate_dataset(self, filepath: str) -> None:
        """Validate the collected dataset"""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Dataset validation:")
            self.logger.info(f"  Total rows: {len(df)}")
            self.logger.info(f"  Subreddits: {df['subreddit'].value_counts().to_dict()}")
            self.logger.info(f"  Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
            self.logger.info(f"  Missing values: {df.isnull().sum().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")


def main():
    """Main execution function"""
    collector = RedditDataCollector()
    
    # Collect data with error recovery
    try:
        collector.collect_data(
            subreddits=['technology', 'startups'],
            days_back=180
        )
    except KeyboardInterrupt:
        collector.logger.info("Collection interrupted by user")
    except Exception as e:
        collector.logger.error(f"Collection failed: {e}")
        

if __name__ == '__main__':
    main()

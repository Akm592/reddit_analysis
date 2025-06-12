import os
import pandas as pd
import nltk
import re
import logging
from typing import List, Set, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import multiprocessing as mp
from functools import partial
import numpy as np

class TextPreprocessor:
    """
    Advanced text preprocessor with robust NLTK handling and fallback tokenization.
    """
    
    def __init__(self, custom_stopwords: Optional[List[str]] = None):
        self.setup_logging()
        self.setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._create_stopword_set(custom_stopwords)
        self.stats = {'processed': 0, 'empty_after_processing': 0, 'errors': 0}
        self.nltk_available = self._test_nltk_functionality()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_nltk(self):
        """Download required NLTK data with comprehensive resource list"""
        nltk_downloads = [
            'punkt',           # Original punkt tokenizer
            'punkt_tab',       # New punkt tokenizer (required in recent NLTK versions)
            'stopwords', 
            'wordnet', 
            'omw-1.4', 
            'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            try:
                self.logger.info(f"Downloading NLTK resource: {item}")
                nltk.download(item, quiet=True)
                self.logger.info(f"Successfully downloaded: {item}")
            except Exception as e:
                self.logger.warning(f"Failed to download {item}: {e}")
                # Try alternative download method
                try:
                    nltk.download(item, quiet=False)  # Verbose mode for debugging
                except Exception as e2:
                    self.logger.error(f"Alternative download also failed for {item}: {e2}")
    
    def _test_nltk_functionality(self) -> bool:
        """Test if NLTK tokenization is working properly"""
        try:
            test_text = "This is a test sentence."
            tokens = word_tokenize(test_text)
            self.logger.info("NLTK tokenization is working properly")
            return True
        except Exception as e:
            self.logger.warning(f"NLTK tokenization failed: {e}")
            self.logger.info("Will use fallback tokenization method")
            return False
    
    def fallback_tokenize(self, text: str) -> List[str]:
        """Fallback tokenization method when NLTK fails"""
        # Simple regex-based tokenization
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def safe_tokenize(self, text: str) -> List[str]:
        """Safe tokenization with fallback"""
        if self.nltk_available:
            try:
                return word_tokenize(text)
            except Exception as e:
                self.logger.warning(f"NLTK tokenization failed for text, using fallback: {e}")
                return self.fallback_tokenize(text)
        else:
            return self.fallback_tokenize(text)
    
    def _create_stopword_set(self, custom_stopwords: Optional[List[str]] = None) -> Set[str]:
        """Create comprehensive stopword set with fallback"""
        try:
            # Try to get NLTK stopwords
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.warning(f"Failed to load NLTK stopwords: {e}")
            # Fallback to basic English stopwords
            stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Reddit-specific stopwords
        reddit_stopwords = {
            'reddit', 'subreddit', 'post', 'comment', 'thread', 'op', 'edit', 'update',
            'tldr', 'tl', 'dr', 'imho', 'imo', 'btw', 'fyi', 'lol', 'lmao', 'wtf',
            'tbh', 'ngl', 'omg', 'jk', 'smh', 'fml', 'idk', 'irl', 'til', 'eli5',
            'ama', 'dae', 'iirc', 'afaik', 'cmv', 'psa', 'nsfw', 'sfw'
        }
        
        # Tech/startup specific stopwords
        tech_stopwords = {
            'tech', 'technology', 'startup', 'company', 'business', 'product', 
            'service', 'app', 'software', 'platform', 'system', 'tool', 'solution'
        }
        
        # Combine all stopwords
        stop_words.update(reddit_stopwords)
        stop_words.update(tech_stopwords)
        
        if custom_stopwords:
            stop_words.update(set(word.lower() for word in custom_stopwords))
        
        return stop_words
    
    def safe_lemmatize(self, word: str) -> str:
        """Safe lemmatization with fallback"""
        try:
            return self.lemmatizer.lemmatize(word)
        except Exception as e:
            self.logger.warning(f"Lemmatization failed for word '{word}': {e}")
            return word  # Return original word if lemmatization fails
    
    def preprocess_text_advanced(self, text: str) -> str:
        """
        Advanced text preprocessing with robust error handling.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # 1. Initial cleaning
            text = text.strip()
            
            # 2. Remove Reddit-specific formatting
            text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
            text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold formatting
            text = re.sub(r'\*(.+?)\*', r'\1', text)  # Remove italic formatting
            text = re.sub(r'~~(.+?)~~', r'\1', text)  # Remove strikethrough
            text = re.sub(r'\^(\w+)', r'\1', text)  # Remove superscript
            
            # 3. Remove URLs and links
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # 4. Remove emails
            text = re.sub(r'\S+@\S+', '', text)
            
            # 5. Remove special characters but keep apostrophes temporarily
            text = re.sub(r"[^a-zA-Z\s']", ' ', text)
            
            # 6. Handle contractions before removing apostrophes
            contractions = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
                "'m": " am", "let's": "let us", "that's": "that is",
                "there's": "there is", "here's": "here is", "what's": "what is",
                "where's": "where is", "how's": "how is", "it's": "it is"
            }
            
            for contraction, expansion in contractions.items():
                text = re.sub(f"\\b\\w+{contraction}\\b", lambda m: m.group().replace(contraction, expansion), text, flags=re.IGNORECASE)
            
            # 7. Remove remaining apostrophes
            text = re.sub(r"'", '', text)
            
            # 8. Convert to lowercase
            text = text.lower()
            
            # 9. Safe tokenization
            tokens = self.safe_tokenize(text)
            
            # 10. Filter tokens
            filtered_tokens = []
            for token in tokens:
                # Skip if too short or too long
                if len(token) < 3 or len(token) > 20:
                    continue
                # Skip if it's a stopword
                if token in self.stop_words:
                    continue
                # Skip if it's all digits
                if token.isdigit():
                    continue
                # Skip if it's mostly non-alphabetic
                if sum(c.isalpha() for c in token) / len(token) < 0.5:
                    continue
                
                filtered_tokens.append(token)
            
            # 11. Safe lemmatization
            lemmatized_tokens = [self.safe_lemmatize(token) for token in filtered_tokens]
            
            # 12. Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token in lemmatized_tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)
            
            # 13. Join and clean final text
            result = ' '.join(unique_tokens)
            result = re.sub(r'\s+', ' ', result).strip()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error preprocessing text: {e}")
            self.stats['errors'] += 1
            return ""
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts with error handling"""
        results = []
        for text in texts:
            processed = self.preprocess_text_advanced(text)
            results.append(processed)
            self.stats['processed'] += 1
            if not processed:
                self.stats['empty_after_processing'] += 1
        return results
    
    def validate_preprocessing_results(self, df: pd.DataFrame) -> dict:
        """Validate preprocessing results and return statistics with safe division"""
        validation_stats = {
            'total_rows': len(df),
            'empty_processed': df['processed_text'].str.strip().eq('').sum(),
            'avg_original_length': df['comment_body'].str.len().mean() if len(df) > 0 else 0,
            'avg_processed_length': df['processed_text'].str.len().mean() if len(df) > 0 else 0,
            'unique_words_original': len(set(' '.join(df['comment_body'].fillna('')).split())),
            'unique_words_processed': len(set(' '.join(df['processed_text'].fillna('')).split()))
        }
        
        # Safe division check for potential issues
        if validation_stats['total_rows'] > 0:
            empty_ratio = validation_stats['empty_processed'] / validation_stats['total_rows']
            if empty_ratio > 0.5:
                self.logger.warning(f"Warning: {empty_ratio:.1%} of texts became empty after preprocessing")
        
        return validation_stats
    
    def preprocess_dataframe(self, df: pd.DataFrame, batch_size: int = 1000, 
                           use_multiprocessing: bool = False) -> pd.DataFrame:  # Disabled MP by default
        """
        Process entire dataframe with batch processing
        """
        self.logger.info(f"Starting preprocessing of {len(df)} comments...")
        
        # Clean the dataframe first
        df = df.copy()
        df.dropna(subset=['comment_body'], inplace=True)
        df = df[df['comment_body'].str.strip() != '']
        
        self.logger.info(f"After cleaning: {len(df)} comments to process")
        
        if len(df) == 0:
            self.logger.warning("No comments to process after cleaning!")
            df['processed_text'] = []
            return df
        
        # Prepare for batch processing
        processed_texts = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        # Single-threaded processing to avoid NLTK issues
        for i in range(0, len(df), batch_size):
            batch_texts = df['comment_body'].iloc[i:i+batch_size].tolist()
            batch_results = self.process_batch(batch_texts)
            processed_texts.extend(batch_results)
            
            # Progress reporting
            current_batch = (i // batch_size) + 1
            if current_batch % 5 == 0:
                self.logger.info(f"Processed batch {current_batch}/{total_batches}")
        
        # Add processed text to dataframe
        df['processed_text'] = processed_texts
        
        # Remove rows where processed text is empty
        original_len = len(df)
        df = df[df['processed_text'].str.strip() != '']
        
        self.logger.info(f"Preprocessing complete. Removed {original_len - len(df)} empty results.")
        self.logger.info(f"Final dataset: {len(df)} rows")
        return df


def main():
    """
    Enhanced main function with comprehensive error handling
    """
    print("=" * 60)
    print("Fixed Reddit Text Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    custom_stopwords = ['like', 'really', 'actually', 'basically', 'literally']
    preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)
    
    # File paths
    input_path = 'data/raw/combined_comments.csv'
    output_path = 'data/processed/cleaned_comments.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load data
        preprocessor.logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        preprocessor.logger.info(f"Loaded {len(df)} rows")
        
        if len(df) == 0:
            preprocessor.logger.error("No data found in input file!")
            return
        
        # Preprocess data
        df_processed = preprocessor.preprocess_dataframe(
            df, 
            batch_size=500,  # Smaller batches for better error handling
            use_multiprocessing=False  # Disabled to avoid NLTK issues
        )
        
        if len(df_processed) == 0:
            preprocessor.logger.error("All texts were removed during preprocessing!")
            preprocessor.logger.info("This might indicate an issue with the input data or preprocessing settings.")
            return
        
        # Validate results
        validation_stats = preprocessor.validate_preprocessing_results(df_processed)
        
        # Log validation statistics
        preprocessor.logger.info("Validation Statistics:")
        for key, value in validation_stats.items():
            preprocessor.logger.info(f"  {key}: {value}")
        
        # Save processed data
        df_processed.to_csv(output_path, index=False)
        preprocessor.logger.info(f"Saved processed data to {output_path}")
        
        # Final statistics
        preprocessor.logger.info("Processing Statistics:")
        for key, value in preprocessor.stats.items():
            preprocessor.logger.info(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Original rows: {len(df)}")
        print(f"Final rows: {len(df_processed)}")
        print(f"Data saved to: {output_path}")
        print("=" * 60)
        
    except FileNotFoundError:
        preprocessor.logger.error(f"Input file not found: {input_path}")
        print(f"Error: Could not find {input_path}")
        print("Please run the data collection script first.")
        
    except Exception as e:
        preprocessor.logger.error(f"Preprocessing failed: {e}")
        print(f"Error during preprocessing: {e}")


if __name__ == '__main__':
    main()

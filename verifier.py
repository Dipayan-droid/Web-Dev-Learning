"""
Kharagpur Data Science Hackathon 2026 - Track A
Narrative Claim Verification System
Team- hack Strike
This script verifies character backstory claims against novel texts.

Requirements:
    pip install openai pandas

Usage:
    python claim_verifier.py --data_dir ./data --output submission.csv

Data directory should contain:
    - In search of the castaways.txt (or similar)
    - The Count of Monte Cristo.txt (or similar)
    - train.csv
    - test.csv

Environment Variables:
    PPLX_API_KEY - Perplexity API key (required)
"""

import os
import sys
import argparse
import json
import re
import time
from typing import List, Tuple, Dict
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    print("Please run: pip install openai pandas")
    sys.exit(1)


class ClaimVerifier:
    """
    Verifies character backstory claims against novel texts.
    
    Uses hierarchical keyword-based retrieval and LLM reasoning
    to determine if claims are consistent or contradict the source text.
    """
    
    def __init__(self, novels: Dict[str, str], api_key: str):
        """
        Initialize the verifier.
        
        Args:
            novels: Dictionary mapping book names to full text
            api_key: Perplexity API key
        """
        self.novels = novels
        self.client = OpenAI(
            api_key=api_key,#api key not provided, ours exhausted
            base_url="https://api.perplexity.ai"
        )
        
        self.novel_lookup = {}
        for key in novels.keys():
            self.novel_lookup[key.lower()] = key
            self.novel_lookup[key.title().lower()] = key
            self.novel_lookup[key.replace(' ', '').lower()] = key
        
        print(f"✓ Verifier initialized with {len(novels)} novels")
        print(f"  Available: {list(novels.keys())}")
    
    def get_novel(self, book_name: str) -> Tuple[str, str]:
        """
        Get novel text with fuzzy matching.
        
        Args:
            book_name: Name of the book to retrieve
            
        Returns:
            Tuple of (novel_text, matched_name) or (None, None) if not found
        """
        # Try exact match
        if book_name in self.novels:
            return self.novels[book_name], book_name
        
        # Try case-insensitive
        key_lower = book_name.lower()
        if key_lower in self.novel_lookup:
            matched_key = self.novel_lookup[key_lower]
            return self.novels[matched_key], matched_key
        
        # Try partial matching
        for novel_key in self.novels.keys():
            if novel_key.lower() in key_lower or key_lower in novel_key.lower():
                return self.novels[novel_key], novel_key
        
        return None, None
    
    def chunk_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def find_relevant_passages(self, 
                               novel: str, 
                               character: str, 
                               claim: str, 
                               top_k: int = 12) -> List[str]:
        """
        Find relevant passages using keyword matching.
        
        Args:
            novel: Full novel text
            character: Character name
            claim: Claim to verify
            top_k: Number of passages to return
            
        Returns:
            List of relevant text passages
        """
        chunks = self.chunk_text(novel, 1500)
        
        # Build keywords
        keywords = []
        if character and pd.notna(character):
            keywords.append(character.lower())
        
        if claim and pd.notna(claim):
            words = claim.lower().split()
            keywords.extend([w for w in words if len(w) > 4][:10])
        
        # Score chunks by keyword frequency
        scored = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(chunk_lower.count(kw) for kw in keywords)
            
            # Bonus for character name
            if character and character.lower() in chunk_lower:
                score += 10
            
            if score > 0:
                scored.append((score, chunk))
        
        # Return top-k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored[:top_k]]
    
    def verify_claim(self, 
                     book_name: str, 
                     character: str, 
                     caption: str, 
                     claim: str) -> Tuple[str, str]:
        """
        Verify if a claim is consistent with the novel.
        
        Args:
            book_name: Name of the book
            character: Character name
            caption: Topic/caption of the claim
            claim: Claim text to verify
            
        Returns:
            Tuple of (prediction, reasoning)
            prediction: 'consistent' or 'contradict'
            reasoning: Brief explanation
        """
        # Handle NaN values
        character = str(character) if pd.notna(character) else "Unknown"
        caption = str(caption) if pd.notna(caption) else "General"
        claim = str(claim) if pd.notna(claim) else ""
        
        if not claim or claim == 'nan':
            return 'contradict', 'Empty claim'
        
        # Get novel
        novel, matched_name = self.get_novel(book_name)
        
        if not novel:
            print(f"    WARNING: Novel not found: '{book_name}'")
            return 'contradict', f'Novel not found'
        
        # Find relevant passages
        passages = self.find_relevant_passages(novel, character, claim, top_k=12)
        
        if not passages:
            # Fallback to first chunks
            all_chunks = self.chunk_text(novel, 1500)
            passages = all_chunks[:8]
        
        # Prepare evidence
        evidence = "\n\n".join(passages[:10])
        
        # Create verification prompt
        prompt = f"""You are verifying a character backstory claim against a novel.

BOOK: {book_name}
CHARACTER: {character}
TOPIC: {caption}

CLAIM TO VERIFY:
{claim}

EVIDENCE FROM NOVEL:
{evidence[:22000]}

TASK: Determine if this claim is CONSISTENT or CONTRADICTS the novel.

CRITICAL INSTRUCTIONS:
1. CONSISTENT means:
   - The claim aligns with facts stated in the novel
   - The claim is plausible given what's shown (even if not explicitly stated)
   - The claim adds reasonable interpretation or detail not contradicted by the text
   - If the novel doesn't mention something, but it COULD be true → CONSISTENT

2. CONTRADICT means:
   - The claim states facts that are EXPLICITLY CONTRADICTED by the novel
   - The claim makes something IMPOSSIBLE given established facts
   - The claim changes core facts: wrong names, wrong relationships, impossible timeline
   - There is CLEAR, DIRECT textual evidence proving the claim wrong

3. When in doubt → CONSISTENT (benefit of the doubt)

4. Don't contradict just because:
   - The claim adds extra detail not in the novel
   - The claim interprets character motivations
   - The claim is about backstory not explicitly covered

ONLY mark as CONTRADICT if there's clear evidence the claim is WRONG.

Respond ONLY in JSON:
{{"prediction": "consistent" or "contradict", "reasoning": "brief explanation (max 100 chars)"}}"""

        try:
            # Call Perplexity API
            response = self.client.chat.completions.create(
                model="sonar-pro",#we used sonar pro, credits exhausted
                messages=[
                    {
                        "role": "system",
                        "content": "You verify claims against novels. Be balanced and fair. Only contradict when there's clear evidence the claim is wrong."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                prediction = result.get('prediction', '').lower()
                reasoning = result.get('reasoning', '')[:100]
                
                if prediction not in ['consistent', 'contradict']:
                    # Fallback parsing
                    text_lower = response_text.lower()
                    consistent_count = text_lower.count('consistent')
                    contradict_count = text_lower.count('contradict')
                    
                    prediction = 'consistent' if consistent_count > contradict_count else 'contradict'
                
                return prediction, reasoning
            else:
                # Manual text parsing
                text_lower = response_text.lower()
                
                if 'clear contradiction' in text_lower or 'explicitly contradicts' in text_lower:
                    return 'contradict', 'Clear contradiction found'
                elif 'consistent' in text_lower or 'plausible' in text_lower:
                    return 'consistent', 'Aligns with novel'
                else:
                    return 'consistent', 'No clear contradiction found'
                
        except Exception as e:
            print(f"    ERROR in API call: {str(e)[:100]}")
            # Conservative fallback
            if len(passages) > 5:
                return 'consistent', 'Error but evidence exists'
            else:
                return 'contradict', f'Error: {str(e)[:50]}'
    
    def process_dataframe(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire test dataframe.
        
        Args:
            test_df: Test dataframe with columns: id, book_name, char, caption, content
            
        Returns:
            DataFrame with predictions
        """
        results = []
        total = len(test_df)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {total} TEST EXAMPLES")
        print(f"{'='*70}\n")
        
        for idx, row in test_df.iterrows():
            print(f"[{idx+1}/{total}] ID: {row['id']} - {row['char']}")
            
            try:
                prediction, reasoning = self.verify_claim(
                    row['book_name'],
                    row['char'],
                    row['caption'],
                    row['content']
                )
                
                results.append({
                    'id': row['id'],
                    'prediction': prediction,
                    'reasoning': reasoning
                })
                
                print(f"  → {prediction.upper()}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'id': row['id'],
                    'prediction': 'contradict',
                    'reasoning': f'Processing error'
                })
        
        return pd.DataFrame(results)


def load_novels(data_dir: Path) -> Dict[str, str]:
    """
    Load all novel text files from data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary mapping book names to full text
    """
    novels = {}
    
    print(f"\nLoading novels from: {data_dir}")
    
    txt_files = list(data_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"ERROR: No .txt files found in {data_dir}")
        return novels
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean name
            clean_name = txt_file.stem.replace(' (1)', '').strip()
            novels[clean_name] = content
            
            print(f"  ✓ Loaded: {txt_file.name} ({len(content):,} chars)")
            
        except Exception as e:
            print(f"  ✗ Error loading {txt_file.name}: {e}")
    
    return novels


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Verify character backstory claims against novels'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory containing novels and CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='Perplexity API key (or set PPLX_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('PPLX_API_KEY')
    if not api_key:
        print("ERROR: Perplexity API key not provided")
        print("Set PPLX_API_KEY environment variable or use --api_key argument")
        sys.exit(1)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("="*70)
    print("KHARAGPUR DATA SCIENCE HACKATHON 2026 - TRACK A")
    print("Narrative Claim Verification System")
    print("="*70)
    
    # Load data
    print("\n1. Loading data files...")
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    if not test_path.exists():
        print(f"ERROR: test.csv not found in {data_dir}")
        sys.exit(1)
    
    test_df = pd.read_csv(test_path)
    print(f"  ✓ Loaded test.csv: {len(test_df)} examples")
    
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        print(f"  ✓ Loaded train.csv: {len(train_df)} examples")
        print(f"  Training distribution: {train_df['label'].value_counts().to_dict()}")
    
    # Load novels
    print("\n2. Loading novels...")
    novels = load_novels(data_dir)
    
    if not novels:
        print("ERROR: No novels loaded")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(novels)} novel(s)")
    
    # Initialize verifier
    print("\n3. Initializing verifier...")
    verifier = ClaimVerifier(novels, api_key)
    
    # Process test data
    print("\n4. Processing test data...")
    results_df = verifier.process_dataframe(test_df)
    
    # Create submission
    print("\n5. Creating submission file...")
    
    submission_df = test_df.merge(
        results_df[['id', 'prediction']],
        on='id',
        how='left'
    ).rename(columns={'prediction': 'label'})
    
    # Fill any missing
    submission_df['label'] = submission_df['label'].fillna('contradict')
    
    # Save
    output_path = Path(args.output)
    submission_df.to_csv(output_path, index=False)
    
    # Save detailed results
    details_path = output_path.parent / f"results_detailed.csv"
    results_df.to_csv(details_path, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    consistent_count = sum(submission_df['label'] == 'consistent')
    contradict_count = sum(submission_df['label'] == 'contradict')
    consistent_pct = consistent_count / len(submission_df) * 100
    
    print(f"\nTotal examples: {len(submission_df)}")
    print(f"Consistent: {consistent_count} ({consistent_pct:.1f}%)")
    print(f"Contradict: {contradict_count} ({100-consistent_pct:.1f}%)")
    
    if train_path.exists():
        train_pct = (train_df['label'] == 'consistent').mean() * 100
        diff = abs(consistent_pct - train_pct)
        print(f"\nTraining distribution: {train_pct:.1f}% consistent")
        print(f"Difference: {diff:.1f}%")
        
        if diff < 10:
            print("✅ Distribution looks good!")
        else:
            print("⚠️ Large difference from training distribution")
    
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"✓ Detailed results saved to: {details_path}")
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
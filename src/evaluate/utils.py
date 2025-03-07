import tiktoken
import warnings
from transformers import AutoTokenizer

# Global tokenizer cache
_tokenizer_cache = {}

# Load the tokenizer for GPT-4o lazily
def _get_gpt4_tokenizer():
    if 'gpt-4o' not in _tokenizer_cache:
        _tokenizer_cache['gpt-4o'] = tiktoken.encoding_for_model("gpt-4o")
    return _tokenizer_cache['gpt-4o']

def get_tokenizer(model_name):
    """Get the appropriate tokenizer based on model name with lazy loading.
    
    Args:
        model_name (str): Name of the model to use for tokenization
    
    Returns:
        tokenizer: Either a tiktoken.Encoding or transformers.PreTrainedTokenizer instance
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
        
    if model_name.startswith(('gpt-', 'claude-')):
        return _get_gpt4_tokenizer()
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            _tokenizer_cache[model_name] = tokenizer
            return tokenizer
        except Exception as e:
            warnings.warn(f"Failed to load HuggingFace tokenizer for {model_name}, falling back to GPT-4 tokenizer. Error: {e}")
            return _get_gpt4_tokenizer()

def truncate_text(text, max_tokens, output_len, model_name="gpt-4o"):
    """Truncate text to fit within max_tokens limit.
    
    Args:
        text (str): Text to truncate
        max_tokens (int): Maximum number of tokens allowed
        output_len (int): Expected length of the output in tokens
        model_name (str): Name of the model to use for tokenization
    """
    tokenizer = get_tokenizer(model_name)
    
    # Handle different tokenizer types
    if isinstance(tokenizer, tiktoken.Encoding):
        tokens = tokenizer.encode(text)
    else:  # HuggingFace tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
    limit = max_tokens - output_len
    if len(tokens) > limit:
        tokens = tokens[:limit]  # Truncate to max length
        warnings.warn(f"Truncated text to {limit} tokens.")
        
    # Decode based on tokenizer type
    if isinstance(tokenizer, tiktoken.Encoding):
        return tokenizer.decode(tokens)
    else:  # HuggingFace tokenizer
        return tokenizer.decode(tokens, skip_special_tokens=True)

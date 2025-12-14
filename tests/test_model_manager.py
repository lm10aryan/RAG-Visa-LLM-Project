
from src.reasoning.model_manager import ModelManager
import os

print('1. API key exists:', bool(os.getenv('GROQ_API_KEY')))
print('2. API key length:', len(os.getenv('GROQ_API_KEY', '')))

try:
    m = ModelManager()
    print('3. ModelManager initialized')
    
    result = m.test_connection()
    print('4. Connection result:', result)
    
except Exception as e:
    print('ERROR:', e)
    import traceback
    traceback.print_exc()

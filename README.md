# Financial-News-Analyzer
    

## Coding style
1. add this comment above each class and function definition to make others easy to understand.  
2. import typing to mark the return type.
```
'''
    function: crawl text or model for what
    input: input variables, types and what it is for
    output: return output, types and what it is for
'''
```

3. add "##" for complex variables or implemention details.  
4. and "#" is for temporary comments like test the function, would be deleted afterwards.  
5. add newline after a specific code part
```
## this is created for ...
# try this
```

 for example:
```
# this one need pip install
import translators as ts
from typing import str

'''
    function: api for translating english phrase to taiwanese
    input: str, the english phrase need to be translated
    output: str, the translated sentences in taiwanese
'''
def translate_en2tw(phrase) -> str:
    return ts.google(phrase, from_language='en', to_language='zh-TW')

## check the translation result
phrase = 'The quick brown fox jumps over the lazy dog.'
print(translate_en2tw(phrase))

```
## Environment
when you use packages not in requirements.txt, add the package names into 'requirements.txt' and push to guthub to make sure everyone is in the same environment.  
install requirements to ensure you can use all packages in this project
```
pip install -r requirements.txt
```
you can also build a virtual environment for this project
```
python -m venv dla-env

## if Unix, activate like this
source dla-env/bin/activate

## if Windows
dla-env\Scripts\activate.bat

## install
pip install -r requirements.txt

## deactivate to return to your own env
deactivate
    
```

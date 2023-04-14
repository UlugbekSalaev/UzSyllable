# UzSyllable

https://pypi.org/project/UzSyllable <br>
https://github.com/UlugbekSalaev/UzSyllable

UzSyllable tool is focused to make division of syllables and end of line breaks of Uzbek language. The tool includes Syllabification, End-of-Line, Count of Syllables methods.
It is created as a python library and uploaded to [PyPI](https://pypi.org/). It is simply easy to use in your python project or other programming language projects via the API. 

## About project
The UzSyllable project is a text processing tool that includes three main methods: syllabification, end-of-line, and count of syllables. Syllabification involves dividing words in Uzbek text into their constituent syllables, which can be useful for pronunciation, spelling, and linguistic analysis. End-of-line justification involves determining the appropriate places to break lines in Uzbek text, which can improve the readability and aesthetics of written materials. Count of syllables involves counting the number of syllables in a given word or sentence, which can be useful for metrics such as rhyme and meter in poetry or for determining the complexity of a text. The UzSyllable project uses machine learning algorithms and linguistic rules to perform these methods accurately and efficiently on Uzbek text.
## Quick links

- [Github](https://github.com/UlugbekSalaev/UzSyllable)
- [PyPI](https://pypi.org/project/UzSyllable/)
- [Web-UI](https://nlp.urdu.uz/?menu=uzsyllable)

## Demo

You can use [web interface](http://nlp.urdu.uz/?menu=uzsyllable).

## Features

- Syllabification
- End-of-Line
- Count of Syllables

## Usage

Three options to run UzSyllable:

- pip
- API 
- Web interface

### pip installation

To install UzSyllable, simply run:

```code
pip install UzSyllable
```

After installation, use in python like following:
```yml
import UzSyllable
# call syllables method
print(UzSyllable.syllables('maktabimda'))
# Output : ['mak-ta-bim-da']
# call end-of-line method
print(UzSyllable.line_break('maktabimda'))
# Output : ['mak-tabimda', 'makta-bimda', 'maktabim-da']
# call count of syllables method
print(UzSyllable.count('maktabimda'))
# Output : 4
```

### API
API configurations: 
 - Method: GET
 - Response type: <code>string</code>


 - URL: https://uz-translit.herokuapp.com/stem
   - Parameters: <code>word:string</code></code>
   - Sample Request: https://uztranslit.herokuapp.com/stem?word=maktabimda


 - https://uz-translit.herokuapp.com/lemmatize
   - Parameters: <code>word:string</code>, <code>pos:string</code>
   - Sample Request: https://uztranslit.herokuapp.com/lemmatize?word=maktabimda&pos=NOUN


 - https://uz-translit.herokuapp.com/analyze
   - Parameters: <code>word:string</code>, <code>pos:string</code>
   - Sample Request: https://uztranslit.herokuapp.com/analyze?word=maktabimda&pos=NOUN

<i>Note: argument <code>pos</code> is optional in all methods</i>
### Web-UI

The web interface created to use easily the library:
You can use web interface [here](http://nlp.urdu.uz/?menu=morphanalyser).

![Demo image](./docs/images/web-interface-ui.png)


### Options
When you use PyPI or API, you should use following options as POS tag of a word which is optional parameters of `lemmatize()` and `analyze()` metods:<br>
    `NOUN`  Noun<br>
    `VERB`  Verb<br>
    `ADJ`   Adjective<br>
    `NUM`   Numerical<br>
    `PRN`   Pronoun<br>
    `ADV`   Adverb

_`pos` parameters is optional for `lemmatize` and `analyze` metods._

### Result Explaining

It returns single word in a string type from each method, `stem` and `lemmatize`, that is stem and lemma of given word, respectively. 
#### Result from `analyze` method
`analyze` method returns a response as list of dictionary which is may contain following keys: 
```yml
 {'word', 'lemma', 'pos', 'affix','affixed','tense','person','cases','singular','plural','question','negative','impulsion','copula','verb_voice','verb_func'}: 
```

## Documentation

See [here](https://github.com/UlugbekSalaev/UzMorphAnalyser).

## Citation

```tex
@misc{UzMorphAnalyser,
  title={{UzMorphAnalyser}: Morphological Analyser Tool for Uzbek},
  url={https://github.com/UlugbekSalaev/UzMorphAnalyser},
  note={Software available from https://github.com/UlugbekSalaev/UzMorphAnalyser},
  author={
    Ulugbek Salaev},
  year={2022},
}
```

## Contact

For help and feedback, please feel free to contact [the author](https://github.com/UlugbekSalaev).
